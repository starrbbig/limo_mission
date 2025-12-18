#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

class LimoFinalController:
    def __init__(self):
        rospy.init_node("limo_final_controller")

        # ---------------- ROS ----------------
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_cb, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb, queue_size=1)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # ---------------- CMD ----------------
        self.cmd = Twist()
        self.current_lin = 0.0
        self.current_ang = 0.0

        # ---------------- STATE ----------------
        self.state = "LANE"     # LANE / BACK / ESCAPE
        self.state_start = rospy.Time.now().to_sec()

        # ---------------- IMAGE ----------------
        self.encoding = None

        # ---------------- LANE 파라미터 ----------------
        self.forward_speed = 0.12
        self.search_spin_speed = 0.25
        self.k_angle = 0.010

        # ---------------- LIDAR 파라미터 ----------------
        self.scan_ranges = []
        self.front = 999.0
        self.escape_angle = 0.0
        self.robot_width = 0.13

        rospy.loginfo("✅ LIMO FINAL CONTROLLER (STABLE CONE VER.) STARTED")

    # ============================================================
    # LIDAR
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        if len(raw) > 0:
            front_zone = np.concatenate([raw[:10], raw[-10:]])
            cleaned = [d for d in front_zone if d > 0.15 and not np.isnan(d)]
            self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # IMAGE CALLBACK (STATE MACHINE)
    # ============================================================
    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()
        
        if self.state == "BACK":
            self.back_control(now)
            return

        if self.state == "ESCAPE":
            self.escape_control(now)
            return

        # 장애물 감지 트리거 (라바콘 진입 시 임계값 완화)
        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        obs_dist = 0.35 if self.detect_cone(img) else 0.45
        if self.front < obs_dist:
            self.state = "BACK"
            self.state_start = now
            return

        if self.detect_cone(img):
            self.cone_control(img)
        else:
            self.edge_lane_control(img)

    # ============================================================
    # BACK / ESCAPE
    # ============================================================
    def back_control(self, now):
        if now - self.state_start < 1.2:
            self.current_lin = -0.15
            self.current_ang = 0.0
        else:
            self.escape_angle = self.find_gap_max_forward()
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self, now):
        if now - self.state_start < 1.5:
            self.current_lin = 0.12
            self.current_ang = np.clip(self.escape_angle * 1.5, -0.8, 0.8)
        else:
            self.state = "LANE"

    def find_gap_max_forward(self):
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        window_size = 20
        smoothed = np.convolve(ranges, np.ones(window_size)/window_size, mode='same')
        best_idx = np.argmax(smoothed)
        angle_deg = best_idx - 60
        safe_margin = 5 if angle_deg > 0 else -5
        return (angle_deg + safe_margin) * np.pi / 180.0

    # ============================================================
    # CONE / LANE (보정 및 이탈 방지 적용)
    # ============================================================
    def detect_cone(self, img):
        h, w = img.shape[:2]
        # [수정] ROI 확장: 0.55 -> 0.4 (멀리 있는 콘 미리 인식)
        roi = img[int(h * 0.4):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])) | \
               cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.red_contours = [c for c in contours if cv2.contourArea(c) > 150]
        return len(self.red_contours) > 0

    def get_lane_center(self, img):
        """차선 이탈 방지를 위한 차선 중심점 반환"""
        h, w = img.shape[:2]
        roi = img[int(h * 0.5):, :]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5,5), 0)
        # [수정] 고정 임계값 60 적용 (흰색 바닥 무시)
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        col_sum = np.sum(binary > 0, axis=0)
        if np.max(col_sum) < 5: return None
        idx = np.where(col_sum >= max(5, int(np.max(col_sum) * 0.3)))[0]
        return np.mean(idx)

    def cone_control(self, img):
        h, w = img.shape[:2]
        centers = [int(cv2.moments(c)["m10"]/cv2.moments(c)["m00"]) for c in self.red_contours if cv2.moments(c)["m00"] > 0]
        lane_center = self.get_lane_center(img)
        if not centers: return

        # 1. 라바콘 기반 타겟 설정
        if len(centers) >= 2:
            mid = (min(centers) + max(centers)) // 2
        else:
            cone_x = centers[0]
            # [수정] 세이프티 오프셋 140 적용
            mid = (cone_x - 140) if cone_x > (w // 2) else (cone_x + 140)

        # 2. 차선 중심 가중치 섞기 (이탈 방지: 라바콘7 : 차선3)
        if lane_center is not None:
            mid = (mid * 0.7) + (lane_center * 0.3)

        error = mid - (w // 2)
        self.current_lin = 0.10 # 진입 안정성 위해 감속
        self.current_ang = np.clip(-error / 140.0, -0.6, 0.6)

    def edge_lane_control(self, img):
        lane_center = self.get_lane_center(img)
        if lane_center is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return
        
        offset = lane_center - (img.shape[1] / 2.0)
        self.current_lin = self.forward_speed
        self.current_ang = np.clip(-self.k_angle * offset, -0.8, 0.8)

    def msg_to_cv2(self, msg):
        if self.encoding is None: self.encoding = msg.encoding
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = arr.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if self.encoding == "rgb8" else img

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.cmd.linear.x, self.cmd.angular.z = self.current_lin, self.current_ang
            self.pub.publish(self.cmd)
            rate.sleep()

if __name__ == "__main__":
    LimoFinalController().spin()
