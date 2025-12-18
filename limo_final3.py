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

        # ---------------- 파라미터 ----------------
        self.forward_speed = 0.12
        self.search_spin_speed = 0.25
        self.k_angle = 0.010
        self.robot_width = 0.13

        # ---------------- LIDAR ----------------
        self.scan_ranges = []
        self.front = 999.0
        self.escape_angle = 0.0

        rospy.loginfo("✅ LIMO FINAL CONTROLLER (LANE+CONE+OBSTACLE) STARTED")

    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.15 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()
        
        # 1. 장애물 회피 상태 머신
        if self.state == "BACK":
            self.back_control(now)
            return
        if self.state == "ESCAPE":
            self.escape_control(now)
            return
        if self.front < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        # 2. 라바콘 감지 시 라바콘 제어 우선
        if self.detect_cone(img):
            self.cone_control(img)
        else:
            # 3. 라바콘이 없으면 정밀 차선 추종
            self.edge_lane_control(img)

    # ============================================================
    # CONE CONTROL (요청하신 라바콘 로직 그대로 적용)
    # ============================================================
    def detect_cone(self, img):
        h, w = img.shape[:2]
        # 하단 45% 영역에서 빨간색 라바콘 탐색
        roi = img[int(h * 0.55):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 빨간색 범위 (두 영역 합침)
        lower_r1, upper_r1 = np.array([0,120,80]), np.array([10,255,255])
        lower_r2, upper_r2 = np.array([170,120,80]), np.array([180,255,255])
        
        mask = cv2.inRange(hsv, lower_r1, upper_r1) | cv2.inRange(hsv, lower_r2, upper_r2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 면적이 200 이상인 것만 유효한 라바콘으로 판단
        self.red_contours = [c for c in contours if cv2.contourArea(c) > 200]
        return len(self.red_contours) > 0

    def cone_control(self, img):
        h, w = img.shape[:2]
        centers = []
        for c in self.red_contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                centers.append(int(M["m10"] / M["m00"]))
        
        if not centers: return

        # 라바콘이 여러 개면 중앙값을, 하나면 그 위치를 타겟으로 설정
        mid = (min(centers) + max(centers)) // 2 if len(centers) >= 2 else centers[0]
        error = mid - (w // 2)
        
        self.current_lin = 0.13
        self.current_ang = error / 180.0 # 라바콘 추종 조향
        rospy.loginfo_throttle(1.0, "🔴 CONE DETECTED - Following...")

    # ============================================================
    # EDGE LANE CONTROL (정밀 가중 평균 로직)
    # ============================================================
    def edge_lane_control(self, img):
        h, w, _ = img.shape
        center = w / 2.0
        roi = img[int(h * 0.5):, :]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        col_sum = np.sum(binary > 0, axis=0)
        max_val = int(np.max(col_sum)) if col_sum.size > 0 else 0

        if max_val < 5:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        threshold_val = max(5, int(max_val * 0.3))
        candidates = np.where(col_sum >= threshold_val)[0]

        if candidates.size == 0:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        x_indices = np.arange(len(col_sum))
        track_center_x = float(np.sum(x_indices[candidates] * col_sum[candidates]) / 
                               np.sum(col_sum[candidates]))

        offset = track_center_x - center
        self.current_ang = np.clip(-self.k_angle * offset, -0.8, 0.8)
        self.current_lin = self.forward_speed

    # ============================================================
    # OBSTACLE ESCAPE (ARC 주행 방식)
    # ============================================================
    def back_control(self, now):
        if now - self.state_start < 1.2:
            self.current_lin, self.current_ang = -0.15, 0.0
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
        smoothed = np.convolve(ranges, np.ones(20)/20, mode='same')
        best_idx = np.argmax(smoothed)
        angle_deg = best_idx - 60
        return (angle_deg + (5 if angle_deg > 0 else -5)) * np.pi / 180.0

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
