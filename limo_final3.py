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

        rospy.loginfo("✅ LIMO FINAL CONTROLLER (WIDE LANE & CONE FIX) STARTED")

    # ============================================================
    # LIDAR (원본 유지: 정면 20도)
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.15 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # IMAGE CALLBACK (원본 유지)
    # ============================================================
    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()
       
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

        if self.detect_cone(img):
            self.cone_control(img)
        else:
            self.edge_lane_control(img)

    # ============================================================
    # BACK / ESCAPE (원본 유지)
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
    # CONE CONTROL (개선: 하나만 보일 때 밖으로 나가는 것 방지)
    # ============================================================
    def detect_cone(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.55):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,120,80]), np.array([15,255,255])) | \
               cv2.inRange(hsv, np.array([165,120,80]), np.array([180,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.red_contours = [c for c in contours if cv2.contourArea(c) > 300]
        return len(self.red_contours) > 0

    def cone_control(self, img):
        """라바콘이 하나만 보여도 그 안쪽 공간을 타겟팅하여 사이 주행 유도"""
        h, w = img.shape[:2]
        centers = [int(cv2.moments(c)["m10"]/cv2.moments(c)["m00"]) for c in self.red_contours if cv2.moments(c)["m00"] > 0]
        if not centers: return
        centers.sort()

        # 1. 라바콘이 2개 이상 보일 때: 두 라바콘의 중앙으로
        if len(centers) >= 2:
            mid_target = (centers[0] + centers[-1]) // 2
        
        # 2. 하나만 보일 때: 라바콘을 화면 가장자리로 밀어내어 사이 공간 확보
        else:
            cone_x = centers[0]
            if cone_x < (w // 2): # 왼쪽 라바콘 발견 -> 그 오른쪽 공간을 타겟
                mid_target = cone_x + int(w * 0.35)
            else:                 # 오른쪽 라바콘 발견 -> 그 왼쪽 공간을 타겟
                mid_target = cone_x - int(w * 0.35)

        error = mid_target - (w // 2)
        # 조향이 너무 급격하지 않게 분모(180) 조절
        self.current_lin, self.current_ang = 0.12, np.clip(-error / 160.0, -0.7, 0.7)

    # ============================================================
    # LANE CONTROL (개선: 넓은 길 오프셋 로직 적용)
    # ============================================================
    def edge_lane_control(self, img):
        h, w, _ = img.shape
        roi = img[int(h * 0.5):, :]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5,5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
       
        col_sum = np.sum(binary > 0, axis=0)
        if np.max(col_sum) < 5:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        # 선의 위치 파악
        idx = np.where(col_sum >= max(5, int(np.max(col_sum) * 0.3)))[0]
        left_edge = np.min(idx)
        right_edge = np.max(idx)
        detected_width = right_edge - left_edge

        # 상황 1: 두 선이 다 보일 정도로 폭이 넓음 -> 사이 중앙값 사용
        if detected_width > (w * 0.45):
            track_center = (left_edge + right_edge) / 2.0
        
        # 상황 2: 왼쪽 선만 보일 때 (넓은 길 진입 전/후) -> 오른쪽으로 오프셋
        elif left_edge < (w * 0.3):
            track_center = left_edge + (w * 0.38) # 화면의 38% 지점으로 밀어줌
            
        # 상황 3: 오른쪽 선만 보일 때 -> 왼쪽으로 오프셋
        elif right_edge > (w * 0.7):
            track_center = right_edge - (w * 0.38)
            
        else:
            track_center = np.mean(idx)

        offset = track_center - (w / 2.0)
        self.current_lin, self.current_ang = self.forward_speed, np.clip(-self.k_angle * offset, -0.8, 0.8)

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
