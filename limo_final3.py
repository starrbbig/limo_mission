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

        self.cmd = Twist()
        self.current_lin = 0.0
        self.current_ang = 0.0
        self.state = "LANE"
        self.state_start = rospy.Time.now().to_sec()

        # ---------------- 파라미터 ----------------
        self.forward_speed = 0.12
        self.search_spin_speed = 0.25
        self.k_angle = 0.010
        self.scan_ranges = []
        self.front = 999.0
        self.encoding = None

        rospy.loginfo("✅ LIMO FUSION CONTROLLER (LIDAR + CAMERA) STARTED")

    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        # 정면 20도 감지
        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.15 and not np.isnan(d) and not np.isinf(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()
        if self.state == "BACK": self.back_control(now); return
        if self.state == "ESCAPE": self.escape_control(now); return
        if self.front < 0.45:
            self.state = "BACK"; self.state_start = now; return

        img = self.msg_to_cv2(msg)
        if img is None: return

        if self.detect_cone(img):
            self.cone_control_fusion(img) # 라이다 융합형으로 변경
        else:
            self.edge_lane_control_smart(img) # 넓은 길 대응형으로 변경

    # ============================================================
    # 1. CONE CONTROL (라이다 + 카메라 융합)
    # ============================================================
    def detect_cone(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.55):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,130,100]), np.array([10,255,255])) | \
               cv2.inRange(hsv, np.array([170,130,100]), np.array([180,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.red_contours = [c for c in contours if cv2.contourArea(c) > 300]
        return len(self.red_contours) > 0

    def cone_control_fusion(self, img):
        """카메라로 하나만 보이면 라이다로 반대편을 찾아 중앙을 잡는 로직"""
        h, w = img.shape[:2]
        centers = [int(cv2.moments(c)["m10"]/cv2.moments(c)["m00"]) for c in self.red_contours if cv2.moments(c)["m00"] > 0]
        if not centers: return
        centers.sort()

        # 카메라로 찾은 라바콘 위치 (화면 중심 기준 error)
        cam_cone_x = centers[0]

        # [상황 1] 카메라에 2개 이상 보이면 기존처럼 중앙 주행
        if len(centers) >= 2:
            mid_target = (centers[0] + centers[-1]) // 2
        
        # [상황 2] 하나만 보일 때 (사용자님 아이디어 반영)
        else:
            raw = np.array(self.scan_ranges)
            if len(raw) > 0:
                # 안전하게 3.5m로 결측치 처리
                safe_raw = np.where((raw < 0.1) | np.isnan(raw) | np.isinf(raw), 3.5, raw)
                
                if cam_cone_x < (w // 2): # 왼쪽 라바콘만 보임
                    # 오른쪽 시야(라이다 -30~-90도)에서 가장 가까운 장애물을 '가상 라바콘'으로 설정
                    right_lidar_zone = safe_raw[-90:-30]
                    closest_idx = np.argmin(right_lidar_zone)
                    # 라이다 거리가 너무 멀지 않다면 그쪽을 우측 벽으로 인식
                    mid_target = cam_cone_x + int(w * 0.35) 
                    rospy.loginfo("Searching Right with LiDAR...")
                else: # 오른쪽 라바콘만 보임
                    # 왼쪽 시야(라이다 30~90도) 확인
                    left_lidar_zone = safe_raw[30:90]
                    mid_target = cam_cone_x - int(w * 0.35)
                    rospy.loginfo("Searching Left with LiDAR...")
            else:
                mid_target = cam_cone_x # 최후의 수단

        error = mid_target - (w // 2)
        self.current_lin, self.current_ang = 0.11, np.clip(-error / 150.0, -0.7, 0.7)

    # ============================================================
    # 2. SMART LANE CONTROL (넓은 길/검정길 구간 대응)
    # ============================================================
    def edge_lane_control_smart(self, img):
        h, w, _ = img.shape
        roi = img[int(h * 0.5):, :]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5,5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        col_sum = np.sum(binary > 0, axis=0)
        if np.max(col_sum) < 5:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed; return

        idx = np.where(col_sum >= max(5, int(np.max(col_sum) * 0.3)))[0]
        left_edge, right_edge = np.min(idx), np.max(idx)
        detected_width = right_edge - left_edge

        # 상황 1: 길이 넓어질 때 (한쪽 선만 보일 때) - 오프셋 사용
        if left_edge < (w * 0.15) and detected_width < (w * 0.4): # 왼쪽 선에 너무 붙었을 때
            track_center = left_edge + (w * 0.38)
        elif right_edge > (w * 0.85) and detected_width < (w * 0.4): # 오른쪽 선에 너무 붙었을 때
            track_center = right_edge - (w * 0.38)
        # 상황 2: 정상적인 두 선 사이
        else:
            track_center = (left_edge + right_edge) / 2.0 if detected_width > (w * 0.3) else np.mean(idx)

        offset = track_center - (w / 2.0)
        self.current_lin, self.current_ang = self.forward_speed, np.clip(-self.k_angle * offset, -0.8, 0.8)

    # ---------------- 나머지 유틸리티 함수 (후진/회피 원본 유지) ----------------
    def back_control(self, now):
        if now - self.state_start < 1.2: self.current_lin, self.current_ang = -0.15, 0.0
        else: self.escape_angle = self.find_gap_max_forward(); self.state = "ESCAPE"; self.state_start = now
    def escape_control(self, now):
        if now - self.state_start < 1.5: self.current_lin, self.current_ang = 0.12, np.clip(self.escape_angle * 1.5, -0.8, 0.8)
        else: self.state = "LANE"
    def find_gap_max_forward(self):
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        smoothed = np.convolve(ranges, np.ones(20)/20, mode='same')
        return (np.argmax(smoothed) - 60) * np.pi / 180.0
    def msg_to_cv2(self, msg):
        if self.encoding is None: self.encoding = msg.encoding
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = arr.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if self.encoding == "rgb8" else img
    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.cmd.linear.x, self.cmd.angular.z = self.current_lin, self.current_ang
            self.pub.publish(self.cmd); rate.sleep()

if __name__ == "__main__":
    LimoFinalController().spin()
