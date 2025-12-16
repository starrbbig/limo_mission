#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2

class EdgeLaneNoBridge:
    def __init__(self):
        rospy.init_node("edge_lane_nobridge_node")

        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)

        self.cmd = Twist()
        self.current_lin = 0.0
        self.current_ang = 0.0
        self.encoding = None

        # ===== [기존 파라미터 유지] =====
        self.forward_speed = 0.12 
        self.search_spin_speed = 0.25 
        self.k_angle = 0.010 

        # ===== [상태 제어 변수] =====
        self.state = "LANE"
        self.state_start = 0.0
        self.front_dist = 999.0
        self.scan_ranges = []
        self.escape_angle = 0.0

        rospy.loginfo("🚀 장애물 회피 최종 최적화 버전 시작")

    def lidar_callback(self, scan):
        self.scan_ranges = np.array(scan.ranges)
        # 감지 범위를 정면 좁게 설정하여 옆 장애물에 간섭받지 않게 함
        front_indices = np.concatenate([self.scan_ranges[:10], self.scan_ranges[-10:]])
        cleaned = [d for d in front_indices if d > 0.15 and not np.isnan(d)]
        self.front_dist = np.median(cleaned) if cleaned else 999.0

    def image_callback(self, msg: Image):
        now = rospy.Time.now().to_sec()

        # [1. 장애물 회피 로직]
        
        # 후진
        if self.state == "BACK":
            if now - self.state_start < 1.3:
                self.current_lin = -0.15
                self.current_ang = 0.0
            else:
                self.current_lin = 0.0
                self.escape_angle = self.find_best_gap()
                self.state = "ESCAPE_TURN"
                self.state_start = now
            return

        # 제자리 회전 (빈 공간 정확히 조준)
        if self.state == "ESCAPE_TURN":
            if now - self.state_start < 1.0:
                self.current_lin = 0.0
                self.current_ang = np.clip(self.escape_angle * 2.5, -1.2, 1.2)
            else:
                self.state = "ESCAPE_STRAIGHT"
                self.state_start = now
            return

        # [핵심 수정] 탈출 직진 유지 (장애물을 완전히 지나칠 때까지 차선 인식 차단)
        if self.state == "ESCAPE_STRAIGHT":
            # 2.0초로 연장하여 장애물 영향권을 확실히 벗어남
            if now - self.state_start < 2.0:
                self.current_lin = 0.12
                self.current_ang = 0.0 # 조향 개입 차단 (똑바로 가기)
            else:
                self.state = "LANE"
                rospy.loginfo("🏁 탈출 완료, 차선 추종 복귀")
            return

        # [감지] 정면 장애물 발견 시 회피 시작
        if self.front_dist < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        # [2. 기존 차선 인식 로직 - 원본 보존]
        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        h, w, _ = img.shape
        center = w / 2.0
        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
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
        track_center_x = float(np.sum(x_indices[candidates] * col_sum[candidates]) / np.sum(col_sum[candidates]))

        offset = track_center_x - center
        ang = -self.k_angle * offset
        self.current_lin = self.forward_speed
        self.current_ang = np.clip(ang, -0.8, 0.8)

    def find_best_gap(self):
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-90:], raw[:90]])
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        
        window_size = 35 # 틈새를 더 엄격하게 필터링
        smoothed = np.convolve(ranges, np.ones(window_size)/window_size, mode='same')
        
        best_idx = np.argmax(smoothed)
        angle_deg = (best_idx - 90)
        # 안전 여유를 더 늘려 장애물 사이 정중앙을 정확히 찌르게 함
        return angle_deg * (np.pi / 180.0)

    def msg_to_cv2(self, msg):
        if self.encoding is None: self.encoding = msg.encoding
        h, w = msg.height, msg.width
        if self.encoding in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = arr.reshape(h, msg.step // 3, 3)[:, :w, :]
            if self.encoding == "rgb8": img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        return None

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.cmd.linear.x = self.current_lin
            self.cmd.angular.z = self.current_ang
            self.cmd_pub.publish(self.cmd)
            rate.sleep()

if __name__ == "__main__":
    node = EdgeLaneNoBridge()
    node.spin()
