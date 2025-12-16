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

        # Subscriber & Publisher
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

        # ===== [장애물 회피용 변수] =====
        self.state = "LANE"
        self.state_start = 0.0
        self.front_dist = 999.0
        self.scan_ranges = []
        self.escape_angle = 0.0

        rospy.loginfo("✅ EdgeLane (Original Logic) + Advanced Obstacle Avoidance Started")

    def lidar_callback(self, scan):
        self.scan_ranges = np.array(scan.ranges)
        # 정면 ±15도 감지
        front_indices = np.concatenate([self.scan_ranges[:15], self.scan_ranges[-15:]])
        cleaned = [d for d in front_indices if d > 0.15 and not np.isnan(d)]
        self.front_dist = np.median(cleaned) if cleaned else 999.0

    def image_callback(self, msg: Image):
        now = rospy.Time.now().to_sec()

        # [1. 장애물 회피 상태 머신 - 이 부분만 추가됨]
        if self.state == "BACK":
            if now - self.state_start < 1.2:
                self.current_lin = -0.15
                self.current_ang = 0.0
            else:
                self.current_lin = 0.0
                # [수정] 가장 넓게 뚫린 방향을 찾아 escape_angle 설정
                self.escape_angle = self.find_best_gap()
                self.state = "ESCAPE"
                self.state_start = now
            return

        if self.state == "ESCAPE":
            if now - self.state_start < 1.8: # 탈출 시간을 넉넉히 주어 장애물 옆을 지나치게 함
                self.current_lin = 0.10
                # 찾은 방향으로 확실히 꺾음
                self.current_ang = np.clip(self.escape_angle * 2.0, -1.0, 1.0)
            else:
                self.state = "LANE"
            return

        # [장애물 감지]
        if self.front_dist < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        # [2. 기존 차선 인식 로직 - 건드리지 않음]
        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        h, w, _ = img.shape
        center = w / 2.0

        # ROI 설정 (하단 50%)
        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold (검은 트랙 강조)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise 제거
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 열별 합계 및 무게중심 계산 (원본 로직 유지)
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

        # 조향 제어 (원본 부호 및 게인 유지)
        offset = track_center_x - center
        ang = -self.k_angle * offset
        self.current_lin = self.forward_speed
        self.current_ang = np.clip(ang, -0.8, 0.8)

    # ----------------------------- #
    # [장애물 회피용 보조 함수]
    # ----------------------------- #
    def find_best_gap(self):
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        # 좌우 90도씩 총 180도 확인
        ranges = np.concatenate([raw[-90:], raw[:90]])
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        
        # 이동 평균 필터: 로봇 몸체가 지나갈 수 있는 '넓은 공간' 우선 순위
        window_size = 25 
        smoothed = np.convolve(ranges, np.ones(window_size)/window_size, mode='same')
        
        best_idx = np.argmax(smoothed)
        return (best_idx - 90) * (np.pi / 180.0)

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
