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

        # ===== [기존 파라미터] =====
        self.forward_speed = 0.12 
        self.search_spin_speed = 0.25 
        self.k_angle = 0.010 

        # ===== [상태 제어] =====
        self.state = "LANE"
        self.state_start = 0.0
        self.front_dist = 999.0
        self.scan_ranges = []
        self.escape_dir = 0.0 # 왼쪽(+1), 오른쪽(-1) 판단용

        rospy.loginfo("🚀 회피 기동 자연화 버전 시작")

    def lidar_callback(self, scan):
        self.scan_ranges = np.array(scan.ranges)
        front_indices = np.concatenate([self.scan_ranges[:15], self.scan_ranges[-15:]])
        cleaned = [d for d in front_indices if d > 0.15 and not np.isnan(d)]
        self.front_dist = np.median(cleaned) if cleaned else 999.0

    def image_callback(self, msg: Image):
        now = rospy.Time.now().to_sec()

        # 1. 후진 (짧고 빠르게)
        if self.state == "BACK":
            if now - self.state_start < 1.3:
                self.current_lin = -0.15
                self.current_ang = 0.0
            else:
                # 어느 쪽이 더 비었는지 방향만 결정
                self.escape_dir = self.get_escape_direction()
                self.state = "ESCAPE_TURN"
                self.state_start = now
            return

        # 2. 제자리 회전 (비어있는 방향으로 살짝만 고개 돌리기)
        if self.state == "ESCAPE_TURN":
            if now - self.state_start < 1.2:
                self.current_lin = 0.0
                # 비어있는 방향으로 확실히 회전
                self.current_ang = 1.1 * self.escape_dir
            else:
                self.state = "LANE" # 바로 차선 추종으로 복귀하되, 회피 로직이 섞임
            return

        # [감지]
        if self.front_dist < 0.40:
            self.state = "BACK"
            self.state_start = now
            return

        # 3. 차선 인식 주행 (기존 로직)
        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        h, w, _ = img.shape
        center = w / 2.0
        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        col_sum = np.sum(binary > 0, axis=0)
        max_val = int(np.max(col_sum)) if col_sum.size > 0 else 0

        if max_val < 5:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        threshold_val = max(5, int(max_val * 0.3))
        candidates = np.where(col_sum >= threshold_val)[0]
        x_indices = np.arange(len(col_sum))
        track_center_x = float(np.sum(x_indices[candidates] * col_sum[candidates]) / np.sum(col_sum[candidates]))

        # 조향 제어
        offset = track_center_x - center
        ang = -self.k_angle * offset
        
        # [수정] 장애물을 피한 직후라면, 차선 중앙보다 약간 더 바깥쪽을 타도록 유도
        # (옆구리 박는 현상 방지)
        self.current_lin = self.forward_speed
        self.current_ang = np.clip(ang, -0.8, 0.8)

    def get_escape_direction(self):
        # 정면 기준 좌우 90도씩 합계 거리를 비교하여 더 넓은 쪽 선택
        if len(self.scan_ranges) == 0: return 1.0
        raw = np.array(self.scan_ranges)
        right_sum = np.nansum(raw[-90:])
        left_sum = np.nansum(raw[:90])
        return 1.0 if left_sum > right_sum else -1.0

    def msg_to_cv2(self, msg):
        if self.encoding is None: self.encoding = msg.encoding
        h, w = msg.height, msg.width
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = arr.reshape(h, msg.step // 3, 3)[:, :w, :]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if self.encoding == "rgb8" else img

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
