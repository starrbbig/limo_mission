#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2

class LimoArcAvoidance:
    def __init__(self):
        rospy.init_node("limo_arc_avoidance_node")

        # Subscriber & Publisher
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)

        self.cmd = Twist()
        self.current_lin = 0.0
        self.current_ang = 0.0
        self.encoding = None

        # ===== 제어 파라미터 =====
        self.forward_speed = 0.12     # 기본 주행 속도
        self.search_spin_speed = 0.25 # 차선 상실 시 회전 속도
        self.k_angle = 0.010          # 차선 추종 감도

        # ===== 상태 제어 변수 (LANE -> BACK -> ESCAPE) =====
        self.state = "LANE"
        self.state_start = 0.0
        self.front_dist = 999.0
        self.scan_ranges = []
        self.escape_angle = 0.0

        rospy.loginfo("🚀 2단계 곡선 회피(Arc Avoidance) 버전 시작")

    def lidar_callback(self, scan):
        self.scan_ranges = np.array(scan.ranges)
        # 정면 감지 (좌우 15도씩 총 30도)
        front_indices = np.concatenate([self.scan_ranges[:15], self.scan_ranges[-15:]])
        cleaned = [d for d in front_indices if d > 0.15 and not np.isnan(d)]
        self.front_dist = np.median(cleaned) if cleaned else 999.0

    def image_callback(self, msg: Image):
        now = rospy.Time.now().to_sec()

        # [1. 장애물 회피 로직 - 2단계 모드]
        
        # 1-1. 후진 (BACK)
        if self.state == "BACK":
            if now - self.state_start < 1.2: 
                self.current_lin = -0.15
                self.current_ang = 0.0
            else:
                # 후진 완료 후, 전방 120도 범위 내에서만 최적의 각도 탐색
                self.escape_angle = self.find_best_gap_forward()
                self.state = "ESCAPE"
                self.state_start = now
            return

        # 1-2. 곡선 탈출 (ESCAPE: 전진 + 회전 동시)
        if self.state == "ESCAPE":
            # 1.5초간 핸들을 꺾은 채 앞으로 전진하여 장애물을 옆으로 비껴감
            if now - self.state_start < 1.5:
                self.current_lin = 0.12
                # 제자리 회전이 아닌 '주행 중 회전'이므로 배율을 적절히 조절 (1.5 ~ 2.0)
                self.current_ang = np.clip(self.escape_angle * 1.8, -0.8, 0.8)
            else:
                self.state = "LANE"
            return

        # [장애물 감지 트리거]
        if self.front_dist < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        # [2. 기본 차선 인식 주행 (LANE)]
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
        binary = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

        col_sum = np.sum(binary > 0, axis=0)
        max_val = int(np.max(col_sum)) if col_sum.size > 0 else 0

        if max_val < 5:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        threshold_val = max(5, int(max_val * 0.3))
        candidates = np.where(col_sum >= threshold_val)[0]

        if candidates.size > 0:
            x_indices = np.arange(len(col_sum))
            track_center_x = float(np.sum(x_indices[candidates] * col_sum[candidates]) / np.sum(col_sum[candidates]))
            offset = track_center_x - center
            self.current_ang = np.clip(-self.k_angle * offset, -0.8, 0.8)
            self.current_lin = self.forward_speed

    def find_best_gap_forward(self):
        """뒤쪽을 보지 않고 전방 좌우 60도(총 120도) 안에서만 탈출구 탐색"""
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        
        # 뒤쪽 데이터(90~270도)를 완전히 배제하여 한바퀴 도는 현상 방지
        ranges = np.concatenate([raw[-60:], raw[:60]]) 
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        
        # 윈도우 평균으로 안정적인 빈 공간 탐색
        window_size = 20 
        smoothed = np.convolve(ranges, np.ones(window_size)/window_size, mode='same')
        
        best_idx = np.argmax(smoothed)
        angle_deg = (best_idx - 60) # -60 ~ +60도 사이의 결과값
        
        # 장애물로부터 약간 더 멀어지기 위한 보정
        if angle_deg > 0: angle_deg += 5
        else: angle_deg -= 5
            
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
    node = LimoArcAvoidance()
    node.spin()
