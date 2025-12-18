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

        # ===== [상태 제어 변수 수정] =====
        # LANE -> BACK -> ESCAPE_TURN -> ESCAPE_STRAIGHT 순서로 동작
        self.state = "LANE"
        self.state_start = 0.0
        self.front_dist = 999.0
        self.scan_ranges = []
        self.escape_angle = 0.0

        rospy.loginfo("🚀 장애물 회피 강화 버전(2단계 탈출) 시작")

    def lidar_callback(self, scan):
        self.scan_ranges = np.array(scan.ranges)
        front_indices = np.concatenate([self.scan_ranges[:15], self.scan_ranges[-15:]])
        cleaned = [d for d in front_indices if d > 0.15 and not np.isnan(d)]
        self.front_dist = np.median(cleaned) if cleaned else 999.0

    def image_callback(self, msg: Image):
        now = rospy.Time.now().to_sec()

        # [1. 장애물 회피 로직 - 강화됨]
        
        # 1-1. 후진
        if self.state == "BACK":
            if now - self.state_start < 1.3: # 후진 시간 약간 늘림
                self.current_lin = -0.15
                self.current_ang = 0.0
            else:
                self.current_lin = 0.0
                self.escape_angle = self.find_best_gap()
                self.state = "ESCAPE_TURN" # 바로 안 나가고 '회전'부터 함
                self.state_start = now
            return

        # 1-2. 제자리 회전 (장애물 없는 쪽으로 고개 돌리기)
        if self.state == "ESCAPE_TURN":
            if now - self.state_start < 1.0: # 1초간 제자리 회전
                self.current_lin = 0.0
                # 각도 배율을 높여 더 확실하게 꺾음
                self.current_ang = np.clip(self.escape_angle * 2.5, -1.2, 1.2)
            else:
                self.state = "ESCAPE_STRAIGHT" # 이제 앞으로 나감
                self.state_start = now
            return

        # 1-3. 전방 주행 (장애물 옆 통과하기)
        if self.state == "ESCAPE_STRAIGHT":
            if now - self.state_start < 1.2: # 1.2초간 장애물 옆을 지나침
                self.current_lin = 0.12
                self.current_ang = 0.0 # 직진해서 옆구리 안 걸리게 함
            else:
                self.state = "LANE"
            return

        # [감지] 
        if self.front_dist < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

    def detect_cone(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.55):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_r1 = np.array([0, 120, 80])
        upper_r1 = np.array([10, 255, 255])
        lower_r2 = np.array([170, 120, 80])
        upper_r2 = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower_r1, upper_r1) | \
               cv2.inRange(hsv, lower_r2, upper_r2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.red_contours = [c for c in contours if cv2.contourArea(c) > 200]
        return len(self.red_contours) > 0

      
        
    def cone_control(self, img):
        h, w = img.shape[:2]
        centers = []

        for c in self.red_contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                centers.append(int(M["m10"] / M["m00"]))

        if not centers:
            return

        if len(centers) >= 2:
            mid = (min(centers) + max(centers)) // 2
        else:
            mid = centers[0]

        error = mid - (w // 2)
        self.current_lin = 0.13
        self.current_ang = error / 180.0
    

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
        
        # 윈도우 사이즈를 늘려(30) 더 넓은 공간을 찾게 함
        window_size = 30 
        smoothed = np.convolve(ranges, np.ones(window_size)/window_size, mode='same')
        
        best_idx = np.argmax(smoothed)
        # 선택된 각도에서 바깥쪽으로 5도 정도 더 여유를 줌 (장애물에서 멀어지게)
        angle_deg = (best_idx - 90)
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
    node = EdgeLaneNoBridge()
    node.spin()
