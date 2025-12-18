#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2

class LimoFullController:
    def __init__(self):
        rospy.init_node("limo_full_controller")

        # 통신 설정
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)

        self.cmd = Twist()
        self.current_lin = 0.0
        self.current_ang = 0.0
        self.encoding = None

        # 파라미터
        self.forward_speed = 0.12
        self.search_spin_speed = 0.25
        self.k_angle = 0.010

        # 상태 제어
        self.state = "LANE"
        self.state_start = 0.0
        self.front_dist = 999.0
        self.scan_ranges = []
        self.escape_angle = 0.0
        self.red_contours = []

        rospy.loginfo("🚀 라바콘 인식 + 2단계 회피 통합 버전 시작")

    def lidar_callback(self, scan):
        self.scan_ranges = np.array(scan.ranges)
        front_indices = np.concatenate([self.scan_ranges[:15], self.scan_ranges[-15:]])
        cleaned = [d for d in front_indices if d > 0.15 and not np.isnan(d)]
        self.front_dist = np.median(cleaned) if cleaned else 999.0

    def image_callback(self, msg: Image):
        now = rospy.Time.now().to_sec()
        img = self.msg_to_cv2(msg)
        if img is None: return

        # [1. 장애물 회피 상태 우선 순위]
        if self.state == "BACK":
            if now - self.state_start < 1.2:
                self.current_lin, self.current_ang = -0.15, 0.0
            else:
                self.escape_angle = self.find_best_gap_forward()
                self.state = "ESCAPE"
                self.state_start = now
            return

        if self.state == "ESCAPE":
            if now - self.state_start < 1.5:
                self.current_lin = 0.12
                self.current_ang = np.clip(self.escape_angle * 1.8, -0.8, 0.8)
            else:
                self.state = "LANE"
            return

        # [2. 장애물 감지 체크]
        if self.front_dist < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        # [3. 미션 수행: 라바콘 vs 차선]
        # 빨간색 라바콘이 보이면 라바콘 제어, 아니면 차선 제어
        if self.detect_cone(img):
            self.cone_control(img)
        else:
            self.edge_lane_control(img)

    def detect_cone(self, img):
        """HSV 색상 영역을 사용하여 빨간색 라바콘 감지"""
        h, w = img.shape[:2]
        # 화면 하단 절반에서만 탐색 (먼 배경 무시)
        roi = img[int(h * 0.5):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 빨간색은 HSV에서 양 끝단에 위치함
        lower_r1, upper_r1 = np.array([0, 120, 80]), np.array([10, 255, 255])
        lower_r2, upper_r2 = np.array([170, 120, 80]), np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower_r1, upper_r1) | cv2.inRange(hsv, lower_r2, upper_r2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 너무 작은 점(노이즈)은 무시 (크기 200 이상만)
        self.red_contours = [c for c in contours if cv2.contourArea(c) > 200]
        return len(self.red_contours) > 0

    def cone_control(self, img):
        """감지된 라바콘의 중심을 따라 주행"""
        h, w = img.shape[:2]
        centers = []
        for c in self.red_contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                centers.append(int(M["m10"] / M["m00"]))

        if not centers: return
        
        # 라바콘이 여러 개면 그 중간 지점을 목표로 설정
        mid = (min(centers) + max(centers)) // 2 if len(centers) >= 2 else centers[0]
        error = mid - (w // 2)
        
        self.current_lin = 0.13 # 라바콘 주행 시 약간의 속도
        self.current_ang = np.clip(-error / 150.0, -0.8, 0.8)

    def edge_lane_control(self, img):
        """기본 차선 추종 로직"""
        h, w, _ = img.shape
        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        col_sum = np.sum(binary > 0, axis=0)
        if np.max(col_sum) < 5:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        idx = np.where(col_sum >= max(5, int(np.max(col_sum) * 0.3)))[0]
        track_center_x = np.mean(idx)
        offset = track_center_x - (w / 2.0)
        
        self.current_lin = self.forward_speed
        self.current_ang = np.clip(-self.k_angle * offset, -0.8, 0.8)

    def find_best_gap_forward(self):
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]]) # 전방 120도
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        smoothed = np.convolve(ranges, np.ones(20)/20, mode='same')
        best_idx = np.argmax(smoothed)
        angle_deg = (best_idx - 60)
        return (angle_deg + (5 if angle_deg > 0 else -5)) * (np.pi / 180.0)

    def msg_to_cv2(self, msg):
        if self.encoding is None: self.encoding = msg.encoding
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = arr.reshape(msg.height, msg.step // 3, 3)[:, :msg.width, :]
        if self.encoding == "rgb8": img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.cmd.linear.x, self.cmd.angular.z = self.current_lin, self.current_ang
            self.cmd_pub.publish(self.cmd)
            rate.sleep()

if __name__ == "__main__":
    LimoFullController().spin()
