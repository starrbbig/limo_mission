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
        self.encoding = None

        # 주행 파라미터
        self.forward_speed = 0.15
        self.search_spin_speed = 0.25
        self.k_angle = 0.010
        self.scan_ranges = []
        self.front = 999.0
        self.escape_angle = 0.0

        rospy.loginfo("✅ LIMO FINAL (REVERSE-RIGHT ESCAPE) READY")

    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        if len(raw) > 0:
            front_zone = np.concatenate([raw[:10], raw[-10:]])
            cleaned = [d for d in front_zone if d > 0.15 and not np.isnan(d) and not np.isinf(d)]
            self.front = np.median(cleaned) if cleaned else 999.0

    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()
        if self.state == "BACK":
            self.back_control(now)
            return
        if self.state == "ESCAPE":
            self.escape_control(now)
            return

        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        # 장애물 거리 체크 (라바콘 감지 시 더 조심)
        obs_limit = 0.35 if self.detect_cone(img) else 0.45
        if self.front < obs_limit:
            self.state = "BACK"
            self.state_start = now
            return

        if self.detect_cone(img):
            self.cone_control(img)
        else:
            self.edge_lane_control(img)

    def back_control(self, now):
        if now - self.state_start < 1.2:
            self.current_lin, self.current_ang = -0.15, 0.0
        else:
            # 후진 끝난 후, 이미 설정된 escape_angle이 없다면(라이다 기반) 각도 계산
            if self.escape_angle == 0.0:
                self.escape_angle = self.find_gap_max_forward()
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self, now):
        if now - self.state_start < 1.5:
            self.current_lin = 0.12
            self.current_ang = np.clip(self.escape_angle * 1.5, -0.8, 0.8)
        else:
            self.state = "LANE"
            self.escape_angle = 0.0 # 초기화

    def find_gap_max_forward(self):
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        smoothed = np.convolve(ranges, np.ones(20)/20, mode='same')
        penalty = 1.0 - (np.abs(np.arange(120) - 60) / 120.0) * 0.5
        best_idx = np.argmax(smoothed * penalty)
        return np.clip((best_idx - 60), -40, 40) * np.pi / 180.0

    # ------------------ 핵심: 차선 중심 추출 함수 ------------------
    def get_lane_center(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.5):, :]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5,5), 0)
        # [수정] 고정값 60 사용하여 흰 바닥 무시
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        col_sum = np.sum(binary > 0, axis=0)
        if np.max(col_sum) < 5: return None
        idx = np.where(col_sum >= max(5, int(np.max(col_sum) * 0.3)))[0]
        return np.mean(idx)

    def detect_cone(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.4):, :] # 조금 더 위쪽까지 봄
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])) | \
               cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.red_contours = [c for c in contours if cv2.contourArea(c) > 150]
        return len(self.red_contours) > 0

    def cone_control(self, img):
        h, w = img.shape[:2]
        centers = [int(cv2.moments(c)["m10"]/cv2.moments(c)["m00"]) for c in self.red_contours if cv2.moments(c)["m00"] > 0]
        if not centers: return

        # [수정] 라바콘 1개 감지 시: 즉시 후진 후 오른쪽 회피 예약
        if len(centers) == 1:
            rospy.logwarn("⚠️ Cone x1: Reversing & Aiming RIGHT")
            self.state = "BACK"
            self.state_start = rospy.Time.now().to_sec()
            self.escape_angle = 35.0 * np.pi / 180.0 # 35도 우측 고정
            return

        # 정상 주행 (2개 이상)
        mid = (min(centers) + max(centers)) // 2
        lane_c = self.get_lane_center(img)
        if lane_c is not None:
            mid = (mid * 0.7) + (lane_c * 0.3)
        
        error = mid - (w // 2)
        self.current_lin, self.current_ang = 0.10, np.clip(-error / 150.0, -0.5, 0.5)

    def edge_lane_control(self, img):
        lane_c = self.get_lane_center(img)
        if lane_c is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return
        offset = lane_c - (img.shape[1] / 2.0)
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
