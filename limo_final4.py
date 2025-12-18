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

        # ---------------- 주행 파라미터 (와리가리 방지 튜닝) ----------------
        self.forward_speed = 0.12
        self.search_spin_speed = 0.25
        self.k_angle = 0.007  # 기존 0.010에서 낮춤 (차선 주행 안정화)

        # ---------------- LIDAR 파라미터 ----------------
        self.scan_ranges = []
        self.front = 999.0
        self.escape_angle = 0.0

        rospy.loginfo("✅ LIMO FINAL (CONE LOGIC UPDATED) STARTED")

    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.15 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()
        
        if self.state == "BACK":
            self.back_control(now)
            return
        if self.state == "ESCAPE":
            self.escape_control(now)
            return

        # 장애물 거리 기반 상태 전환
        if self.front < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        # [수정] 라바콘 감지 시 전용 제어, 아닐 시 차선 제어
        if self.detect_cone(img):
            self.cone_control(img)
        else:
            self.edge_lane_control(img)

    def back_control(self, now):
        if now - self.state_start < 1.2:
            self.current_lin, self.current_ang = -0.15, 0.0
        else:
            # 후진 종료 시, 예약된 escape_angle이 없다면 라이다로 탐색
            if self.escape_angle == 0.0:
                self.escape_angle = self.find_gap_max_forward()
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self, now):
        if now - self.state_start < 1.5:
            self.current_lin = 0.12
            self.current_ang = np.clip(self.escape_angle * 1.5, -0.6, 0.6)
        else:
            self.state = "LANE"
            self.escape_angle = 0.0 # 초기화

    def find_gap_max_forward(self):
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        smoothed = np.convolve(ranges, np.ones(20)/20, mode='same')
        best_idx = np.argmax(smoothed)
        return (best_idx - 60) * np.pi / 180.0

    # ============================================================
    # 💡 핵심 수정: 라바콘 감지 및 제어 로직
    # ============================================================
    def detect_cone(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.45):, :] # 약간 더 멀리서 감지하도록 범위 확장
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 빨간색 라바콘 마스크 (조명 강인성 확보)
        mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) | \
               cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 면적 필터링으로 노이즈 제거
        self.red_contours = [c for c in contours if cv2.contourArea(c) > 180]
        return len(self.red_contours) > 0

    def cone_control(self, img):
        h, w = img.shape[:2]
        centers = []
        for c in self.red_contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                centers.append(int(M["m10"] / M["m00"]))
        
        if not centers: return

        # [로직 1] 라바콘이 1개만 보일 때 -> 오른쪽 30도 회피 모드 진입
        if len(centers) == 1:
            rospy.logwarn("⚠️ 라바콘 1개 감지! 오른쪽으로 30도 회피 시퀀스 시작")
            self.state = "BACK" # 안전을 위해 살짝 후진 후 회전
            self.state_start = rospy.Time.now().to_sec()
            self.escape_angle = 30.0 * np.pi / 180.0 # 오른쪽 30도 고정
            return

        # [로직 2] 라바콘이 2개 이상일 때 -> 그 사이로 주행
        rospy.loginfo("🎯 라바콘 사이 통과 중...")
        mid = (min(centers) + max(centers)) // 2
        error = mid - (w // 2)
        
        # 분모를 200.0으로 키워 부드럽게 주행 (와리가리 방지)
        self.current_lin = 0.10
        self.current_ang = np.clip(-error / 200.0, -0.5, 0.5)

    def edge_lane_control(self, img):
        h, w, _ = img.shape
        roi = img[int(h * 0.5):, :]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5,5), 0)
        # OTSU 대신 고정 임계값(70)을 사용하여 화면 바르르 떨림 방지
        _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        
        col_sum = np.sum(binary > 0, axis=0)
        if np.max(col_sum) < 5:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        idx = np.where(col_sum >= max(5, int(np.max(col_sum) * 0.3)))[0]
        track_center = np.mean(idx)
        offset = track_center - (w / 2.0)
        
        # 조향 출력 범위를 -0.5 ~ 0.5로 제한하여 안정감 부여
        self.current_lin, self.current_ang = self.forward_speed, np.clip(-self.k_angle * offset, -0.5, 0.5)

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
