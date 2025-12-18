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
        self.state = "LANE"     # LANE / BACK / ESCAPE / CONE_SEARCH
        self.state_start = rospy.Time.now().to_sec()

        # ---------------- IMAGE ----------------
        self.encoding = None

        # ---------------- 파라미터 ----------------
        self.forward_speed = 0.12
        self.search_spin_speed = 0.3  # 길 찾을 때 회전 속도
        self.k_angle = 0.010
        self.robot_width = 0.13

        # ---------------- LIDAR ----------------
        self.scan_ranges = []
        self.front = 999.0

        rospy.loginfo("✅ LIMO INTELLIGENT CONTROLLER STARTED")

    # ============================================================
    # LIDAR: 벽(우드락)과 장애물(박스) 감지 범위 확대 (좌우 45도)
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        
        # [수정] 정면 90도(좌우 45도)를 감시하여 '옆구리 벽 충돌' 방지
        # 벽(우드락)은 색상 인식이 안 되므로 라이다가 유일한 생명줄입니다.
        front_wide_zone = np.concatenate([raw[:45], raw[-45:]])
        cleaned = [d for d in front_wide_zone if d > 0.10 and not np.isnan(d)]
        
        # 장애물/벽에 박기 전 최소 거리 (Min값 사용으로 즉각 반응)
        self.front = np.min(cleaned) if cleaned else 999.0

    # ============================================================
    # IMAGE CALLBACK (상태 머신 유지)
    # ============================================================
    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()
        
        # 1. 후진 및 탈출 모드 (기본 로직 유지)
        if self.state == "BACK":
            self.back_control(now)
            return
        if self.state == "ESCAPE":
            self.escape_control(now)
            return

        # 2. 장애물(박스) 혹은 벽(우드락) 감지 시 회피
        if self.front < 0.30: # 30cm 이내면 벽이나 박스로 판단
            self.state = "BACK"
            self.state_start = now
            return

        img = self.msg_to_cv2(msg)
        if img is None: return

        # 3. 라바콘 감지 및 주행 전략
        if self.detect_cone(img):
            self.cone_control(img)
        else:
            # 라바콘이 없으면 차선 주행
            self.edge_lane_control(img)

    # ============================================================
    # CONE CONTROL: 사이 주행 및 경로 재탐색 지능화
    # ============================================================
    def detect_cone(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.5):, :] # 조금 더 넓게 탐색
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,120,80]), np.array([15,255,255])) | \
               cv2.inRange(hsv, np.array([165,120,80]), np.array([180,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 면적 필터링으로 가짜 빨간색 제거
        self.red_contours = [c for c in contours if cv2.contourArea(c) > 150]
        return len(self.red_contours) > 0

    def cone_control(self, img):
        """[수정] '사이' 인지 판단 후 주행, 아니면 길 찾기"""
        h, w = img.shape[:2]
        centers = []
        for c in self.red_contours:
            m = cv2.moments(c)
            if m["m00"] > 0:
                centers.append(int(m["m10"] / m["m00"]))
        
        if not centers: return
        centers.sort() # 왼쪽부터 정렬

        # [상황 1] 두 개 이상 보임 -> '사이'에 있을 확률 높음
        if len(centers) >= 2:
            left_cone = centers[0]
            right_cone = centers[-1]
            # 두 라바콘 사이의 거리가 너무 좁으면(한쪽으로 쏠리면) 보정
            mid_target = (left_cone + right_cone) // 2
            error = mid_target - (w // 2)
            
            self.current_lin = 0.12
            self.current_ang = np.clip(-error / 150.0, -0.6, 0.6)
            # rospy.loginfo("🎯 Between two cones: Stable driving")

        # [상황 2] 하나만 보임 -> 사이가 아닐 수 있음 (길 찾기 병행)
        else:
            cone_x = centers[0]
            
            # 너무 큰 라바콘(가까움)인데 하나뿐이면 코스 이탈 위험
            # 이때는 전진을 멈추고 반대편 라바콘을 찾기 위해 제자리 회전
            if cv2.contourArea(self.red_contours[0]) > 5000:
                self.current_lin = 0.05 # 아주 천천히 전진하며
                # 라바콘이 오른쪽에 있으면 왼쪽으로 돌아서 다른 놈 찾기
                self.current_ang = 0.3 if cone_x > (w//2) else -0.3
                # rospy.loginfo("🔄 Search mode: Looking for the other cone...")
            else:
                # 멀리 있는 하나는 일단 조심스럽게 접근
                target = (w // 2) # 일단 중앙 유지
                error = cone_x - target
                self.current_lin = 0.10
                # 라바콘을 정면으로 보지 않도록 오프셋 주행
                self.current_ang = 0.2 if error > 0 else -0.2

    # ============================================================
    # 나머지 함수 (기본 유지)
    # ============================================================
    def back_control(self, now):
        if now - self.state_start < 1.0:
            self.current_lin, self.current_ang = -0.15, 0.0
        else:
            self.escape_angle = self.find_gap_max_forward()
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self, now):
        if now - self.state_start < 1.3:
            self.current_lin = 0.12
            self.current_ang = np.clip(self.escape_angle * 1.6, -0.7, 0.7)
        else:
            self.state = "LANE"

    def find_gap_max_forward(self):
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        smoothed = np.convolve(ranges, np.ones(20)/20, mode='same')
        best_idx = np.argmax(smoothed)
        return (best_idx - 60) * np.pi / 180.0

    def edge_lane_control(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.6):, :]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5,5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        col_sum = np.sum(binary > 0, axis=0)
        if np.max(col_sum) < 5:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return
        idx = np.where(col_sum >= max(5, int(np.max(col_sum) * 0.3)))[0]
        offset = np.mean(idx) - (w / 2.0)
        self.current_lin = self.forward_speed
        self.current_ang = np.clip(-self.k_angle * offset, -0.7, 0.7)

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
