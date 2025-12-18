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

        # ---------------- LANE 파라미터 ----------------
        self.forward_speed = 0.12
        self.search_spin_speed = 0.25
        self.k_angle = 0.010

        # ---------------- LIDAR 파라미터 ----------------
        self.scan_ranges = []
        self.front = 999.0
        self.escape_angle = 0.0
        self.robot_width = 0.13

        rospy.loginfo("✅ LIMO FINAL (CONE/WALL/BOX) STARTED")

    # ============================================================
    # LIDAR: [수정] 벽과 장애물을 구분하기 위해 감지 범위 확대
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        
        # [변경 이유] 기존 20도(10+10)는 너무 좁아 측면의 '벽'을 못 봅니다.
        # 정면 90도 영역(좌우 45도씩)을 감시하여 측면 벽 충돌을 방지합니다.
        front_wide_zone = np.concatenate([raw[:45], raw[-45:]])
        cleaned = [d for d in front_wide_zone if d > 0.15 and not np.isnan(d)]
        
        # 하얀색 박스나 벽이 감지되면 BACK 상태를 트리거하기 위해 최소값 사용
        self.front = np.min(cleaned) if cleaned else 999.0

    # ============================================================
    # IMAGE CALLBACK (STATE MACHINE) - 구조 유지
    # ============================================================
    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()
        
        if self.state == "BACK":
            self.back_control(now)
            return

        if self.state == "ESCAPE":
            self.escape_control(now)
            return

        # [단계 2: 하얀 박스나 벽에 가까워지면 회피 동작 실행]
        if self.front < 0.35: # 너무 가까우면 박으니까 0.35m로 설정
            self.state = "BACK"
            self.state_start = now
            return

        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        # [단계 3: 빨간 라바콘 감지 시 사이 주행 로직 실행]
        if self.detect_cone(img):
            self.cone_control(img)
        else:
            self.edge_lane_control(img)

    # ============================================================
    # BACK / ESCAPE - 기존 코드 그대로 유지
    # ============================================================
    def back_control(self, now):
        if now - self.state_start < 1.2:
            self.current_lin = -0.15
            self.current_ang = 0.0
        else:
            self.escape_angle = self.find_gap_max_forward()
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self, now):
        if now - self.state_start < 1.5:
            self.current_lin = 0.12
            self.current_ang = np.clip(self.escape_angle * 1.5, -0.8, 0.8)
        else:
            self.state = "LANE"

    def find_gap_max_forward(self):
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        window_size = 20
        smoothed = np.convolve(ranges, np.ones(window_size)/window_size, mode='same')
        best_idx = np.argmax(smoothed)
        angle_deg = best_idx - 60
        safe_margin = 5 if angle_deg > 0 else -5
        return (angle_deg + safe_margin) * np.pi / 180.0

    # ============================================================
    # CONE: [수정] 박치기 방지 및 사이 주행 로직
    # ============================================================
    def detect_cone(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.55):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,120,80]), np.array([10,255,255])) | \
               cv2.inRange(hsv, np.array([170,120,80]), np.array([180,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.red_contours = [c for c in contours if cv2.contourArea(c) > 200]
        return len(self.red_contours) > 0

    def cone_control(self, img):
        """[변경] 라바콘 사이를 목표로 하고, 하나일 땐 빗겨가기"""
        h, w = img.shape[:2]
        centers = [int(cv2.moments(c)["m10"]/cv2.moments(c)["m00"]) for c in self.red_contours if cv2.moments(c)["m00"] > 0]
        if not centers: return

        # 라바콘이 2개 보이면 -> 그 정중앙(사이)으로 가기
        if len(centers) >= 2:
            mid_target = (min(centers) + max(centers)) // 2
        # 라바콘이 1개만 보이면 -> 박치기 하지 말고 옆으로 빗겨가기
        else:
            cone_x = centers[0]
            # 라바콘이 왼쪽에 있으면 오른쪽 빈 공간(w/3) 타겟팅
            mid_target = cone_x + (w // 3) if cone_x < (w // 2) else cone_x - (w // 3)

        error = mid_target - (w // 2)
        # 조향 상수 180.0을 사용하여 부드럽게 꺾음
        self.current_lin, self.current_ang = 0.12, np.clip(-error / 180.0, -0.6, 0.6)

    # ============================================================
    # LANE / UTIL - 기존 코드 그대로 유지
    # ============================================================
    def edge_lane_control(self, img):
        h, w, _ = img.shape
        roi = img[int(h * 0.5):, :]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5,5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        col_sum = np.sum(binary > 0, axis=0)
        if np.max(col_sum) < 5:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        idx = np.where(col_sum >= max(5, int(np.max(col_sum) * 0.3)))[0]
        track_center = np.mean(idx)
        offset = track_center - (w / 2.0)
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
