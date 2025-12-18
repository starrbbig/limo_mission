#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

class LimoEnhancedNavigator:
    def __init__(self):
        rospy.init_node("limo_enhanced_navigator_node")

        # --- Subscriber & Publisher ---
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)

        # --- 제어 및 상태 변수 ---
        self.cmd = Twist()
        self.current_lin = 0.0
        self.current_ang = 0.0
        self.encoding = None
        
        # 주행 파라미터
        self.forward_speed = 0.12
        self.search_spin_speed = 0.25
        self.k_angle = 0.012  # 차선 추종 감도 약간 상향

        # 상태 머신: LANE -> BACK -> ESCAPE_TURN -> ESCAPE_STRAIGHT
        self.state = "LANE"
        self.state_start_time = 0.0
        self.front_dist = 999.0
        self.scan_ranges = []
        self.escape_angle = 0.0

        rospy.loginfo("🚀 LIMO 통합 주행 노드 시작 (장애물 회피 + 콘 감지 + 차선)")

    def lidar_callback(self, scan):
        # 라이다 데이터 전처리
        self.scan_ranges = np.array(scan.ranges)
        # 전방 30도 범위 추출 (정면 기준 좌우 15도)
        front_indices = np.concatenate([self.scan_ranges[:15], self.scan_ranges[-15:]])
        cleaned = [d for d in front_indices if d > 0.15 and not np.isinf(d) and not np.isnan(d)]
        self.front_dist = np.median(cleaned) if cleaned else 999.0

    def find_best_gap(self):
        """가장 비어있는 각도를 찾아 라디안으로 반환"""
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        # 전방 180도 감시
        ranges = np.concatenate([raw[-90:], raw[:90]])
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        
        window_size = 30 
        smoothed = np.convolve(ranges, np.ones(window_size)/window_size, mode='same')
        
        best_idx = np.argmax(smoothed)
        angle_deg = (best_idx - 90)
        # 장애물에서 더 멀어지도록 보정
        if angle_deg > 0: angle_deg += 7
        else: angle_deg -= 7
        return angle_deg * (np.pi / 180.0)

    def detect_and_control_cone(self, img):
        """빨간색 콘을 감지하고 조향값을 계산"""
        h, w = img.shape[:2]
        roi = img[int(h * 0.5):, :] # 하단 절반 사용
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 빨간색 범위 (두 영역 합침)
        lower_r1, upper_r1 = np.array([0, 120, 70]), np.array([10, 255, 255])
        lower_r2, upper_r2 = np.array([170, 120, 70]), np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_r1, upper_r1) | cv2.inRange(hsv, lower_r2, upper_r2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours = [c for c in contours if cv2.contourArea(c) > 150]

        if len(red_contours) > 0:
            centers = []
            for c in red_contours:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    centers.append(int(M["m10"] / M["m00"]))
            
            # 여러 콘이 있으면 평균점, 하나면 그 중심점 사용
            target_x = sum(centers) / len(centers)
            error = target_x - (w / 2.0)
            self.current_lin = 0.10
            self.current_ang = -error / 150.0 # 콘 방향으로 조향
            return True
        return False

    def follow_lane(self, img):
        """기존의 차선 인식 및 추종 로직"""
        h, w = img.shape[:2]
        roi = img[int(h * 0.6):, :] # 차선은 조금 더 아래쪽 위주로
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        col_sum = np.sum(binary > 0, axis=0)
        max_val = np.max(col_sum) if col_sum.size > 0 else 0

        if max_val < 5: # 차선 안 보임
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            return

        threshold_val = max(5, int(max_val * 0.3))
        candidates = np.where(col_sum >= threshold_val)[0]
        
        if candidates.size > 0:
            track_center_x = np.mean(candidates)
            error = track_center_x - (w / 2.0)
            self.current_lin = self.forward_speed
            self.current_ang = np.clip(-self.k_angle * error, -0.8, 0.8)

    def image_callback(self, msg):
        now = rospy.Time.now().to_sec()
        img = self.msg_to_cv2(msg)
        if img is None: return

        # === [A] 장애물 회피 상태 머신 (최우선순위) ===
        if self.state == "BACK":
            if now - self.state_start_time < 1.3:
                self.current_lin, self.current_ang = -0.15, 0.0
            else:
                self.escape_angle = self.find_best_gap()
                self.state = "ESCAPE_TURN"
                self.state_start_time = now
            return

        elif self.state == "ESCAPE_TURN":
            if now - self.state_start_time < 1.0:
                self.current_lin = 0.0
                self.current_ang = np.clip(self.escape_angle * 2.5, -1.2, 1.2)
            else:
                self.state = "ESCAPE_STRAIGHT"
                self.state_start_time = now
            return

        elif self.state == "ESCAPE_STRAIGHT":
            if now - self.state_start_time < 1.2:
                self.current_lin, self.current_ang = 0.12, 0.0
            else:
                self.state = "LANE"
            return

        # === [B] 평시 주행 상태 (LANE) ===
        # 1. 장애물 감지 체크
        if self.front_dist < 0.40: # 40cm 이내 장애물 시 회피 시작
            rospy.logwarn("⚠️ 장애물 감지! 회피 기동 시작")
            self.state = "BACK"
            self.state_start_time = now
            return

        # 2. 미션 수행 (빨간 콘 감지 우선)
        is_cone_visible = self.detect_and_control_cone(img)
        
        # 3. 콘이 안 보이면 차선 추종
        if not is_cone_visible:
            self.follow_lane(img)

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
    try:
        node = LimoEnhancedNavigator()
        node.spin()
    except rospy.ROSInterruptException:
        pass
