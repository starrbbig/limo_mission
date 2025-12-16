#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image, LaserScan # [수정/추가] LaserScan 메시지 추가
from geometry_msgs.msg import Twist
import numpy as np
import cv2

class EdgeLaneNoBridge:
    def __init__(self):
        rospy.init_node("edge_lane_nobridge_node")

        # Subscriber & Publisher
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        # [수정/추가] 라이다(LiDAR) 서브스크라이버 추가 (장애물 감지용)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback, queue_size=1)
        
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)

        self.cmd = Twist()
        self.current_lin = 0.0
        self.current_ang = 0.0

        self.encoding = None

        # ===== 튜닝 파라미터 (예전에 잘 움직이던 쪽에 가깝게) =====
        self.forward_speed = 0.12   # 기본 전진 속도
        self.search_spin_speed = 0.25 # 라인 못 찾을 때 회전 속도

        self.canny_low = 50
        self.canny_high = 150
        self.edge_thresh = 10 
        self.k_angle = 0.010 

        # [수정/추가] 장애물 회피를 위한 변수 초기화
        self.state = "LANE"        # 현재 상태: LANE(주행), BACK(후진), ESCAPE(탈출)
        self.state_start = 0.0     # 상태 변경 시간 기록
        self.front_dist = 999.0    # 전방 장애물 거리
        self.scan_ranges = []      # 라이다 데이터 저장
        self.escape_angle = 0.0    # 탈출할 방향 각도
        self.robot_width = 0.13    # 로봇 폭 (틈새 판단용)

        rospy.loginfo("✅ EdgeLaneNoBridge node started (with Obstacle Avoidance)")

    # ----------------------------- #
    # [수정/추가] 라이다 콜백 함수
    # ----------------------------- #
    def lidar_callback(self, scan):
        # 전체 스캔 데이터를 numpy 배열로 저장 (나중에 틈새 찾을 때 씀)
        self.scan_ranges = np.array(scan.ranges)
        
        # 정면 기준 ±10도 부근의 장애물 거리만 추출하여 중앙값 계산
        # 전방에 벽/장애물이 있는지 판단하는 용도
        front_zone = np.concatenate([self.scan_ranges[:10], self.scan_ranges[-10:]])
        # 유효한 거리 데이터(0.2m 이상)만 필터링
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        
        if cleaned:
            self.front_dist = np.median(cleaned)
        else:
            self.front_dist = 999.0

    # ----------------------------- #
    # Image msg -> numpy (cv2용)
    # ----------------------------- #
    def msg_to_cv2(self, msg: Image):
        if self.encoding is None:
            self.encoding = msg.encoding
            rospy.loginfo("📷 image encoding: %s", self.encoding)

        h = msg.height
        w = msg.width

        # 3채널 영상 (rgb8/bgr8)
        if self.encoding in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            try:
                img = arr.reshape(h, msg.step // 3, 3)
                img = img[:, :w, :]
            except Exception as e:
                rospy.logwarn("reshape error: %s", e)
                return None
            
            if self.encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        # 1채널 영상 (mono8)
        if self.encoding == "mono8":
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            try:
                img = arr.reshape(h, msg.step)
                img = img[:, :w]
            except Exception as e:
                rospy.logwarn("reshape mono8 error: %s", e)
                return None
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        rospy.logwarn_throttle(2.0, "Unsupported encoding: %s", self.encoding)
        return None

    # ----------------------------- #
    # 이미지 콜백: "검은 트랙" 중앙 추종 버전
    # ----------------------------- #
    def image_callback(self, msg: Image):
        # [수정/추가] 장애물 회피 로직 우선 처리
        # 현재 상태가 LANE이 아니거나, 전방에 장애물이 있으면 회피 로직으로 점프
        
        now = rospy.Time.now().to_sec()

        # 1. 후진(BACK) 상태 처리
        if self.state == "BACK":
            # 1.2초 동안 후진
            if now - self.state_start < 1.2:
                self.current_lin = -0.15  # 후진 속도
                self.current_ang = 0.0
            else:
                # 후진 끝, 빈 공간(Gap) 찾아서 탈출 모드로 변경
                angle = self.find_gap_max()
                self.escape_angle = angle
                self.state = "ESCAPE"
                self.state_start = now
                rospy.loginfo("[STATE] BACK -> ESCAPE (angle: %.2f)", angle)
            return # 이미지 처리 안 하고 리턴

        # 2. 탈출(ESCAPE) 상태 처리
        if self.state == "ESCAPE":
            # 1.0초 동안 계산된 각도로 회전하며 전진/조향
            if now - self.state_start < 1.0:
                self.current_lin = 0.12
                self.current_ang = self.escape_angle * 1.3 # 조향 게인
            else:
                # 탈출 끝, 다시 라인 트레이싱 모드로
                self.state = "LANE"
                rospy.loginfo("[STATE] ESCAPE -> LANE")
            return # 이미지 처리 안 하고 리턴

        # 3. 주행(LANE) 상태인데 장애물이 너무 가까울 때 (감지)
        # 0.45m 이내면 충돌 위험으로 판단 -> BACK 상태로 진입
        if self.front_dist < 0.45:
            self.state = "BACK"
            self.state_start = now
            self.current_lin = 0.0 # 즉시 정지 준비
            rospy.loginfo("[STATE] LANE -> BACK (Obstacle detected: %.2fm)", self.front_dist)
            return

        # ==============================================================
        # [기존 코드 유지] 아래부터는 기존의 영상 처리 로직입니다.
        # ==============================================================
        
        img = self.msg_to_cv2(msg)
        if img is None:
            # 이미지 못 읽으면 회전만
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            return

        h, w, _ = img.shape
        center = w / 2.0

        # 1) 바닥 쪽 ROI (하단 50% 사용해서 트랙 폭 넓게 보기)
        roi_y_start = int(h * 0.5) 
        roi = img[roi_y_start:, :]

        # 2) 그레이 + 블러
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3) 검은 트랙 강조: THRESH_BINARY_INV + OTSU
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 4) 노이즈 제거 (3x3 작은 커널만)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 5) 열별 "검은 픽셀(=255)" 개수
        mask = (binary > 0)
        col_sum = np.sum(mask, axis=0) # shape: (w,)
        max_val = int(np.max(col_sum)) if col_sum.size > 0 else 0

        # 너무 어둡게 안 잡히면 트랙 못 찾았다고 보고 회전
        dark_min_pixels = 5 
        if max_val < dark_min_pixels:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            rospy.loginfo_throttle(
                0.8,
                f"[BLACK] no dark enough column (max={max_val}) → spin"
            )
            return

        # 6) max의 일정 비율 이상인 열들만 "트랙 후보"로 사용
        dark_col_ratio = 0.3 
        threshold_val = max(dark_min_pixels, int(max_val * dark_col_ratio))
        candidates = np.where(col_sum >= threshold_val)[0]

        if candidates.size == 0:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            rospy.loginfo_throttle(
                0.8,
                f"[BLACK] no candidate columns (max={max_val}) → spin"
            )
            return

        # 7) 후보 열들의 무게중심 = 검은 트랙 중앙 x
        x = np.arange(len(col_sum))
        track_center_x = float(np.sum(x[candidates] * col_sum[candidates]) /
                               np.sum(col_sum[candidates]))

        offset = track_center_x - center 
        offset_norm = offset / (w / 2.0)

        ang = -self.k_angle * offset
        ang = max(min(ang, 0.8), -0.8)

        self.current_lin = self.forward_speed
        self.current_ang = ang

        rospy.loginfo_throttle(
            0.3,
            f"[BLACK] center={track_center_x:.1f} off={offset:.1f} "
            f"norm={offset_norm:.2f} w={ang:.3f} max={max_val} cand={candidates.size}"
        )

    # ----------------------------- #
    # [수정/추가] 빈 공간(Gap) 찾는 함수
    # ----------------------------- #
    def find_gap_max(self):
        if len(self.scan_ranges) == 0:
            return 0.0
            
        raw = np.array(self.scan_ranges)
        # 로봇 기준 뒤쪽(-180~180도 중 뒤쪽)을 제외하고, 
        # 대략 -60도(우측) ~ +60도(좌측) 사이만 봅니다.
        # 인덱스상 0이 정면이므로, 배열 뒤쪽(-60개)과 앞쪽(60개)을 합침
        ranges = np.concatenate([raw[-60:], raw[:60]])

        # 너무 가까운 거리나 NaN은 0으로 처리해서 무시
        ranges = np.where((ranges < 0.20) | np.isnan(ranges), 0.0, ranges)
        
        # 가장 먼 거리(열린 공간)의 인덱스 찾기
        idx = np.argmax(ranges)
        
        # 만약 가장 먼 곳도 로봇이 지나가기 좁다면(안전 거리 포함) 그냥 0도 리턴
        if ranges[idx] < (self.robot_width + 0.10):
            return 0.0

        # 인덱스를 각도로 변환 (인덱스 0이 -60도, 인덱스 60이 0도, 인덱스 120이 +60도)
        # idx - 60 => 정면 기준 각도(degree)
        angle_deg = idx - 60
        
        # 라디안으로 변환
        return angle_deg * np.pi / 180.0

    # ----------------------------- #
    # /cmd_vel 계속 발행
    # ----------------------------- #
    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.cmd.linear.x = self.current_lin
            self.cmd.angular.z = self.current_ang
            self.cmd_pub.publish(self.cmd)
            rate.sleep()

if __name__ == "__main__":
    node = EdgeLaneNoBridge()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
