#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

# ============================================================
#  FINAL MERGED : My Lane Base + Friend's Obstacle/Cone Logic
# ============================================================
class LimoFinalController:
    def __init__(self):
        rospy.init_node("limo_final_controller")

        # ---------------- ROS (작성자님 코드 설정 유지) ----------------
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

        # ---------------- PARAMETERS (속도 UP) ----------------
        # [요청] 속도가 너무 느려 0.35로 상향 조정 (기존 0.12)
        self.forward_speed = 0.35
        self.search_spin_speed = 0.4 
        self.k_angle = 0.015          # 속도가 빨라져서 조향 반응(Gain)을 약간 키움

        # ---------------- LIDAR & OBSTACLE (친구 코드 변수 이식) ----------------
        self.scan_ranges = []
        self.front = 999.0
        self.escape_angle = 0.0
        self.robot_width = 0.13       # 리모 폭 고려
        self.left_escape_count = 0
        self.force_right_escape = 0
        
        # 라바콘용 변수
        self.red_contours = []

        rospy.loginfo("✅ LIMO FINAL: My Base + Friend's Logic (Speed: %.2f)", self.forward_speed)

    # ============================================================
    # 1. LIDAR CALLBACK
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        # [친구 코드 방식 + 작성자님 스타일]
        # 정면 기준 ±10도 (총 20도) 범위의 평균 거리를 구합니다.
        # 인덱스 0이 정면이라고 가정 (LIMO 일반적 설정)
        front_zone = np.concatenate([raw[:10], raw[-10:]])
        
        # 0.1m 이하 노이즈 및 inf/nan 제거
        cleaned = [d for d in front_zone if d > 0.1 and not np.isnan(d)]
        
        if cleaned:
            self.front = np.median(cleaned)
        else:
            self.front = 999.0

    # ============================================================
    # 2. IMAGE CONVERTER (작성자님 코드 그대로 사용)
    # ============================================================
    def msg_to_cv2(self, msg):
        if self.encoding is None:
            self.encoding = msg.encoding

        h, w = msg.height, msg.width

        if self.encoding in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = arr.reshape(h, msg.step // 3, 3)[:, :w]
            if self.encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        if self.encoding == "mono8":
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = arr.reshape(h, msg.step)[:, :w]
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return None

    # ============================================================
    # 3. MAIN LOGIC (STATE MACHINE)
    # ============================================================
    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()

        # (1) 장애물 회피 상태 처리 (친구 로직)
        if self.state == "BACK":
            self.back_control()
            return

        if self.state == "ESCAPE":
            self.escape_control()
            return

        # (2) 장애물 감지 (Trigger)
        # [수정] 박스 충돌 방지를 위해 감지 거리를 0.45 -> 0.55로 늘림
        if self.front < 0.55:
            rospy.logwarn("🚨 Obstacle Detected! Distance: %.2f", self.front)
            self.state = "BACK"
            self.state_start = now
            # 관성 방지를 위해 즉시 정지 명령
            self.current_lin = 0.0
            self.current_ang = 0.0
            self.pub_cmd() # 즉시 반영
            return

        # 이미지 변환
        img = self.msg_to_cv2(msg)
        if img is None:
            return

        # (3) 미션 4: 라바콘 감지 (친구 로직)
        # 라바콘이 보이면 라인보다 우선순위를 가집니다.
        if self.detect_cone(img):
            self.cone_control(img)
            return

        # (4) 미션 1,2,5: 기본 라인트레이싱 (작성자님 코드)
        self.edge_lane_control(img)

    # ============================================================
    # 4. 미션 4: 라바콘 (친구 코드 로직 이식)
    # ============================================================
    def detect_cone(self, img):
        h, w = img.shape[:2]
        
        # ROI 설정 (친구 코드: 화면 하단 45% 사용)
        roi = img[int(h * 0.55):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 친구 코드의 빨간색 임계값
        lower_r1 = np.array([0, 120, 80])
        upper_r1 = np.array([10, 255, 255])
        lower_r2 = np.array([170, 120, 80])
        upper_r2 = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower_r1, upper_r1) | \
               cv2.inRange(hsv, lower_r2, upper_r2)

        # 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 친구 코드: 면적 200 이상인 덩어리만 인정
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

        # [친구 알고리즘]
        # 콘이 2개 이상이면 그 사이(평균)를 향해 가고, 1개면 그 콘 쪽으로 감
        if len(centers) >= 2:
            mid = (min(centers) + max(centers)) // 2
        else:
            mid = centers[0]

        error = mid - (w // 2)
        
        # 라바콘 주행 속도 및 조향 (속도 0.25로 상향)
        self.current_lin = 0.25
        # 조향 게인: 친구 코드는 error / 180.0
        self.current_ang = error / 180.0
        
        self.pub_cmd()

    # ============================================================
    # 5. 미션 3: 장애물 회피 (친구 코드 로직 이식)
    # ============================================================
    def back_control(self):
        now = rospy.Time.now().to_sec()

        # [수정] 후진 시간: 속도가 빨라졌으므로 1.2s -> 1.0s로 단축
        if now - self.state_start < 1.0:
            self.current_lin = -0.35  # 후진 속도도 빠르게
            self.current_ang = 0.0
            self.pub_cmd()
        else:
            # 후진이 끝나면 라이다로 뚫린 구멍(Gap)을 찾음
            angle = self.find_gap_max()
            
            # 친구의 방향 결정 로직 (계속 한쪽으로만 도는거 방지) 적용
            angle = self.apply_escape_direction_logic(angle)

            self.escape_angle = angle
            self.state = "ESCAPE"
            self.state_start = now
            rospy.loginfo("↪️ Finding Gap: Angle %.2f", self.escape_angle)

    def escape_control(self):
        now = rospy.Time.now().to_sec()

        # [수정] 탈출 주행: 1.0초 동안 계산된 각도로 진입
        if now - self.state_start < 1.0:
            self.current_lin = 0.28   # 탈출 속도
            self.current_ang = self.escape_angle * 1.5 # 확실하게 꺾기 위해 게인 1.5배
            self.pub_cmd()
        else:
            self.state = "LANE"
            rospy.loginfo("🚀 Escape Done -> Back to LANE")

    # [친구 코드 핵심] 라이다로 가장 넓은 구멍 찾기 (±60도 스캔)
    def find_gap_max(self):
        if len(self.scan_ranges) == 0:
            return 0.0

        raw = np.array(self.scan_ranges)
        # raw[-60:] (오른쪽 60도) + raw[:60] (왼쪽 60도) = 정면 120도 부채꼴
        ranges = np.concatenate([raw[-60:], raw[:60]])
        
        # 0.2m 보다 가깝거나 nan인 값은 0으로 처리 (벽으로 인식)
        ranges = np.where((ranges < 0.20) | np.isnan(ranges), 0.0, ranges)

        # 가장 먼 거리를 가리키는 인덱스 찾기
        idx = np.argmax(ranges)
        
        # 만약 가장 먼 곳도 너무 좁으면(로봇 폭 고려) 탈출 불가 -> 그냥 0 리턴
        if ranges[idx] < (self.robot_width + 0.10):
            return 0.0

        # 인덱스(0~120)를 각도(-60~+60)로 변환
        angle_deg = idx - 60
        return angle_deg * np.pi / 180.0

    # [친구 코드 핵심] 한쪽으로만 도는 루프 방지
    def apply_escape_direction_logic(self, angle):
        # 강제 우회전 플래그가 켜져있으면
        if self.force_right_escape > 0:
            self.force_right_escape -= 1
            return 0.9 # 강제 우회전 (친구 코드 값)

        # 만약 계속 왼쪽(음수 or 양수, 라이다 설정에 따름)으로만 가려하면
        # 친구 코드 로직: angle < 0 이면 count 증가 (왼쪽이라고 가정)
        if angle < 0:
            self.left_escape_count += 1
            # 4번 연속 왼쪽이면
            if self.left_escape_count >= 4:
                self.force_right_escape = 2 # 다음 2번은 강제 오른쪽
                self.left_escape_count = 0
        else:
            self.left_escape_count = 0

        return angle

    # ============================================================
    # 6. 미션 1,2,5: 라인 트레이싱 (작성자님 코드 그대로 + 속도)
    # ============================================================
    def edge_lane_control(self, img):
        h, w, _ = img.shape
        center = w / 2.0

        # ROI 설정 (작성자님 코드: 반 잘라서 아래만 봄)
        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        # 이진화
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 모폴로지 연산 (노이즈 제거)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 픽셀 히스토그램
        col_sum = np.sum(binary > 0, axis=0)
        max_val = np.max(col_sum) if col_sum.size > 0 else 0

        # 라인이 거의 안 보일 때 -> 제자리 회전 탐색
        if max_val < 5:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            self.pub_cmd()
            return

        # 라인 무게중심 찾기
        threshold = max(5, int(max_val * 0.3))
        idx = np.where(col_sum >= threshold)[0]

        if idx.size == 0:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            self.pub_cmd()
            return

        track_center = np.sum(idx * col_sum[idx]) / np.sum(col_sum[idx])
        offset = track_center - center

        # 조향 계산 (P 제어)
        ang = -self.k_angle * offset
        
        # [수정] 속도가 빨라졌으므로 조향각 제한을 -0.8 -> -1.0으로 조금 풀어줌
        ang = np.clip(ang, -1.0, 1.0)

        self.current_lin = self.forward_speed # 0.35
        self.current_ang = ang
        self.pub_cmd()

    def pub_cmd(self):
        self.cmd.linear.x = self.current_lin
        self.cmd.angular.z = self.current_ang
        self.pub.publish(self.cmd)

if __name__ == "__main__":
    node = LimoFinalController()
    rospy.spin()
