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
        self.red_contours = []

        # ---------------- LANE 파라미터 ----------------
        self.forward_speed = 0.12
        self.search_spin_speed = 0.25
        self.k_angle = 0.010

        # ---------------- LIDAR 파라미터 ----------------
        self.scan_ranges = []
        self.front = 999.0
        self.escape_angle = 0.0
        self.robot_width = 0.13

        rospy.loginfo("✅ LIMO FINAL CONTROLLER (STABLE & OPTIMIZED VER.) STARTED")

    # ============================================================
    # LIDAR
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        # 정면 20도 영역 감지 (좌우 10도씩)
        if len(raw) > 0:
            front_zone = np.concatenate([raw[:10], raw[-10:]])
            cleaned = [d for d in front_zone if d > 0.15 and not np.isnan(d) and not np.isinf(d)]
            self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # IMAGE CALLBACK (STATE MACHINE) - CMD 퍼블리시 로직 제거 (Spin으로 위임)
    # ============================================================
    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()
       
        # [단계 1: 장애물 회피 상태 우선 실행]
        if self.state == "BACK":
            self.back_control(now)
            return

        if self.state == "ESCAPE":
            self.escape_control(now)
            return

        # [단계 2: 장애물 감지 트리거]
        if self.front < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        # [단계 3: 카메라 기반 주행 (라바콘 or 차선)]
        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        if self.detect_cone(img):
            self.cone_control(img)
        else:
            self.edge_lane_control(img)


    # ============================================================
    # BACK / ESCAPE (무한 루프 방지 로직 유지)
    # ============================================================
    def back_control(self, now):
        """1단계: 짧게 후진하며 탈출 각도 계산 + 장애물 위치 파악"""
        if now - self.state_start < 1.2:
            self.current_lin = -0.15
            self.current_ang = 0.0
        else:
            # 후진 끝나는 시점에 가장 뚫린 방향 찾기
            best_angle = self.find_gap_max_forward()
           
            raw = np.array(self.scan_ranges)
           
            # [수정된 로직: 좌우 밀도 비교하여 강제 회전]
            if len(raw) > 0:
                # 0.0(에러)이나 inf(무한대)를 3.5m(안전값)로 치환하여 평균 계산 왜곡 방지
                safe_raw = np.where((raw < 0.1) | np.isnan(raw) | np.isinf(raw), 3.5, raw)
               
                # 정면 기준 좌측(10~60도) vs 우측(-60~-10도) 평균 거리 계산
                left_zone = safe_raw[-60:-10]
                right_zone = safe_raw[10:60]
               
                if len(left_zone) > 0 and len(right_zone) > 0:
                    avg_left = np.mean(left_zone)
                    avg_right = np.mean(right_zone)
                   
                    # 왼쪽 벽이 현저히 가까움 -> 오른쪽(-)으로 가야 함
                    if avg_left < avg_right * 0.8:
                        if best_angle > -0.1: # 현재 계산된 각도가 왼쪽(+)을 보고 있다면
                            best_angle = -0.7 # 강제로 우회전(약 -40도)
                            rospy.loginfo(f"🚧 Left Obstacle({avg_left:.2f}m) -> Force RIGHT Turn")
                           
                    # 오른쪽 벽이 현저히 가까움 -> 왼쪽(+)으로 가야 함
                    elif avg_right < avg_left * 0.8:
                        if best_angle < 0.1: # 현재 계산된 각도가 오른쪽(-)을 보고 있다면
                            best_angle = 0.7  # 강제로 좌회전
                            rospy.loginfo(f"🚧 Right Obstacle({avg_right:.2f}m) -> Force LEFT Turn")

            self.escape_angle = best_angle
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self, now):
        """2단계: 전진과 회전을 동시에 하여 곡선으로 탈출"""
        if now - self.state_start < 1.5:
            self.current_lin = 0.12
            self.current_ang = np.clip(self.escape_angle * 1.5, -0.8, 0.8)
        else:
            self.state = "LANE"

    def find_gap_max_forward(self):
        """전방 120도 안에서 로봇이 지나갈 수 있는 가장 넓은 공간 탐색"""
        if len(self.scan_ranges) == 0: return 0.0
       
        raw = np.array(self.scan_ranges)
        # 전방 좌우 60도씩 총 120도
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
       
        window_size = 20
        smoothed = np.convolve(ranges, np.ones(window_size)/window_size, mode='same')
       
        best_idx = np.argmax(smoothed)
        angle_deg = best_idx - 60
       
        safe_margin = 5 if angle_deg > 0 else -5
        return (angle_deg + safe_margin) * np.pi / 180.0

    # ============================================================
    # CONE / LANE (라바콘 가상 차선 로직 유지)
    # ============================================================
    def detect_cone(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.4):, :]
       
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
       
        # 붉은색 범위 확장
        mask1 = cv2.inRange(hsv, np.array([0, 80, 60]), np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 80, 60]), np.array([180, 255, 255]))
        mask = mask1 | mask2
       
        # 노이즈 제거
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
       
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.red_contours = [c for c in contours if cv2.contourArea(c) > 300]
       
        return len(self.red_contours) > 0

    def cone_control(self, img):
        h, w = img.shape[:2]
        img_center = w // 2
       
        centers = []
        for c in self.red_contours:
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"])
                centers.append(cx)
       
        if not centers:
            return

        # [핵심 로직: 가상 차선 생성]
        left_cones = [c for c in centers if c < img_center]
        right_cones = [c for c in centers if c >= img_center]

        target_x = img_center # 기본값은 정면
        offset_pixels = 140   # 콘에서 띄울 거리 (트랙 폭에 따라 조절)
       
        # 1. 양쪽 콘이 다 보임 -> 사이의 중간으로 주행
        if left_cones and right_cones:
            l_max = max(left_cones)    # 왼쪽 콘 중 가장 오른쪽
            r_min = min(right_cones)   # 오른쪽 콘 중 가장 왼쪽
            target_x = (l_max + r_min) // 2

        # 2. 왼쪽 콘만 보임 -> 오른쪽으로 offset 띄움
        elif left_cones:
            l_max = max(left_cones)
            target_x = l_max + offset_pixels

        # 3. 오른쪽 콘만 보임 -> 왼쪽으로 offset 띄움
        elif right_cones:
            r_min = min(right_cones)
            target_x = r_min - offset_pixels
           
        # 조향 계산
        error = target_x - img_center
       
        # 가변 속도: 에러가 클수록 속도를 줄여 안정적 회전 유도
        base_speed = 0.13
        if abs(error) > 80:
            base_speed = 0.08
       
        self.current_lin = base_speed
        self.current_ang = np.clip(-error / 160.0, -0.9, 0.9)


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
        # 원본 코드의 안전한 인코딩 처리 로직 복원 (RGB2BGR 변환 포함)
        if self.encoding is None: self.encoding = msg.encoding
        arr = np.frombuffer(msg.data, dtype=np.uint8)
       
        # ROS 이미지 메시지 데이터를 OpenCV 이미지 형식으로 변환
        try:
            img = arr.reshape(msg.height, msg.width, 3)
            if self.encoding == "rgb8":
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif self.encoding == "bgr8":
                return img
            else:
                return None
        except ValueError:
            # reshape 실패 시 (데이터 길이 문제)
            rospy.logwarn("Failed to reshape image data.")
            return None


    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            # [오류 복구] CMD를 spin 루프에서 고정된 20Hz로 지속 전송 (가장 안정적)
            self.cmd.linear.x, self.cmd.angular.z = self.current_lin, self.current_ang
            self.pub.publish(self.cmd)
            rate.sleep()

if __name__ == "__main__":
    LimoFinalController().spin()
