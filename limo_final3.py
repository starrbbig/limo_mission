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

        rospy.loginfo("✅ LIMO FINAL CONTROLLER (SMART ESCAPE VER.) STARTED")

    # ============================================================
    # LIDAR
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        # 정면 20도 영역 감지 (좌우 10도씩)
        # 인덱스 0이 정면이라고 가정 (LIMO 일반적 설정)
        if len(raw) > 0:
            front_zone = np.concatenate([raw[:10], raw[-10:]])
            cleaned = [d for d in front_zone if d > 0.15 and not np.isnan(d) and not np.isinf(d)]
            self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # IMAGE CALLBACK (STATE MACHINE)
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
    # BACK / ESCAPE (수정된 부분: 스마트 회피 로직 적용)
    # ============================================================
    def back_control(self, now):
        """1단계: 후진 후 장애물 위치를 판단하여 반대로 방향 설정"""
        # 1.2초 동안 후진
        if now - self.state_start < 1.2:
            self.current_lin = -0.15
            self.current_ang = 0.0
        else:
            # 후진이 끝나는 순간 판단
            best_angle = self.find_gap_max_forward()
           
            raw = np.array(self.scan_ranges)
           
            # [수정된 로직: 좌우 밀도 비교하여 강제 회전]
            if len(raw) > 0:
                # 0.0(에러)이나 inf(무한대)를 3.5m(안전값)로 치환하여 평균 계산 왜곡 방지
                safe_raw = np.where((raw < 0.1) | np.isnan(raw) | np.isinf(raw), 3.5, raw)
               
                # 정면 기준 좌측(10~60도) vs 우측(-60~-10도) 평균 거리 계산
                left_zone = safe_raw[10:60]
                right_zone = safe_raw[-60:-10]
               
                if len(left_zone) > 0 and len(right_zone) > 0:
                    avg_left = np.mean(left_zone)
                    avg_right = np.mean(right_zone)
                   
                    # 왼쪽 벽이 오른쪽보다 현저히 가까움 -> 장애물이 왼쪽에 있음 -> 오른쪽으로 가야 함
                    if avg_left < avg_right * 0.8:
                        if best_angle > -0.1: # 현재 계산된 각도가 왼쪽을 보고 있다면
                            best_angle = -0.7 # 강제로 우회전(약 40도)
                            rospy.loginfo(f"🚧 Left Obstacle({avg_left:.2f}m) -> Force RIGHT Turn")
                           
                    # 오른쪽 벽이 현저히 가까움 -> 장애물이 오른쪽에 있음 -> 왼쪽으로 가야 함
                    elif avg_right < avg_left * 0.8:
                        if best_angle < 0.1: # 현재 계산된 각도가 오른쪽을 보고 있다면
                            best_angle = 0.7  # 강제로 좌회전
                            rospy.loginfo(f"🚧 Right Obstacle({avg_right:.2f}m) -> Force LEFT Turn")

            self.escape_angle = best_angle
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self, now):
        """2단계: 전진과 회전을 동시에 하여 곡선으로 탈출"""
        if now - self.state_start < 1.5:  # 1.5초간 곡선 주행
            self.current_lin = 0.12
            # 찾은 각도에 가중치를 주어 부드럽게 회전
            self.current_ang = np.clip(self.escape_angle * 1.5, -0.8, 0.8)
        else:
            self.state = "LANE"

    def find_gap_max_forward(self):
        """전방 120도 안에서 로봇이 지나갈 수 있는 가장 넓은 공간 탐색"""
        if len(self.scan_ranges) == 0: return 0.0
       
        raw = np.array(self.scan_ranges)
        # 뒤쪽은 아예 안 봄 (전방 좌우 60도씩 총 120도)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        # 결측치 및 너무 가까운 거리 처리
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
       
        # 윈도우 평균(Convolution)을 통해 '한 점'이 아닌 '길'을 찾음
        window_size = 20
        smoothed = np.convolve(ranges, np.ones(window_size)/window_size, mode='same')
       
        best_idx = np.argmax(smoothed)
        angle_deg = best_idx - 60 # 인덱스를 각도로 변환 (-60 ~ +60)
       
        # 장애물로부터 조금 더 안전하게 떨어지기 위한 보정(+/- 5도)
        safe_margin = 5 if angle_deg > 0 else -5
        return (angle_deg + safe_margin) * np.pi / 180.0

    # ============================================================
    # CONE / LANE
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
        h, w = img.shape[:2]
        centers = [int(cv2.moments(c)["m10"]/cv2.moments(c)["m00"]) for c in self.red_contours if cv2.moments(c)["m00"] > 0]
        if not centers: return
        mid = (min(centers) + max(centers)) // 2 if len(centers) >= 2 else centers[0]
        error = mid - (w // 2)
        self.current_lin, self.current_ang = 0.13, np.clip(-error / 180.0, -0.8, 0.8)

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
