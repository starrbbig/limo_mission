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

        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_cb, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb, queue_size=1)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.cmd = Twist()

        # ================= FSM =================
        self.state = "LANE"      # LANE / BACK / ESCAPE
        self.state_start = rospy.Time.now().to_sec()

        # ================= LANE =================
        self.forward_speed = 0.24        # ★ 2배 속도
        self.search_spin_speed = 0.30
        self.k_angle = 0.010
        self.last_lane_dir = 1.0

        # ================= LIDAR =================
        self.scan_ranges = []
        self.front = 999.0
        self.robot_width = 0.13

        self.escape_angle = 0.0
        self.left_escape_count = 0
        self.force_right_escape = 0

        # ================= CMD =================
        self.current_lin = 0.0
        self.current_ang = 0.0

        self.encoding = None

        rospy.loginfo("✅ LIMO FINAL CONTROLLER : ZERO FINAL VERSION")

    # ============================================================
    # LIDAR CALLBACK  (긁힘 + 조기 감지)
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        # 정면 ±30도
        front_zone = np.concatenate([raw[:18], raw[-18:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]

        # 모서리 기준 판단
        self.front = min(cleaned) if cleaned else 999.0

    # ============================================================
    # IMAGE → CV2
    # ============================================================
    def msg_to_cv2(self, msg):
        if self.encoding is None:
            self.encoding = msg.encoding

        h, w = msg.height, msg.width
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = arr.reshape(h, msg.step // 3, 3)[:, :w]

        if self.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    # ============================================================
    # IMAGE CALLBACK
    # ============================================================
    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()

        # ---------- ESCAPE ----------
        if self.state == "ESCAPE":
            self.escape_control()
            return

        # ---------- BACK ----------
        if self.state == "BACK":
            self.back_control()
            return

        # ---------- 사전 감속 (조향 간섭 X) ----------
        if self.front < 0.60:
            self.current_lin = 0.12   # 절반 속도만
            # 조향은 건드리지 않음

        # ---------- BACK 진입 ----------
        if self.front < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed * self.last_lane_dir
            return

        # ---------- 라바콘 ----------
        if self.cone_control(img):
            return

        # ---------- 라인 ----------
        self.edge_lane_control(img)

    # ============================================================
    # CONE (두 번째 코드 방식)
    # ============================================================
    def cone_control(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.55):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_r1 = np.array([0,120,80])
        upper_r1 = np.array([10,255,255])
        lower_r2 = np.array([170,120,80])
        upper_r2 = np.array([180,255,255])

        mask = cv2.inRange(hsv, lower_r1, upper_r1) | \
               cv2.inRange(hsv, lower_r2, upper_r2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for c in contours:
            if cv2.contourArea(c) < 200:
                continue
            M = cv2.moments(c)
            if M["m00"] > 0:
                centers.append(int(M["m10"] / M["m00"]))

        if not centers:
            return False

        mid = (min(centers) + max(centers)) // 2
        error = mid - (w // 2)

        self.current_lin = 0.26
        self.current_ang = error / 180.0
        return True

    # ============================================================
    # EDGE LANE (기존 로직 유지)
    # ============================================================
    def edge_lane_control(self, img):
        h, w, _ = img.shape
        center = w / 2.0

        roi = img[int(h * 0.45):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        col_sum = np.sum(binary > 0, axis=0)
        max_val = np.max(col_sum) if col_sum.size > 0 else 0

        if max_val < 5:
            self.current_lin = 0.12
            self.current_ang = self.search_spin_speed * self.last_lane_dir
            return

        idx = np.where(col_sum >= max(5, int(max_val * 0.3)))[0]
        if idx.size == 0:
            return

        track_center = np.sum(idx * col_sum[idx]) / np.sum(col_sum[idx])
        offset = track_center - center

        ang = -self.k_angle * offset
        ang = np.clip(ang, -0.9, 0.9)

        if abs(ang) > 0.05:
            self.last_lane_dir = np.sign(ang)

        self.current_lin = self.forward_speed
        self.current_ang = ang

    # ============================================================
    # BACK / ESCAPE (무한루프 제거)
    # ============================================================
    def back_control(self):
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.2:
            self.current_lin = -0.26
            self.current_ang = 0.0
        else:
            angle = self.find_gap_max()

            # ★ 핵심 : 각도 0 방지
            if abs(angle) < 0.15:
                angle = 0.6

            angle = self.apply_escape_direction_logic(angle)

            self.escape_angle = angle
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self):
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.0:
            self.current_lin = 0.22
            self.current_ang = self.escape_angle * 1.3
        else:
            self.state = "LANE"

    # ============================================================
    # Escape helpers
    # ============================================================
    def apply_escape_direction_logic(self, angle):
        if self.force_right_escape > 0:
            self.force_right_escape -= 1
            return 0.9

        if angle < 0:
            self.left_escape_count += 1
            if self.left_escape_count >= 4:
                self.force_right_escape = 2
                self.left_escape_count = 0
        else:
            self.left_escape_count = 0

        return angle

    def find_gap_max(self):
        if len(self.scan_ranges) == 0:
            return 0.0

        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.where((ranges < 0.20) | np.isnan(ranges), 0.0, ranges)

        idx = np.argmax(ranges)
        if ranges[idx] < (self.robot_width + 0.10):
            return 0.0

        return (idx - 60) * np.pi / 180.0

    # ============================================================
    # LOOP
    # ============================================================
    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.cmd.linear.x = self.current_lin
            self.cmd.angular.z = self.current_ang
            self.pub.publish(self.cmd)
            rate.sleep()


if __name__ == "__main__":
    node = LimoFinalController()
    node.spin()
