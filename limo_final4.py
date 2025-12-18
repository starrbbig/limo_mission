#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

# ============================================================
#  FINAL : Edge Lane + Obstacle + Cone (LIMO)
# ============================================================
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
        self.state = "LANE"     # LANE / CONE / BACK / ESCAPE
        self.state_start = rospy.Time.now().to_sec()

        # ---------------- IMAGE ----------------
        self.encoding = None

        # ---------------- LANE (네 코드) ----------------
        self.forward_speed = 0.12
        self.search_spin_speed = 0.25
        self.k_angle = 0.010

        # ---------------- LIDAR (친구 코드) ----------------
        self.scan_ranges = []
        self.front = 999.0
        self.escape_angle = 0.0
        self.robot_width = 0.13
        self.left_escape_count = 0
        self.force_right_escape = 0

        rospy.loginfo("✅ LIMO FINAL CONTROLLER STARTED")

    # ============================================================
    # LIDAR
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # IMAGE → CV2
    # ============================================================
    def msg_to_cv2(self, msg):
        if self.encoding is None:
            self.encoding = msg.encoding
            rospy.loginfo("📷 encoding: %s", self.encoding)

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
    # IMAGE CALLBACK (STATE MACHINE)
    # ============================================================
    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()

        # ---------- 장애물 상태 ----------
        if self.state == "BACK":
            self.back_control()
            return

        if self.state == "ESCAPE":
            self.escape_control()
            return

        # ---------- 장애물 감지 ----------
        if self.front < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            return

        # ---------- 라바콘 ----------
        if self.detect_cone(img):
            self.cone_control(img)
            return

        # ---------- 기본 라인트레이싱 ----------
        self.edge_lane_control(img)

    # ============================================================
    # CONE (친구 코드)
    # ============================================================
    def detect_cone(self, img):
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

        if len(centers) >= 2:
            mid = (min(centers) + max(centers)) // 2
        else:
            mid = centers[0]

        error = mid - (w // 2)
        self.current_lin = 0.13
        self.current_ang = error / 180.0

    # ============================================================
    # EDGE LANE (네 코드)
    # ============================================================
    def edge_lane_control(self, img):
        h, w, _ = img.shape
        center = w / 2.0

        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        col_sum = np.sum(binary > 0, axis=0)
        max_val = np.max(col_sum) if col_sum.size > 0 else 0

        if max_val < 5:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            return

        threshold = max(5, int(max_val * 0.3))
        idx = np.where(col_sum >= threshold)[0]

        if idx.size == 0:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            return

        track_center = np.sum(idx * col_sum[idx]) / np.sum(col_sum[idx])
        offset = track_center - center

        ang = -self.k_angle * offset
        ang = np.clip(ang, -0.8, 0.8)

        self.current_lin = self.forward_speed
        self.current_ang = ang

    # ============================================================
    # BACK / ESCAPE (친구 코드)
    # ============================================================
    def back_control(self):
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.2:
            self.current_lin = -0.15
            self.current_ang = 0.0
        else:
            angle = self.find_gap_max()
            angle = self.apply_escape_direction_logic(angle)

            self.escape_angle = angle
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self):
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.0:
            self.current_lin = 0.12
            self.current_ang = self.escape_angle * 1.3
        else:
            self.state = "LANE"

    def apply_escape_direction_logic(self, angle):
        if self.force_right_escape > 0:
            self.force_right_escape -= 1
            return 0.7

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

        angle_deg = idx - 60
        return angle_deg * np.pi / 180.0

    # ============================================================
    # PUBLISH LOOP
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
