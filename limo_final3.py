#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist


class LimoFinalExam:
    def __init__(self):
        rospy.init_node("limo_final_exam")

        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_cb, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb, queue_size=1)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # ================= STATE =================
        self.state = "LANE"      # LANE / BACK / ESCAPE
        self.state_start = rospy.Time.now().to_sec()

        # ================= LANE PARAM (유지) =================
        self.k_angle = 0.010
        self.search_spin_speed = 0.25
        self.forward_speed = 0.22   # ★ 기본 라인 속도

        # ================= LIDAR =================
        self.scan_ranges = []
        self.front = 999.0
        self.robot_width = 0.13
        self.escape_angle = 0.0

        # ================= ESCAPE LOGIC =================
        self.left_escape_count = 0
        self.force_right_escape = 0

        # ================= IMAGE =================
        self.encoding = None

        rospy.loginfo("✅ LIMO FINAL EXAM CONTROLLER STARTED")

    # ============================================================
    # LIDAR (인식 거리 축소)
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        front_zone = np.concatenate([raw[:10], raw[-10:]])
        valid = [d for d in front_zone if d > 0.10 and not np.isnan(d)]
        self.front = np.median(valid) if valid else 999.0

    # ============================================================
    # IMAGE CALLBACK
    # ============================================================
    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()
        cmd = Twist()

        # ---------- BACK ----------
        if self.state == "BACK":
            lin, ang = self.back_control(now)
            cmd.linear.x = lin
            cmd.angular.z = ang
            self.pub.publish(cmd)
            return

        # ---------- ESCAPE ----------
        if self.state == "ESCAPE":
            lin, ang = self.escape_control(now)
            cmd.linear.x = lin
            cmd.angular.z = ang
            self.pub.publish(cmd)
            return

        img = self.msg_to_cv2(msg)
        if img is None:
            cmd.linear.x = 0.0
            cmd.angular.z = self.search_spin_speed
            self.pub.publish(cmd)
            return

        # ====================================================
        # 1️⃣ 라바콘 우선 (라이다보다 먼저)
        # ====================================================
        cone = self.detect_cone(img)
        if cone is not None:
            self.left_escape_count = 0
            self.force_right_escape = 0
            lin, ang = cone
            cmd.linear.x = lin
            cmd.angular.z = ang
            self.pub.publish(cmd)
            return

        # ====================================================
        # 2️⃣ 장애물 판단 (거리 축소)
        # ====================================================
        if self.front < 0.25:
            self.state = "BACK"
            self.state_start = now
            return

        # ====================================================
        # 3️⃣ 기본 라인트레이싱 (그대로 유지)
        # ====================================================
        lin, ang = self.edge_lane_control(img)
        cmd.linear.x = lin
        cmd.angular.z = ang
        self.pub.publish(cmd)

    # ============================================================
    # IMAGE CONVERT
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
    # CONE (유지)
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
        contours = [c for c in contours if cv2.contourArea(c) > 200]

        if not contours:
            return None

        centers = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                centers.append(int(M["m10"] / M["m00"]))

        mid = (min(centers) + max(centers)) // 2
        error = mid - (w // 2)

        return 0.21, error / 180.0

    # ============================================================
    # EDGE LANE (❗ 완전 유지)
    # ============================================================
    def edge_lane_control(self, img):
        h, w, _ = img.shape
        center = w / 2.0

        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        col_sum = np.sum(binary > 0, axis=0)
        max_val = np.max(col_sum) if col_sum.size > 0 else 0

        if max_val < 5:
            return 0.10, self.search_spin_speed

        idx = np.where(col_sum >= max(5, int(max_val * 0.3)))[0]
        if idx.size == 0:
            return 0.10, self.search_spin_speed

        track_center = np.sum(idx * col_sum[idx]) / np.sum(col_sum[idx])

        offset = (track_center - center) + 5.0   # 중앙 보정 유지
        ang = -self.k_angle * offset
        ang = np.clip(ang, -0.8, 0.8)

        return self.forward_speed, ang

    # ============================================================
    # BACK / ESCAPE
    # ============================================================
    def back_control(self, now):
        if now - self.state_start < 1.4:
            return -0.24, 0.0
        else:
            angle = self.find_gap_max()
            angle = self.apply_escape_direction_logic(angle)
            self.escape_angle = angle
            self.state = "ESCAPE"
            self.state_start = now
            return 0.0, 0.0

    def escape_control(self, now):
        if now - self.state_start < 1.0:
            return 0.19, self.escape_angle * 1.3
        else:
            self.state = "LANE"
            return 0.0, 0.0

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
        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.where((ranges < 0.10) | np.isnan(ranges), 0.0, ranges)

        idx = np.argmax(ranges)
        if ranges[idx] < (self.robot_width + 0.10):
            return 0.0

        return (idx - 60) * np.pi / 180.0


if __name__ == "__main__":
    node = LimoFinalExam()
    rospy.spin()

