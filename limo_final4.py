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

        # ================= ROS =================
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_cb, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb, queue_size=1)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # ================= CMD =================
        self.cmd = Twist()
        self.current_lin = 0.0
        self.current_ang = 0.0

        # ================= STATE =================
        self.state = "LANE"      # LANE / BACK / ESCAPE / CONE_PASS
        self.state_start = rospy.Time.now().to_sec()

        # ================= IMAGE =================
        self.encoding = None
        self.red_contours = []

        # ================= LANE PARAM =================
        self.forward_speed = 0.12
        self.search_spin_speed = 0.25
        self.k_angle = 0.010

        # ================= LIDAR PARAM =================
        self.scan_ranges = []
        self.front = 999.0
        self.escape_angle = 0.0
        self.robot_width = 0.13

        rospy.loginfo("✅ LIMO FINAL CONTROLLER (CONE PASS ENABLED) STARTED")

    # ============================================================
    # LIDAR CALLBACK
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.15 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # IMAGE CALLBACK (STATE MACHINE)
    # ============================================================
    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()

        # ----- BACK / ESCAPE 우선 -----
        if self.state == "BACK":
            self.back_control(now)
            return

        if self.state == "ESCAPE":
            self.escape_control(now)
            return

        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin, self.current_ang = 0.0, self.search_spin_speed
            return

        # ----- CONE PASS 상태 -----
        if self.state == "CONE_PASS":
            if self.detect_cone(img):
                self.cone_control(img)
            else:
                self.state = "LANE"
            return

        # ----- 라바콘 우선 판단 -----
        if self.detect_cone(img):
            self.state = "CONE_PASS"
            self.cone_control(img)
            return

        # ----- 장애물 판단 (라바콘 아닐 때만) -----
        if self.front < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        # ----- 기본 차선 주행 -----
        self.edge_lane_control(img)

    # ============================================================
    # BACK / ESCAPE
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
        if len(self.scan_ranges) == 0:
            return 0.0

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
    # CONE DETECTION & CONTROL
    # ============================================================
    def detect_cone(self, img):
        h, w = img.shape[:2]
        roi = img[int(h * 0.55):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array([0,120,80]), np.array([10,255,255])) | \
               cv2.inRange(hsv, np.array([170,120,80]), np.array([180,255,255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.red_contours = [c for c in contours if cv2.contourArea(c) > 200]

        return len(self.red_contours) >= 2

    def cone_control(self, img):
        h, w = img.shape[:2]

        centers = []
        for c in self.red_contours:
            m = cv2.moments(c)
            if m["m00"] > 0:
                centers.append(int(m["m10"] / m["m00"]))

        if len(centers) < 2:
            return

        mid = (min(centers) + max(centers)) // 2
        error = mid - (w // 2)

        # ★ 라바콘 사이 통과 시 속도 상승
        self.current_lin = 0.18
        self.current_ang = np.clip(-error / 150.0, -0.6, 0.6)

    # ============================================================
    # EDGE LANE
    # ============================================================
    def edge_lane_control(self, img):
        h, w, _ = img.shape
        roi = img[int(h * 0.5):, :]
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5,5), 0)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        col_sum = np.sum(binary > 0, axis=0)
        if np.max(col_sum) < 5:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            return

        idx = np.where(col_sum >= max(5, int(np.max(col_sum) * 0.3)))[0]
        track_center = np.mean(idx)
        offset = track_center - (w / 2.0)

        self.current_lin = self.forward_speed
        self.current_ang = np.clip(-self.k_angle * offset, -0.8, 0.8)

    # ============================================================
    # IMAGE CONVERT
    # ============================================================
    def msg_to_cv2(self, msg):
        if self.encoding is None:
            self.encoding = msg.encoding

        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = arr.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]

        if self.encoding == "rgb8":
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    # ============================================================
    # SPIN
    # ============================================================
    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.cmd.linear.x = self.current_lin
            self.cmd.angular.z = self.current_ang
            self.pub.publish(self.cmd)
            rate.sleep()


if __name__ == "__main__":
    LimoFinalController().spin()
