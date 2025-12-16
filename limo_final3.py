#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image, LaserScan          # ★ MODIFIED: LaserScan 추가
from geometry_msgs.msg import Twist
import numpy as np
import cv2

class EdgeLaneNoBridge:
    def __init__(self):
        rospy.init_node("edge_lane_nobridge_node")

        # Subscriber & Publisher
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback, queue_size=1)  # ★ ADDED
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)

        self.cmd = Twist()
        self.current_lin = 0.0
        self.current_ang = 0.0

        self.encoding = None

        # ===== 튜닝 파라미터 =====
        self.forward_speed = 0.12
        self.search_spin_speed = 0.25

        self.k_angle = 0.010

        # ===== 장애물 회피 관련 =====
        self.front_dist = 999.0            # ★ ADDED
        self.obstacle_thresh = 0.45         # ★ ADDED
        self.state = "LANE"                 # ★ ADDED
        self.state_start = rospy.Time.now().to_sec()  # ★ ADDED
        self.scan_ranges = []               # ★ ADDED
        self.escape_angle = 0.0             # ★ ADDED
        self.robot_width = 0.13             # ★ ADDED

        rospy.loginfo("✅ EdgeLaneNoBridge node started + Obstacle Avoidance")

    # --------------------------------------------------
    # ★ ADDED: LiDAR callback (전방 장애물 감지)
    # --------------------------------------------------
    def lidar_callback(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front_dist = np.median(cleaned) if cleaned else 999.0

    # --------------------------------------------------
    # Image msg -> cv2
    # --------------------------------------------------
    def msg_to_cv2(self, msg: Image):
        if self.encoding is None:
            self.encoding = msg.encoding

        h = msg.height
        w = msg.width

        if self.encoding in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = arr.reshape(h, msg.step // 3, 3)[:, :w, :]
            if self.encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        if self.encoding == "mono8":
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = arr.reshape(h, msg.step)[:, :w]
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return None

    # --------------------------------------------------
    # ★ MODIFIED: image callback (FSM 포함)
    # --------------------------------------------------
    def image_callback(self, msg: Image):
        now = rospy.Time.now().to_sec()

        # ===== 장애물 상태 우선 =====
        if self.state == "BACK":
            self.back_control()
            return

        if self.state == "ESCAPE":
            self.escape_control()
            return

        # ===== LANE → BACK 전환 조건 =====
        if self.front_dist < self.obstacle_thresh:    # ★ ADDED
            self.state = "BACK"
            self.state_start = now
            return

        # ===== 기존 라인 추종 로직 =====
        img = self.msg_to_cv2(msg)
        if img is None:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            return

        h, w, _ = img.shape
        center = w / 2.0

        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        mask = (binary > 0)
        col_sum = np.sum(mask, axis=0)
        max_val = int(np.max(col_sum)) if col_sum.size > 0 else 0

        if max_val < 5:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            return

        candidates = np.where(col_sum >= max(5, int(max_val * 0.3)))[0]
        if candidates.size == 0:
            self.current_lin = 0.0
            self.current_ang = self.search_spin_speed
            return

        x = np.arange(len(col_sum))
        track_center_x = np.sum(x[candidates] * col_sum[candidates]) / np.sum(col_sum[candidates])
        offset = track_center_x - center

        ang = -self.k_angle * offset
        ang = np.clip(ang, -0.8, 0.8)

        self.current_lin = self.forward_speed
        self.current_ang = ang

    # --------------------------------------------------
    # ★ ADDED: BACK 상태
    # --------------------------------------------------
    def back_control(self):
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.2:
            self.current_lin = -0.22
            self.current_ang = 0.0
        else:
            self.escape_angle = self.find_gap_max()
            self.state = "ESCAPE"
            self.state_start = now

    # --------------------------------------------------
    # ★ ADDED: ESCAPE 상태
    # --------------------------------------------------
    def escape_control(self):
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.0:
            self.current_lin = 0.18
            self.current_ang = self.escape_angle * 1.3
        else:
            self.state = "LANE"

    # --------------------------------------------------
    # ★ ADDED: LiDAR gap 기반 회피 방향 계산
    # --------------------------------------------------
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

    # --------------------------------------------------
    # cmd_vel publisher loop
    # --------------------------------------------------
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
