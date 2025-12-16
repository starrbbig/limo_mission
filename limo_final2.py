#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist


class MissionLaneController:
    def __init__(self):
        rospy.init_node("mission_lane_controller")

        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_cb, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb, queue_size=1)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # ================= 상태 =================
        self.mode = "LANE"   # LANE / OBSTACLE / CONE
        self.state = "LANE"  # LANE / BACK / ESCAPE
        self.state_start = rospy.Time.now().to_sec()

        # ================= 라이다 =================
        self.scan = []
        self.front = 999.0
        self.left = 999.0
        self.right = 999.0
        self.robot_width = 0.13
        self.escape_angle = 0.0

        # ================= 카메라 =================
        self.encoding = None

        # ================= 파라미터 =================
        self.lane_speed = 0.24

        rospy.loginfo("Mission Lane Controller STARTED")

    # ============================================================
    # LIDAR CALLBACK
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan = raw

        # 정면
        front_zone = np.concatenate([raw[:10], raw[-10:]])
        front_valid = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front = min(front_valid) if front_valid else 999.0

        # 좌우 (라바콘)
        left_zone = raw[60:90]
        right_zone = raw[270:300]

        left_valid = [d for d in left_zone if d > 0.20 and not np.isnan(d)]
        right_valid = [d for d in right_zone if d > 0.20 and not np.isnan(d)]

        self.left = np.mean(left_valid) if left_valid else 999.0
        self.right = np.mean(right_valid) if right_valid else 999.0

    # ============================================================
    # IMAGE CALLBACK
    # ============================================================
    def image_cb(self, msg):
        now = rospy.Time.now().to_sec()
        twist = Twist()

        # ================= 미션 자동 판단 (A) =================
        if self.front < 0.45:
            self.mode = "OBSTACLE"
        elif abs(self.left - self.right) > 0.35:
            self.mode = "CONE"
        else:
            self.mode = "LANE"

        # ================= 미션 #3 장애물 =================
        if self.mode == "OBSTACLE":
            self.obstacle_control(twist, now)
            self.pub.publish(twist)
            return

        # ================= 미션 #4 라바콘 =================
        if self.mode == "CONE":
            self.cone_control(twist)
            self.pub.publish(twist)
            return

        # ================= 기본 라인트레이싱 =================
        self.lane_control(msg, twist)
        self.pub.publish(twist)

    # ============================================================
    # LANE CONTROL (원본 유지)
    # ============================================================
    def lane_control(self, msg, twist):
        if self.encoding is None:
            self.encoding = msg.encoding

        h, w = msg.height, msg.width
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, msg.step // 3, 3)[:, :w]
        if self.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        col_sum = np.sum(binary > 0, axis=0)
        max_val = np.max(col_sum) if col_sum.size > 0 else 0

        if max_val < 5:
            twist.linear.x = 0.12
            twist.angular.z = 0.25
            return

        idx = np.where(col_sum >= max_val * 0.3)[0]
        if idx.size == 0:
            twist.linear.x = 0.12
            twist.angular.z = 0.25
            return

        track_center = np.mean(idx)
        offset = track_center - (w / 2.0)

        twist.linear.x = self.lane_speed
        twist.angular.z = -0.008 * offset

    # ============================================================
    # OBSTACLE CONTROL (미션 #3)
    # ============================================================
    def obstacle_control(self, twist, now):
        if self.state == "LANE":
            self.state = "BACK"
            self.state_start = now

        if self.state == "BACK":
            if now - self.state_start < 1.3:
                twist.linear.x = -0.24
                twist.angular.z = 0.0
            else:
                self.escape_angle = self.find_gap_max()
                if abs(self.escape_angle) < 0.15:
                    self.escape_angle = 0.6
                self.state = "ESCAPE"
                self.state_start = now

        elif self.state == "ESCAPE":
            if now - self.state_start < 1.0:
                twist.linear.x = 0.20
                twist.angular.z = self.escape_angle
            else:
                self.state = "LANE"

    # ============================================================
    # CONE CONTROL (미션 #4, 라이다)
    # ============================================================
    def cone_control(self, twist):
        error = self.right - self.left
        twist.linear.x = 0.20
        twist.angular.z = error * 0.8

    # ============================================================
    # GAP FINDER
    # ============================================================
    def find_gap_max(self):
        if len(self.scan) == 0:
            return 0.0

        ranges = np.concatenate([self.scan[-60:], self.scan[:60]])
        ranges = np.where((ranges < 0.20) | np.isnan(ranges), 0.0, ranges)

        idx = np.argmax(ranges)
        if ranges[idx] < (self.robot_width + 0.10):
            return 0.0

        return (idx - 60) * np.pi / 180.0


if __name__ == "__main__":
    MissionLaneController()
    rospy.spin()
