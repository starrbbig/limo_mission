#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2


class EdgeLaneNoBridge:
    def __init__(self):
        rospy.init_node("edge_lane_nobridge_node")

        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)

        self.cmd = Twist()
        self.current_lin = 0.0
        self.current_ang = 0.0

        self.front = 999.0
        self.encoding = None

        rospy.loginfo("EdgeLaneNoBridge STARTED")

    # ============================================================
    # LIDAR (아주 단순)
    # ============================================================
    def lidar_callback(self, scan):
        raw = np.array(scan.ranges)

        front_zone = np.concatenate([raw[:15], raw[-15:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front = min(cleaned) if cleaned else 999.0

    # ============================================================
    # IMAGE CALLBACK
    # ============================================================
    def image_callback(self, msg):
        if self.encoding is None:
            self.encoding = msg.encoding

        h, w = msg.height, msg.width
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = arr.reshape(h, msg.step // 3, 3)[:, :w]

        if self.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # =========================
        # EDGE LANE (원본 느낌 유지)
        # =========================
        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        col_sum = np.sum(binary > 0, axis=0)
        max_val = np.max(col_sum) if col_sum.size > 0 else 0

        if max_val < 5:
            self.current_lin = 0.10
            self.current_ang = 0.25
        else:
            idx = np.where(col_sum >= max_val * 0.3)[0]
            if idx.size > 0:
                center = w / 2.0
                track_center = np.mean(idx)
                offset = track_center - center

                self.current_lin = 0.22
                self.current_ang = -0.008 * offset
            else:
                self.current_lin = 0.10
                self.current_ang = 0.25

        # =========================
        # OBSTACLE OVERRIDE (딱 이것만 추가)
        # =========================
        if self.front < 0.35:
            self.current_lin = 0.05
            self.current_ang = 0.45   # 그냥 비켜가기

        # =========================
        # PUBLISH
        # =========================
        self.cmd.linear.x = self.current_lin
        self.cmd.angular.z = self.current_ang
        self.cmd_pub.publish(self.cmd)


if __name__ == "__main__":
    EdgeLaneNoBridge()
    rospy.spin()
