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
        self.state = "LANE"        # LANE / OBSTACLE
        self.state_start = rospy.Time.now().to_sec()

        # ================= LANE PARAM (유지) =================
        self.k_angle = 0.010
        self.forward_speed = 0.22
        self.search_spin_speed = 0.25

        # ================= LIDAR =================
        self.scan = []
        self.obstacle_dir = 0.0

        # ================= IMAGE =================
        self.encoding = None

        rospy.loginfo("🔥 FINAL EXAM CONTROLLER READY")

    # ============================================================
    # LiDAR
    # ============================================================
    def lidar_cb(self, scan):
        self.scan = np.array(scan.ranges)

    # ============================================================
    # MAIN CALLBACK
    # ============================================================
    def image_cb(self, msg):
        cmd = Twist()
        now = rospy.Time.now().to_sec()

        img = self.msg_to_cv2(msg)
        if img is None:
            return

        # ===============================
        # 1️⃣ 장애물 상태
        # ===============================
        if self.state == "OBSTACLE":
            if now - self.state_start < 1.2:
                cmd.linear.x = 0.18
                cmd.angular.z = self.obstacle_dir
                self.pub.publish(cmd)
                return
            else:
                self.state = "LANE"

        # ===============================
        # 2️⃣ 장애물 감지 (전방 가까움)
        # ===============================
        if self.is_obstacle_ahead():
            self.obstacle_dir = self.choose_gap_direction()
            self.state = "OBSTACLE"
            self.state_start = now
            return

        # ===============================
        # 3️⃣ 라바콘 (유지)
        # ===============================
        cone = self.detect_cone(img)
        if cone is not None:
            lin, ang = cone
            cmd.linear.x = lin
            cmd.angular.z = ang
            self.pub.publish(cmd)
            return

        # ===============================
        # 4️⃣ 정상 라인트레이싱
        # ===============================
        lin, ang = self.edge_lane_control(img)
        cmd.linear.x = lin
        cmd.angular.z = ang
        self.pub.publish(cmd)

    # ============================================================
    # OBSTACLE 판단
    # ============================================================
    def is_obstacle_ahead(self):
        if len(self.scan) == 0:
            return False

        front = np.concatenate([self.scan[:8], self.scan[-8:]])
        front = front[(front > 0.12) & (front < 0.5)]
        return len(front) > 5

    def choose_gap_direction(self):
        # 좌 / 중 / 우 섹터
        left  = self.scan[30:60]
        mid   = np.concatenate([self.scan[:10], self.scan[-10:]])
        right = self.scan[-60:-30]

        def score(arr):
            arr = arr[(arr > 0.15) & (arr < 2.0)]
            return np.mean(arr) if len(arr) else 0.0

        scores = {
            "L": score(left),
            "C": score(mid),
            "R": score(right)
        }

        best = max(scores, key=scores.get)

        if best == "L":
            return +0.45
        elif best == "R":
            return -0.45
        else:
            return 0.0

    # ============================================================
    # CONE (4번)
    # ============================================================
    def detect_cone(self, img):
        h, w = img.shape[:2]
        roi = img[int(h*0.55):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0,120,80])
        upper1 = np.array([10,255,255])
        lower2 = np.array([170,120,80])
        upper2 = np.array([180,255,255])

        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 300]

        if not cnts:
            return None

        centers = [int(cv2.moments(c)["m10"] / cv2.moments(c)["m00"]) for c in cnts]
        mid = (min(centers) + max(centers)) // 2
        error = mid - (w // 2)

        return 0.20, error / 170.0

    # ============================================================
    # EDGE LANE (절대 수정 X)
    # ============================================================
    def edge_lane_control(self, img):
        h, w, _ = img.shape
        center = w / 2.0

        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        col_sum = np.sum(binary > 0, axis=0)
        if np.max(col_sum) < 5:
            return 0.10, self.search_spin_speed

        idx = np.where(col_sum >= max(5, int(np.max(col_sum)*0.3)))[0]
        track_center = np.sum(idx * col_sum[idx]) / np.sum(col_sum[idx])

        offset = (track_center - center) + 5.0
        ang = -self.k_angle * offset
        return self.forward_speed, np.clip(ang, -0.8, 0.8)

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


if __name__ == "__main__":
    rospy.spin()
