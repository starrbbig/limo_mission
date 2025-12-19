#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class LineTracerWithObstacleAvoidance:
    def __init__(self):
        rospy.init_node("line_tracer_with_obstacle_avoidance")
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)

        self.bridge = CvBridge()

        # LIDAR
        self.scan_ranges = []
        self.front = 999.0

        # 상태
        self.state = "LANE"
        self.escape_angle = 0.0
        self.state_start = rospy.Time.now().to_sec()

        # 라바콘 이후 모드
        self.after_cone = False   # ⭐ 핵심

        # 파라미터
        self.robot_width = 0.14

        # 흰색 차선
        self.base_gain = 1.0 / 220.0
        self.corner_scale = 140.0
        self.max_steer = 0.85
        self.left_delay_start = None
        self.left_delay_time = 0.6
        self.min_line_area = 300

        # 검은색 바닥
        self.black_speed = 0.20
        self.k_black = 0.010

    # ============================================================
    # LIDAR
  
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # CAMERA
   
    def camera_cb(self, msg):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if self.state == "BACK":
            self.back_control()
            return

        if self.state == "ESCAPE":
            self.escape_control()
            return

        # 장애물 감지
        if self.front < 0.30:
            self.state = "BACK"
            self.state_start = now
            return

        frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.55):h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ========================================================
        # 라바콘 감지 (항상 체크)
   
        lower_r1 = np.array([0, 120, 80])
        upper_r1 = np.array([10, 255, 255])
        lower_r2 = np.array([170, 120, 80])
        upper_r2 = np.array([180, 255, 255])

        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_r1, upper_r1),
            cv2.inRange(hsv, lower_r2, upper_r2)
        )

        red_contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        centers = []
        for c in red_contours:
            if cv2.contourArea(c) < 200:
                continue
            M = cv2.moments(c)
            if M["m00"] != 0:
                centers.append(int(M["m10"] / M["m00"]))

   
        # 라바콘 주행 (처음~중간)
        # ========================================================
        if len(centers) >= 1:
            self.after_cone = True   # ⭐ 여기서 True로 고정

            if len(centers) >= 2:
                centers.sort()
                mid = (centers[0] + centers[-1]) // 2
            else:
                mid = centers[0]

            error = mid - (w // 2)
            twist.linear.x = 0.22
            twist.angular.z = error / 180.0
            self.pub.publish(twist)
            return

        # 라바콘 이후 → 검은색 바닥 ONLY
        # ========================================================
        if self.after_cone:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            mask = binary > 0
            col_sum = np.sum(mask, axis=0)

            if np.max(col_sum) < 5:
                twist.linear.x = self.black_speed
                twist.angular.z = 0.0
                self.pub.publish(twist)
                return

            cand = np.where(col_sum >= np.max(col_sum) * 0.3)[0]
            cx = np.sum(cand * col_sum[cand]) / np.sum(col_sum[cand])

            error = cx - (w / 2)
            twist.linear.x = self.black_speed
            twist.angular.z = -self.k_black * error
            twist.angular.z = max(min(twist.angular.z, 0.8), -0.8)
            self.pub.publish(twist)
            return

        
        # 라바콘 이전 → 흰색 차선
        # ========================================================
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        contours, _ = cv2.findContours(
            mask_white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            twist.linear.x = 0.12
            twist.angular.z = 0.0
            self.pub.publish(twist)
            return

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_line_area:
            twist.linear.x = 0.12
            twist.angular.z = 0.0
            self.pub.publish(twist)
            return

        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            error = cx - (w // 2)
            gain = self.base_gain * (1.0 + abs(error) / self.corner_scale)
            twist.linear.x = 0.22
            twist.angular.z = max(
                min(gain * error, self.max_steer),
                -self.max_steer
            )
            self.pub.publish(twist)

    # BACK / ESCAPE
    # ============================================================
    def back_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.2:
            twist.linear.x = -0.24
            twist.angular.z = 0.0
            self.pub.publish(twist)
        else:
            self.escape_angle = self.find_gap()
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.2:
            twist.linear.x = 0.20
            twist.angular.z = self.escape_angle * 1.5
            self.pub.publish(twist)
        else:
            self.state = "LANE"

    def find_gap(self):
        if len(self.scan_ranges) == 0:
            return 0.0

        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.where((ranges < 0.20) | np.isnan(ranges), 0.0, ranges)

        idx = np.argmax(ranges)
        if ranges[idx] < self.robot_width + 0.10:
            return 0.0

        return (idx - 60) * np.pi / 180


if __name__ == "__main__":
    LineTracerWithObstacleAvoidance()
    rospy.spin()

