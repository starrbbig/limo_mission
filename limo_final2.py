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
        rospy.Subscriber("/usb_cam/image_raw/compressed",
                         CompressedImage, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)

        self.bridge = CvBridge()

        # =====================
        # Parameters
        # =====================
        self.speed = 0.24
        self.robot_width = 0.13

        self.scan_ranges = []
        self.front = 999.0

        # FSM
        self.state = "LANE"
        self.state_start = rospy.Time.now().to_sec()
        self.escape_angle = 0.0

        # Escape direction logic
        self.left_escape_count = 0
        self.force_right_escape = 0

    # ============================================================
    # LIDAR CALLBACK
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        # 정면 ±10도
        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # CAMERA CALLBACK
    # ============================================================
    def camera_cb(self, msg):
        now = rospy.Time.now().to_sec()

        if self.state == "ESCAPE":
            self.escape_control()
            return

        if self.state == "BACK":
            self.back_control()
            return

        twist = Twist()

        # =====================
        # LANE MODE
        # =====================
        if self.front < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        h, w = frame.shape[:2]

        # ROI
        roi_near = frame[int(h * 0.55):h, :]
        roi_mid = frame[int(h * 0.35):int(h * 0.55), :]

        hsv_near = cv2.cvtColor(roi_near, cv2.COLOR_BGR2HSV)
        hsv_mid = cv2.cvtColor(roi_mid, cv2.COLOR_BGR2HSV)

        # =====================
        # White lane mask
        # =====================
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])

        mask_near = cv2.inRange(hsv_near, lower_white, upper_white)
        mask_mid = cv2.inRange(hsv_mid, lower_white, upper_white)

        contours_near, _ = cv2.findContours(
            mask_near, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # =====================
        # Cone (red) detection
        # =====================
        lower_r1 = np.array([0, 120, 80])
        upper_r1 = np.array([10, 255, 255])
        lower_r2 = np.array([170, 120, 80])
        upper_r2 = np.array([180, 255, 255])

        mask_r = cv2.bitwise_or(
            cv2.inRange(hsv_near, lower_r1, upper_r1),
            cv2.inRange(hsv_near, lower_r2, upper_r2)
        )

        red_contours, _ = cv2.findContours(
            mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(red_contours) >= 1:
            centers = []
            for cnt in red_contours:
                if cv2.contourArea(cnt) < 200:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                centers.append(int(M["m10"] / M["m00"]))

            if centers:
                mid = (min(centers) + max(centers)) // 2
                error = mid - (w // 2)

                twist.linear.x = 0.21
                twist.angular.z = error / 180.0
                self.pub.publish(twist)
                return

        # =====================
        # Recovery logic
        # =====================
        if len(contours_near) == 0 and cv2.countNonZero(mask_mid) > 250:
            twist.linear.x = 0.12
            twist.angular.z = -0.55
            self.pub.publish(twist)
            return

        if len(contours_near) == 0:
            twist.linear.x = 0.10
            twist.angular.z = 0.25
            self.pub.publish(twist)
            return

        # =====================
        # Normal lane tracing
        # =====================
        c = max(contours_near, key=cv2.contourArea)
        M = cv2.moments(c)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            error = cx - (w // 2)

            twist.linear.x = 0.22
            twist.angular.z = error / 200.0
            self.pub.publish(twist)

    # ============================================================
    # BACK MODE
    # ============================================================
    def back_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.4:
            twist.linear.x = -0.24
            self.pub.publish(twist)
        else:
            angle = self.find_gap_max()
            angle = self.apply_escape_direction_logic(angle)

            self.escape_angle = angle
            self.state = "ESCAPE"
            self.state_start = now

    # ============================================================
    # ESCAPE MODE
    # ============================================================
    def escape_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.0:
            twist.linear.x = 0.19
            twist.angular.z = self.escape_angle * 1.3
            self.pub.publish(twist)
        else:
            self.state = "LANE"

    # ============================================================
    # Escape direction logic
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

    # ============================================================
    # Gap finder
    # ============================================================
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


if __name__ == "__main__":
    LineTracerWithObstacleAvoidance()
    rospy.spin()


