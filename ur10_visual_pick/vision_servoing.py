import json
import os
import re
import time

import cv2
import numpy as np
import rclpy
import rtde_control
import rtde_receive
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from ur10_visual_pick.robotiq_3f_gripper import RobotiqGripper3F


class UR10VisualServoing(Node):
    def __init__(self):
        super().__init__("ur10_visual_servoing")

        # --- CONFIGURATION ---
        self.robot_ip = "192.168.1.102"
        self.gripper_ip = "192.168.1.105"

        self.PICK_Z = 0.40
        self.SAFE_Z = 0.60
        self.SEARCH_X = 0.40
        self.SEARCH_Y = 0.00
        self.SEARCH_Z = 1.00
        self.SEARCH_RVEC = [0.0, 0.0, 0.0]

        self.PLACE_X = 0.35
        self.PLACE_Y = -0.65
        self.PLACE_Z = 0.42
        self.PLACE_APPROACH_Z = 0.60

        # Servo gains tuned for REF_FRAME_* resolution; auto-scaled at runtime.
        self.GAIN = 0.00015
        self.THRESHOLD = 6
        self.REF_FRAME_W = 1280
        self.REF_FRAME_H = 720
        self.CENTER_BIAS_U_PX = 0
        self.CENTER_BIAS_V_PX = 100
        self.MIN_THRESHOLD_PX = 3
        self.MAX_SERVO_SPEED = 0.08

        self.THRESHOLD_VALUE = 70
        self.MIN_CONTOUR_AREA = 3000
        self.MIN_SIDE_PIXELS = 10
        self.BLUR_KERNEL = (15, 15)
        self.MORPH_KERNEL = np.ones((7, 7), np.uint8)
        self.COLOR_MORPH_KERNEL = np.ones((3, 3), np.uint8)

        # Gripper visible in top-right area.
        self.EXCLUDE_TOP_RIGHT_X_RATIO = 0.80
        self.EXCLUDE_TOP_RIGHT_Y_RATIO = 0.20
        self.EXCLUDE_LEFT_OFFSET_PX = 160
        self.EXCLUDE_RIGHT_OFFSET_PX = 120

        # Wrist3 orientation behavior.
        self.WRIST3_SIGN = -1.0
        self.WRIST3_OFFSET_DEG = 90.0
        self.WRIST3_MAX_ROT_DEG = 120.0

        # Selection mode and anti-retracking behavior.
        self.SELECTED_COLOR_ONLY = True
        self.COLOR_H_TOL = 12
        self.COLOR_S_TOL = 80
        self.COLOR_V_TOL = 80
        self.COLOR_H_TOL_WIDE = 20
        self.COLOR_S_TOL_WIDE = 110
        self.COLOR_V_TOL_WIDE = 110
        self.MIN_CLICK_COMPONENT_AREA = 250
        self.REQUIRE_RESELECTION_AFTER_PLACE = True
        self.POST_PICK_COOLDOWN_SEC = 1.5
        self.FINGER_OPEN_THRESHOLD = 20

        # Voice command JSON produced externally (UR_llm pipeline).
        self.VOICE_COMMAND_JSON = "/tmp/ur10_voice_command.json"
        self.voice_last_mtime = 0.0

        self.latest_hsv = None
        self.selected_hsv = None
        self.selected_color_name = None
        self.selected_source = None  # "click" or "voice"
        self.selected_click_point = None
        self.last_target_center = None
        self.color_presets = {
            "blue": [(np.array([100, 120, 50]), np.array([140, 255, 255]))],
            "red": [
                (np.array([0, 120, 50]), np.array([10, 255, 255])),
                (np.array([170, 120, 50]), np.array([180, 255, 255])),
            ],
            "green": [(np.array([40, 80, 40]), np.array([85, 255, 255]))],
            "yellow": [(np.array([20, 100, 80]), np.array([35, 255, 255]))],
        }

        self.tracking_paused = False
        self.is_picking = False
        self.post_pick_resume_time = 0.0
        self.rtde_c = None

        # --- HARDWARE INITIALIZATION ---
        try:
            self.get_logger().info("Connecting to UR10...")
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            curr_pose = self.rtde_r.getActualTCPPose()
            self.SEARCH_X = curr_pose[0]
            self.SEARCH_Y = curr_pose[1]
            self.SEARCH_RVEC = [curr_pose[3], curr_pose[4], curr_pose[5]]
            self.get_logger().info(
                f"Search pose initialized to x={self.SEARCH_X:.3f}, y={self.SEARCH_Y:.3f}, z={self.SEARCH_Z:.3f}"
            )

            self.get_logger().info("Connecting to Robotiq 3F Gripper...")
            self.gripper = RobotiqGripper3F(ip=self.gripper_ip)
            if self.gripper.activate():
                self.get_logger().info("Gripper activated. Opening fingers...")
                self.gripper.open()
            else:
                self.get_logger().error("Gripper activation failed.")

        except Exception as exc:
            self.get_logger().error(f"Hardware connection failed: {exc}")

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/oak/rgb/image_raw", self.process_frame, 10)
        self.voice_timer = self.create_timer(0.5, self.poll_voice_command_file)

        cv2.namedWindow("Tracking")
        cv2.setMouseCallback("Tracking", self.on_mouse_click)
        self.get_logger().info("Visual Servoing Node started.")

    def process_frame(self, msg):
        if self.is_picking or self.rtde_c is None:
            return

        if time.monotonic() < self.post_pick_resume_time:
            self.rtde_c.speedStop()
            return

        if self.object_likely_held():
            self.rtde_c.speedStop()
            if not self.tracking_paused:
                self.get_logger().warn("Object still held. Tracking paused to prevent self-chasing.")
                self.tracking_paused = True
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = frame.shape
        center_u, center_v = w // 2, h // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.latest_hsv = hsv

        blur = cv2.GaussianBlur(gray, self.BLUR_KERNEL, 0)
        _, thresh = cv2.threshold(blur, self.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.MORPH_KERNEL, iterations=2)
        thresh = cv2.dilate(thresh, self.MORPH_KERNEL, iterations=1)

        selection_mask = self.build_selection_mask(hsv)
        if selection_mask is not None:
            # For explicit color-lock mode, use color mask directly. This is
            # more robust on DepthAI v3 than intersecting with grayscale thresh.
            detect_mask = cv2.morphologyEx(selection_mask, cv2.MORPH_OPEN, self.COLOR_MORPH_KERNEL, iterations=1)
            detect_mask = cv2.dilate(detect_mask, self.COLOR_MORPH_KERNEL, iterations=1)
        elif self.SELECTED_COLOR_ONLY:
            detect_mask = np.zeros_like(thresh)
        else:
            detect_mask = thresh

        target = self.find_target_contour(detect_mask, w, h)

        cv2.drawMarker(frame, (center_u, center_v), (255, 0, 0), cv2.MARKER_CROSS, 30, 2)
        exclude_x = int(w * self.EXCLUDE_TOP_RIGHT_X_RATIO)
        exclude_y = int(h * self.EXCLUDE_TOP_RIGHT_Y_RATIO)
        exclude_left = max(0, exclude_x - self.EXCLUDE_LEFT_OFFSET_PX)
        exclude_right = max(exclude_left + 1, w - self.EXCLUDE_RIGHT_OFFSET_PX)
        cv2.rectangle(frame, (exclude_left, 0), (exclude_right, exclude_y), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "Excluded (gripper)",
            (exclude_left + 5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        if self.selected_source == "click" and self.selected_hsv is not None:
            cv2.putText(
                frame,
                f"Lock: click HSV={self.selected_hsv}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif self.selected_source == "voice" and self.selected_color_name:
            cv2.putText(
                frame,
                f"Lock: voice color={self.selected_color_name}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif self.SELECTED_COLOR_ONLY:
            cv2.putText(
                frame,
                "Selection mode: left-click object / voice select",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 180, 255),
                1,
                cv2.LINE_AA,
            )

        if self.tracking_paused:
            cv2.putText(
                frame,
                "Tracking paused",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            self.rtde_c.speedStop()
        elif target is not None:
            obj_u, obj_v = target["center"]
            self.last_target_center = (obj_u, obj_v)
            cv2.drawContours(frame, [target["box"]], 0, (0, 255, 0), 2)
            cv2.circle(frame, (obj_u, obj_v), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"A:{int(target['area'])} ang:{target['grasp_angle']:.1f}",
                (obj_u - 90, max(15, obj_v - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            bias_u = int(round(self.CENTER_BIAS_U_PX * (w / float(self.REF_FRAME_W))))
            bias_v = int(round(self.CENTER_BIAS_V_PX * (h / float(self.REF_FRAME_H))))
            err_u = obj_u - center_u + bias_u
            err_v = obj_v - center_v + bias_v

            threshold_u = max(self.MIN_THRESHOLD_PX, int(round(self.THRESHOLD * (w / float(self.REF_FRAME_W)))))
            threshold_v = max(self.MIN_THRESHOLD_PX, int(round(self.THRESHOLD * (h / float(self.REF_FRAME_H)))))

            cv2.putText(
                frame,
                f"{w}x{h} err=({err_u},{err_v}) th=({threshold_u},{threshold_v})",
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

            if abs(err_u) < threshold_u and abs(err_v) < threshold_v:
                self.is_picking = True
                self.execute_pick_and_place(target["grasp_angle"])
            else:
                self.servoing_move(err_u, err_v, w, h)
        else:
            self.rtde_c.speedStop()

        cv2.imshow("Tracking", frame)
        cv2.imshow("Contours", detect_mask)
        cv2.waitKey(1)

    def servoing_move(self, err_u, err_v, frame_w, frame_h):
        scale_u = self.REF_FRAME_W / float(max(1, frame_w))
        scale_v = self.REF_FRAME_H / float(max(1, frame_h))
        speed_u = err_u * self.GAIN * scale_u
        speed_v = err_v * self.GAIN * scale_v
        speed_u = float(np.clip(speed_u, -self.MAX_SERVO_SPEED, self.MAX_SERVO_SPEED))
        speed_v = float(np.clip(speed_v, -self.MAX_SERVO_SPEED, self.MAX_SERVO_SPEED))
        self.rtde_c.speedL([speed_u, -speed_v, 0.0, 0.0, 0.0, 0.0], 0.3, 0.1)

    def on_mouse_click(self, event, x, y, flags, param):
        if self.latest_hsv is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            h_img, w_img = self.latest_hsv.shape[:2]
            x0, x1 = max(0, x - 3), min(w_img, x + 4)
            y0, y1 = max(0, y - 3), min(h_img, y + 4)
            patch = self.latest_hsv[y0:y1, x0:x1].reshape(-1, 3)
            if patch.size == 0:
                return
            sat_mask = patch[:, 1] > 40
            ref = patch[sat_mask] if np.any(sat_mask) else patch
            h = int(np.median(ref[:, 0]))
            s = int(np.median(ref[:, 1]))
            v = int(np.median(ref[:, 2]))
            self.selected_hsv = (h, s, v)
            self.selected_color_name = None
            self.selected_source = "click"
            self.selected_click_point = (x, y)
            self.last_target_center = None
            self.tracking_paused = False
            self.get_logger().info(f"Click-selected HSV at ({x},{y}): {self.selected_hsv}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_hsv = None
            self.selected_color_name = None
            self.selected_source = None
            self.selected_click_point = None
            self.last_target_center = None
            self.get_logger().info("Selection lock cleared.")

    def build_selection_mask(self, hsv):
        if self.selected_hsv is not None:
            h, s, v = self.selected_hsv
            mask = self.mask_from_hsv_tolerance(hsv, h, s, v, self.COLOR_H_TOL, self.COLOR_S_TOL, self.COLOR_V_TOL)
            component_mask = self.mask_component_for_click(mask)
            if component_mask is not None:
                return component_mask

            # Fallback for DepthAI v3 color jitter: widen tolerance once.
            wide_mask = self.mask_from_hsv_tolerance(
                hsv, h, s, v, self.COLOR_H_TOL_WIDE, self.COLOR_S_TOL_WIDE, self.COLOR_V_TOL_WIDE
            )
            component_mask = self.mask_component_for_click(wide_mask)
            if component_mask is not None:
                return component_mask

            return wide_mask

        if self.selected_color_name in self.color_presets:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in self.color_presets[self.selected_color_name]:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
            return mask
        return None

    def mask_from_hsv_tolerance(self, hsv, h, s, v, h_tol, s_tol, v_tol):
        h_low = max(0, int(h) - int(h_tol))
        h_high = min(179, int(h) + int(h_tol))
        s_low = max(0, int(s) - int(s_tol))
        s_high = min(255, int(s) + int(s_tol))
        v_low = max(0, int(v) - int(v_tol))
        v_high = min(255, int(v) + int(v_tol))

        if h_low <= h_high:
            return cv2.inRange(
                hsv,
                np.array([h_low, s_low, v_low], dtype=np.uint8),
                np.array([h_high, s_high, v_high], dtype=np.uint8),
            )

        mask1 = cv2.inRange(
            hsv,
            np.array([0, s_low, v_low], dtype=np.uint8),
            np.array([h_high, s_high, v_high], dtype=np.uint8),
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([h_low, s_low, v_low], dtype=np.uint8),
            np.array([179, s_high, v_high], dtype=np.uint8),
        )
        return cv2.bitwise_or(mask1, mask2)

    def mask_component_for_click(self, mask):
        if self.selected_click_point is None:
            return mask

        x, y = self.selected_click_point
        h, w = mask.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return None

        label_at_click = int(labels[y, x])
        if label_at_click > 0 and int(stats[label_at_click, cv2.CC_STAT_AREA]) >= self.MIN_CLICK_COMPONENT_AREA:
            out = np.zeros_like(mask)
            out[labels == label_at_click] = 255
            return out

        best_label = None
        best_dist2 = None
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.MIN_CLICK_COMPONENT_AREA:
                continue
            cx, cy = centroids[label]
            dx = float(cx) - float(x)
            dy = float(cy) - float(y)
            dist2 = dx * dx + dy * dy
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best_label = label

        if best_label is None:
            return None

        out = np.zeros_like(mask)
        out[labels == best_label] = 255
        return out

    def poll_voice_command_file(self):
        path = self.VOICE_COMMAND_JSON
        if not os.path.exists(path):
            return
        try:
            mtime = os.path.getmtime(path)
            if mtime <= self.voice_last_mtime:
                return
            self.voice_last_mtime = mtime

            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                return

            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                if not match:
                    self.get_logger().warn("Voice file updated but no JSON object found.")
                    return
                cmd = json.loads(match.group(0))

            action = str(cmd.get("action", "")).strip().lower()
            obj = str(cmd.get("object", "")).strip().lower()

            if action in {"pause", "stop", "hold"}:
                self.tracking_paused = True
                self.get_logger().info("Voice: tracking paused.")
                return
            if action in {"resume", "start", "continue"}:
                self.tracking_paused = False
                self.get_logger().info("Voice: tracking resumed.")
                return
            if action in {"clear", "reset"}:
                self.selected_hsv = None
                self.selected_color_name = None
                self.selected_source = None
                self.get_logger().info("Voice: selection cleared.")
                return

            if action in {"pick", "select", "track", "target"}:
                for color_name in self.color_presets:
                    if color_name in obj:
                        self.selected_hsv = None
                        self.selected_color_name = color_name
                        self.selected_source = "voice"
                        self.tracking_paused = False
                        self.get_logger().info(f"Voice: tracking color '{color_name}'.")
                        return

        except Exception as exc:
            self.get_logger().error(f"Voice command polling failed: {exc}")

    def object_likely_held(self) -> bool:
        try:
            status = self.gripper.get_status()
            if not status:
                return False
            contact = self.gripper.contact_detected_while_closing(status)
            finger_pos = int(status.get("finger_a_position", 0))
            return contact and finger_pos > self.FINGER_OPEN_THRESHOLD
        except Exception:
            return False

    def find_target_contour(self, thresh, frame_w, frame_h):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        exclude_x = int(frame_w * self.EXCLUDE_TOP_RIGHT_X_RATIO)
        exclude_y = int(frame_h * self.EXCLUDE_TOP_RIGHT_Y_RATIO)
        exclude_left = max(0, exclude_x - self.EXCLUDE_LEFT_OFFSET_PX)
        exclude_right = max(exclude_left + 1, frame_w - self.EXCLUDE_RIGHT_OFFSET_PX)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_CONTOUR_AREA:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            if rw < self.MIN_SIDE_PIXELS or rh < self.MIN_SIDE_PIXELS:
                continue

            if exclude_left <= cx <= exclude_right and cy <= exclude_y:
                continue

            if rw < rh:
                grasp_angle = angle
            else:
                grasp_angle = angle + 90.0

            while grasp_angle > 90.0:
                grasp_angle -= 180.0
            while grasp_angle < -90.0:
                grasp_angle += 180.0

            box = cv2.boxPoints(rect).astype(int)
            cx_i, cy_i = int(cx), int(cy)

            candidates.append(
                {
                    "contour": contour,
                    "area": area,
                    "center": (cx_i, cy_i),
                    "box": box,
                    "grasp_angle": grasp_angle,
                }
            )

        if not candidates:
            return None

        # Highest priority: the contour that contains the exact click point.
        if self.selected_source == "click" and self.selected_click_point is not None:
            px, py = self.selected_click_point
            for c in candidates:
                inside = cv2.pointPolygonTest(c["contour"], (float(px), float(py)), False)
                if inside >= 0:
                    return c

        # Next priority for click mode: contour nearest the previously tracked center.
        if self.selected_source == "click" and self.last_target_center is not None:
            lx, ly = self.last_target_center
            return min(candidates, key=lambda c: (c["center"][0] - lx) ** 2 + (c["center"][1] - ly) ** 2)

        # Fallback: largest area.
        return max(candidates, key=lambda c: c["area"])

    def rotate_pose_about_tool_z(self, pose, delta_deg):
        if abs(delta_deg) < 0.5:
            return list(pose)

        delta_deg = float(np.clip(delta_deg, -self.WRIST3_MAX_ROT_DEG, self.WRIST3_MAX_ROT_DEG))
        rotvec = np.array(pose[3:6], dtype=np.float64).reshape(3, 1)
        r_current, _ = cv2.Rodrigues(rotvec)

        theta = np.deg2rad(delta_deg)
        r_tool_z = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        r_new = r_current @ r_tool_z
        rotvec_new, _ = cv2.Rodrigues(r_new)

        rotated_pose = list(pose)
        rotated_pose[3] = float(rotvec_new[0, 0])
        rotated_pose[4] = float(rotvec_new[1, 0])
        rotated_pose[5] = float(rotvec_new[2, 0])
        return rotated_pose

    def move_to_search_pose(self, reference_pose):
        search_pose = list(reference_pose)
        search_pose[0] = self.SEARCH_X
        search_pose[1] = self.SEARCH_Y
        search_pose[2] = self.SEARCH_Z
        search_pose[3] = self.SEARCH_RVEC[0]
        search_pose[4] = self.SEARCH_RVEC[1]
        search_pose[5] = self.SEARCH_RVEC[2]
        self.get_logger().info(
            f"Returning to search pose: x={self.SEARCH_X:.3f}, y={self.SEARCH_Y:.3f}, z={self.SEARCH_Z:.3f}"
        )
        self.rtde_c.moveL(search_pose, 0.15, 0.30)

    def execute_pick_and_place(self, grasp_angle_deg):
        self.get_logger().info("Target centered! Starting pick-and-place sequence...")
        try:
            self.rtde_c.speedStop()

            curr_pose = self.rtde_r.getActualTCPPose()
            wrist_delta = self.WRIST3_SIGN * (grasp_angle_deg + self.WRIST3_OFFSET_DEG)

            align_reference_pose = list(curr_pose)
            align_reference_pose[3] = self.SEARCH_RVEC[0]
            align_reference_pose[4] = self.SEARCH_RVEC[1]
            align_reference_pose[5] = self.SEARCH_RVEC[2]

            align_pose = self.rotate_pose_about_tool_z(align_reference_pose, wrist_delta)
            self.get_logger().info(
                f"Aligning wrist3 by {float(np.clip(wrist_delta, -self.WRIST3_MAX_ROT_DEG, self.WRIST3_MAX_ROT_DEG)):.1f} deg"
            )
            self.rtde_c.moveL(align_pose, 0.10, 0.20)

            pick_pose = list(align_pose)
            pick_pose[2] = self.PICK_Z
            self.get_logger().info(f"Descending to pick Z: {self.PICK_Z}")
            self.rtde_c.moveL(pick_pose, 0.10, 0.20)

            self.get_logger().info("Closing gripper...")
            close_ok = self.gripper.close(force=10, speed=200)
            if not close_ok:
                status = self.gripper.get_status()
                if self.gripper.contact_detected_while_closing(status):
                    self.get_logger().warn("Gripper close timed out, but contact detected. Continuing.")
                else:
                    self.get_logger().error("Gripper close failed with no contact. Aborting place.")
                    self.move_to_search_pose(curr_pose)
                    return

            lift_pose = list(pick_pose)
            lift_pose[2] = self.SAFE_Z
            self.get_logger().info(f"Lifting to safe Z: {self.SAFE_Z}")
            self.rtde_c.moveL(lift_pose, 0.10, 0.20)

            place_approach_pose = list(lift_pose)
            place_approach_pose[0] = self.PLACE_X
            place_approach_pose[1] = self.PLACE_Y
            place_approach_pose[2] = self.PLACE_APPROACH_Z
            self.get_logger().info(
                f"Moving to place approach pose: x={self.PLACE_X:.3f}, y={self.PLACE_Y:.3f}, z={self.PLACE_APPROACH_Z:.3f}"
            )
            self.rtde_c.moveL(place_approach_pose, 0.15, 0.30)

            place_pose = list(place_approach_pose)
            place_pose[2] = self.PLACE_Z
            self.get_logger().info(f"Descending to place Z: {self.PLACE_Z}")
            self.rtde_c.moveL(place_pose, 0.08, 0.20)

            self.get_logger().info("Opening gripper to place object...")
            if not self.gripper.open(speed=200):
                self.get_logger().error("Gripper open failed while placing.")

            self.get_logger().info("Retracting after place...")
            self.rtde_c.moveL(place_approach_pose, 0.10, 0.25)

            self.move_to_search_pose(place_approach_pose)

            if self.object_likely_held():
                self.get_logger().error("Object still appears held after place. Tracking paused.")
                self.tracking_paused = True
            elif self.REQUIRE_RESELECTION_AFTER_PLACE:
                self.selected_hsv = None
                self.selected_color_name = None
                self.selected_source = None
                self.tracking_paused = True
                self.get_logger().info("Selection mode re-armed at home pose. Click/select next object.")

            self.get_logger().info("Pick-and-place complete.")

        except Exception as exc:
            self.get_logger().error(f"Pick-and-place failed: {exc}")
        finally:
            self.is_picking = False
            self.post_pick_resume_time = time.monotonic() + self.POST_PICK_COOLDOWN_SEC


def main():
    rclpy.init()
    node = UR10VisualServoing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.rtde_c:
            node.rtde_c.speedStop()
            node.rtde_c.stopScript()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
