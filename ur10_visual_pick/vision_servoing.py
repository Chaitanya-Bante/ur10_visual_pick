import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import rtde_control
import rtde_receive

# Import the gripper class from your local file
from ur10_visual_pick.robotiq_3f_gripper import RobotiqGripper3F 

class UR10VisualServoing(Node):
    def __init__(self):
        super().__init__('ur10_visual_servoing')
        
        # --- CONFIGURATION ---
        self.robot_ip = "192.168.1.102" 
        self.gripper_ip = "192.168.1.105"
        self.PICK_Z = 0.40    # Depth to reach the object
        self.SAFE_Z = 0.60    # Height to lift after picking
        self.SEARCH_X = 0.40  # Search X in base frame (updated from startup pose if possible)
        self.SEARCH_Y = 0.00  # Search Y in base frame (updated from startup pose if possible)
        self.SEARCH_Z = 1.00  # Hover height for searching
        self.SEARCH_RVEC = [0.0, 0.0, 0.0]  # Reference orientation (rotvec) for repeatable picks
        self.PLACE_X = 0.35   # Set your desired drop X (base frame)
        self.PLACE_Y = -0.65  # Set your desired drop Y (base frame)
        self.PLACE_Z = 0.42   # Drop height
        self.PLACE_APPROACH_Z = 0.60
        self.CONTINUOUS_MODE = True
        self.WRIST3_SIGN = -1.0         # Flip sign if rotation is opposite on your setup
        self.WRIST3_OFFSET_DEG = 0.0    # Static calibration offset for wrist3
        self.WRIST3_MAX_ROT_DEG = 60.0  # Clamp wrist rotation per pick

        self.GAIN = 0.00015   
        self.THRESHOLD = 6    

        # Contour-based object detection parameters
        self.THRESHOLD_VALUE = 70
        self.MIN_CONTOUR_AREA = 3000
        self.MIN_SIDE_PIXELS = 10
        self.BLUR_KERNEL = (15, 15)
        self.MORPH_KERNEL = np.ones((7, 7), np.uint8)

        # Exclude top-right region where gripper appears in camera view
        self.EXCLUDE_TOP_RIGHT_X_RATIO = 0.80
        self.EXCLUDE_TOP_RIGHT_Y_RATIO = 0.20
        
        self.is_picking = False
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
                self.get_logger().info("Gripper Activated. Opening fingers...")
                self.gripper.open()
            else:
                self.get_logger().error("Gripper Activation Failed!")
                
        except Exception as e:
            self.get_logger().error(f"Hardware Connection Failed: {e}")
        
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, 
            '/oak/rgb/image_raw', 
            self.process_frame, 
            10
        )
        self.get_logger().info("Visual Servoing Node Started.")
        
    def process_frame(self, msg):
        # Ignore frames if we are in the middle of a pick sequence
        if self.is_picking or self.rtde_c is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = frame.shape
        center_u, center_v = w // 2, h // 2

        # Contour-based segmentation for "any object" detection.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.BLUR_KERNEL, 0)
        _, thresh = cv2.threshold(blur, self.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.MORPH_KERNEL, iterations=2)
        thresh = cv2.dilate(thresh, self.MORPH_KERNEL, iterations=1)

        target = self.find_target_contour(thresh, w, h)
        
        # Visualization
        cv2.drawMarker(frame, (center_u, center_v), (255, 0, 0), cv2.MARKER_CROSS, 30, 2)
        exclude_x = int(w * self.EXCLUDE_TOP_RIGHT_X_RATIO)
        exclude_y = int(h * self.EXCLUDE_TOP_RIGHT_Y_RATIO)
        cv2.rectangle(frame, (exclude_x - 160, 0), (w - 120, exclude_y), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "Excluded (gripper)",
            (exclude_x + 5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        if target is not None:
            obj_u, obj_v = target["center"]
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

            err_u = obj_u - center_u + 20
            # Camera-to-tool alignment offsets
            err_v = obj_v - center_v + 100

            if abs(err_u) < self.THRESHOLD and abs(err_v) < self.THRESHOLD:
                self.is_picking = True
                self.execute_pick_and_place(target["grasp_angle"])
            else:
                self.servoing_move(err_u, err_v)
        else:
            self.rtde_c.speedStop()

        cv2.imshow("Tracking", frame)
        cv2.imshow("Contours", thresh)
        cv2.waitKey(1)

    def servoing_move(self, err_u, err_v):
        # Coordinate mapping: vel_x = speed_u, vel_y = -speed_v
        speed_u = err_u * self.GAIN
        speed_v = err_v * self.GAIN
        
        # speedL(vector, acceleration, time)
        self.rtde_c.speedL([speed_u, -speed_v, 0.0, 0.0, 0.0, 0.0], 0.3, 0.1)

    def find_target_contour(self, thresh, frame_w, frame_h):
        """Return largest valid contour and grasp angle from threshold image."""
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_target = None
        best_area = 0.0

        exclude_x = int(frame_w * self.EXCLUDE_TOP_RIGHT_X_RATIO)
        exclude_y = int(frame_h * self.EXCLUDE_TOP_RIGHT_Y_RATIO)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_CONTOUR_AREA:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            if rw < self.MIN_SIDE_PIXELS or rh < self.MIN_SIDE_PIXELS:
                continue

            # Ignore top-right region (visible gripper).
            if cx >= exclude_x and cy <= exclude_y:
                continue

            if rw < rh:
                # Use orientation that aligns gripper closing with the short side.
                grasp_angle = angle
            else:
                grasp_angle = angle + 90.0

            # Keep angle bounded for stable wrist commands.
            while grasp_angle > 90.0:
                grasp_angle -= 180.0
            while grasp_angle < -90.0:
                grasp_angle += 180.0

            box = cv2.boxPoints(rect).astype(int)
            cx_i, cy_i = int(cx), int(cy)

            if area > best_area:
                best_area = area
                best_target = {
                    "contour": contour,
                    "area": area,
                    "center": (cx_i, cy_i),
                    "box": box,
                    "grasp_angle": grasp_angle,
                } 
        self.rtde_c.moveL(search_pose, 0.15, 0.30)

    def execute_pick_and_place(self, grasp_angle_deg):
        self.get_logger().info("Target centered! Starting pick-and-place sequence...")
        try:
            # 1. Stop visual servoing before precise Cartesian moves.
            self.rtde_c.speedStop()

            # 2. Capture current pose/orientation and descend to pick height.
            curr_pose = self.rtde_r.getActualTCPPose()
            wrist_delta = self.WRIST3_SIGN * (grasp_angle_deg + self.WRIST3_OFFSET_DEG)
            # Always apply object angle on top of fixed search orientation to avoid drift.
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

            # 3. Grasp object.
            self.get_logger().info("Closing gripper...")
            close_ok = self.gripper.close(force=10, speed=200)
            if not close_ok:
                status = self.gripper.get_status()
                if self.gripper.contact_detected_while_closing(status):
                    self.get_logger().warn(
                        "Gripper close timed out, but contact was detected. Continuing to place."
                    )
                else:
                    self.get_logger().error("Gripper close failed with no contact. Aborting place motion.")
                    self.move_to_search_pose(curr_pose)
                    return

            # 4. Lift object to safe transport height.
            lift_pose = list(pick_pose)
            lift_pose[2] = self.SAFE_Z
            self.get_logger().info(f"Lifting to safe Z: {self.SAFE_Z}")
            self.rtde_c.moveL(lift_pose, 0.10, 0.20)

            # 5. Move above the defined place position (keep same orientation).
            place_approach_pose = list(lift_pose)
            place_approach_pose[0] = self.PLACE_X
            place_approach_pose[1] = self.PLACE_Y
            place_approach_pose[2] = self.PLACE_APPROACH_Z
            self.get_logger().info(
                f"Moving to place approach pose: x={self.PLACE_X:.3f}, y={self.PLACE_Y:.3f}, z={self.PLACE_APPROACH_Z:.3f}"
            )
            self.rtde_c.moveL(place_approach_pose, 0.15, 0.30)

            # 6. Descend to place height and release.
            place_pose = list(place_approach_pose)
            place_pose[2] = self.PLACE_Z
            self.get_logger().info(f"Descending to place Z: {self.PLACE_Z}")
            self.rtde_c.moveL(place_pose, 0.08, 0.20)

            self.get_logger().info("Opening gripper to place object...")
            if not self.gripper.open(speed=200):
                self.get_logger().error("Gripper open failed while placing.")

            # 7. Retract back to approach height.
            self.get_logger().info("Retracting after place...")
            self.rtde_c.moveL(place_approach_pose, 0.10, 0.25)

            # 8. Return to search pose for the next object.
            self.move_to_search_pose(place_approach_pose)
            self.get_logger().info("Pick-and-place complete.")

        except Exception as e:
            self.get_logger().error(f"Pick-and-place failed: {e}")
        finally:
            # Re-enable tracking for continuous multi-object operation.
            self.is_picking = False
            if not self.CONTINUOUS_MODE:
                self.get_logger().info("Continuous mode disabled. Tracking paused after place.")
                self.is_picking = True

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

if __name__ == '__main__':
    main()
