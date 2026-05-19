# ur10_visual_pick

Python ROS 2 package for vision-driven picking utilities.

This repo is intended to contain higher-level logic around:
- object detection/tracking topics
- pose estimation helpers (if needed)
- coordinating pick sequences (pre-grasp, approach, retreat)

In the current workspace, low-level “align TCP to colored object” planning/execution is implemented in `ur10_tcp_alignment` (C++).

## Build

```bash
cd ~/ur10_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select ur10_visual_pick
source install/setup.bash
```

## Typical Usage

Launch simulation or real hardware via `ur10_bringup`, then call:

```bash
ros2 topic pub --once /pick_command std_msgs/msg/String \"{data: 'pick red box'}\"
```

