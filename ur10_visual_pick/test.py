import argparse
import time

from ur10_visual_pick.robotiq_3f_gripper import RobotiqGripper3F


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual test utility for the Robotiq 3F gripper."
    )
    parser.add_argument(
        "action",
        choices=["activate", "deactivate", "status", "open", "close", "move", "cycle"],
        nargs="?",
        default="status",
        help="Gripper action to execute.",
    )
    parser.add_argument(
        "--ip",
        default="192.168.1.105",
        help="Gripper Modbus TCP IP address.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=502,
        help="Gripper Modbus TCP port.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=200,
        help="Motion speed in the range 0-255.",
    )
    parser.add_argument(
        "--force",
        type=int,
        default=100,
        help="Closing force in the range 0-255.",
    )
    parser.add_argument(
        "--position",
        type=int,
        default=0,
        help="Target position for the move action in the range 0-255.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="Pause between actions in seconds.",
    )
    return parser


def run_test(
    action: str = "status",
    ip: str = "192.168.1.105",
    port: int = 502,
    speed: int = 200,
    force: int = 100,
    position: int = 0,
    pause: float = 2.0,
) -> int:
    print(f"Connecting to gripper at {ip}:{port} ...")
    gripper = RobotiqGripper3F(ip=ip, port=port)

    try:
        if action == "activate":
            print("Activating gripper ...")
            if not gripper.activate():
                print("Activation failed. Check gripper power, Ethernet link, and IP settings.")
                return 1
            print(f"Activation complete. Status: {gripper.get_status()}")

        elif action == "deactivate":
            print("Deactivating gripper ...")
            gripper.deactivate()
            print(f"Deactivation command sent. Status: {gripper.get_status()}")

        elif action == "status":
            is_active = gripper.sync()
            print(f"Gripper active: {is_active}")
            print(f"Status: {gripper.get_status()}")

        else:
            if not gripper.sync():
                print("Gripper is not active. Run the activate command first.")
                print(f"Current status: {gripper.get_status()}")
                return 1

            print(f"Using active gripper. Status: {gripper.get_status()}")

        if action == "open":
            print("Opening gripper ...")
            if not gripper.open(speed=speed):
                return 1
            print(f"Open status: {gripper.get_status()}")

        elif action == "close":
            print(f"Closing gripper with force={force}, speed={speed} ...")
            if not gripper.close(force=force, speed=speed):
                return 1
            print(f"Close status: {gripper.get_status()}")

        elif action == "move":
            print(f"Moving gripper to position={position}, force={force}, speed={speed} ...")
            if not gripper.move(position=position, speed=speed, force=force):
                return 1
            print(f"Move status: {gripper.get_status()}")

        else:
            if action == "cycle":
                print("Cycling gripper: open -> close -> open ...")
                if not gripper.open(speed=speed):
                    return 1
                print(f"After first open: {gripper.get_status()}")
                time.sleep(pause)
                if not gripper.close(force=force, speed=speed):
                    return 1
                print(f"After close: {gripper.get_status()}")
                time.sleep(pause)
                if not gripper.open(speed=speed):
                    return 1
                print(f"After second open: {gripper.get_status()}")

        print("Requested gripper action completed.")
        return 0

    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130

    except Exception as exc:
        print(f"Unexpected error while testing gripper: {exc}")
        return 1

    finally:
        print("Shutting down gripper connection ...")
        gripper.shutdown()


def main() -> int:
    args = _build_parser().parse_args()
    return run_test(
        action=args.action,
        ip=args.ip,
        port=args.port,
        speed=args.speed,
        force=args.force,
        position=args.position,
        pause=args.pause,
    )


if __name__ == "__main__":
    raise SystemExit(main())
