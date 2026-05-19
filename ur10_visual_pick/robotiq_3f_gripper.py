#!/usr/bin/env python3
"""
robotiq_3f_gripper.py
----------------------
Python TCP (Modbus TCP) controller for the Robotiq 3-Finger Adaptive Gripper.

Connects to the gripper's Ethernet adapter at a fixed IP/port.
Activates the gripper on startup and deactivates on shutdown —
the gripper is only powered/active while this script is running.

Usage (from grasp_executor_node in real_hardware mode):
    gripper = RobotiqGripper3F(ip="192.168.1.105", port=502)
    gripper.activate()      # called once at startup
    gripper.open()          # fully open fingers
    gripper.close(force=50) # close with medium force
    gripper.shutdown()      # deactivate on node exit

The official Modbus TCP manual uses register 0x0000 as the first robot
output/input register. For motion commands the bytes are packed as:
  Register 0: ACTION_REQUEST, GRIPPER_OPTIONS
  Register 1: GRIPPER_OPTIONS_2, POSITION_REQUEST
  Register 2: SPEED, FORCE

Example from the manual for a full close at full speed/force:
  0900 00FF FFFF
"""

import socket
import struct
import time


# ---------------------------------------------------------------------------
# Modbus TCP helpers
# ---------------------------------------------------------------------------

_TRANSACTION_ID = 0x0001
_PROTOCOL_ID    = 0x0000
_UNIT_ID        = 0x02

_REG_OUTPUT     = 0x0000        # 4.8.1 Robot Output / Gripper Input First Register
_REG_INPUT      = 0x0000        # 4.8.1 Robot Input / Gripper Output First Register

_TIMEOUT_SEC    = 3.0           # Socket timeout
_ACT_TIMEOUT    = 20.0          # Max wait for activation
_MOVE_TIMEOUT   = 10.0          # Max wait for open/close completion
_IO_RETRIES     = 3             # Retry transient TCP failures
_IO_RETRY_DELAY = 0.10          # Delay between retries


def _build_write_frame(start_addr: int, register_values: list[int]) -> bytes:
    """Build a Modbus TCP 'Write Multiple Registers' (FC 0x10) request frame."""
    n        = len(register_values)
    byte_cnt = n * 2
    # PDU: func(1) + addr(2) + count(2) + byte_count(1) + data(n*2)
    pdu = struct.pack(">BHHB", 0x10, start_addr, n, byte_cnt)
    for v in register_values:
        pdu += struct.pack(">H", v & 0xFFFF)
    # MBAP header: tx_id(2) + proto_id(2) + length(2) + unit_id(1)
    length = 1 + len(pdu)
    header = struct.pack(">HHHB", _TRANSACTION_ID, _PROTOCOL_ID, length, _UNIT_ID)
    return header + pdu


def _build_read_frame(start_addr: int, n_registers: int) -> bytes:
    """Build a Modbus TCP 'Read Input Registers' (FC 0x04) request frame."""
    pdu    = struct.pack(">BHH", 0x04, start_addr, n_registers)
    length = 1 + len(pdu)
    header = struct.pack(">HHHB", _TRANSACTION_ID, _PROTOCOL_ID, length, _UNIT_ID)
    return header + pdu


# ---------------------------------------------------------------------------
# Gripper class
# ---------------------------------------------------------------------------

class RobotiqGripper3F:
    """
    Robotiq 3-Finger Adaptive Gripper — Ethernet (Modbus TCP) controller.

    The gripper is activated on `activate()` and deactivated on `shutdown()`.
    Between those calls it is ready to receive open/close commands.
    """

    # Position byte: 0 = fully open, 255 = fully closed
    POS_OPEN   = 0
    POS_CLOSE  = 255
    SPEED_MAX  = 255
    FORCE_SOFT = 50     # for open (gentle)
    FORCE_FIRM = 100    # for close

    def __init__(self, ip: str = "192.168.1.105", port: int = 502):
        self._ip   = ip
        self._port = port
        self._sock: socket.socket | None = None
        self._active = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self):
        if self._sock is not None:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(_TIMEOUT_SEC)
        self._sock.connect((self._ip, self._port))

    def _disconnect(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _send(self, frame: bytes) -> bytes:
        """Send a Modbus frame and return the response bytes."""
        last_exc = None
        for attempt in range(_IO_RETRIES):
            try:
                self._connect()
                self._sock.sendall(frame)
                return self._sock.recv(256)
            except Exception as exc:
                last_exc = exc
                self._disconnect()
                if attempt < _IO_RETRIES - 1:
                    time.sleep(_IO_RETRY_DELAY)
        raise RuntimeError(f"Modbus TCP communication failed after {_IO_RETRIES} attempts: {last_exc}")

    def _write_registers(self, start: int, values: list[int]):
        frame = _build_write_frame(start, values)
        self._send(frame)

    def _read_input_registers(self, start: int, count: int) -> list[int]:
        """Read input registers; return list of 16-bit values."""
        frame = _build_read_frame(start, count)
        resp  = self._send(frame)
        # Response: MBAP(7) + func(1) + byte_count(1) + data
        if len(resp) < 9:
            return []
        byte_count = resp[8]
        data       = resp[9: 9 + byte_count]
        values     = []
        for i in range(0, len(data), 2):
            values.append(struct.unpack(">H", data[i:i+2])[0])
        return values

    def get_status(self) -> dict | None:
        """
        Read and decode the main input registers.

        The decoded fields follow the naming in the Robotiq manual.
        """
        vals = self._read_input_registers(_REG_INPUT, 8)
        if len(vals) < 3:
            return None

        reg0_hi = (vals[0] >> 8) & 0xFF
        reg0_lo = vals[0] & 0xFF
        reg1_hi = (vals[1] >> 8) & 0xFF
        reg1_lo = vals[1] & 0xFF
        reg2_hi = (vals[2] >> 8) & 0xFF
        reg2_lo = vals[2] & 0xFF

        return {
            "raw": vals,
            "gACT": reg0_hi & 0x01,
            "gMOD": (reg0_hi >> 1) & 0x03,
            "gGTO": (reg0_hi >> 3) & 0x01,
            "gIMC": (reg0_hi >> 4) & 0x03,
            "gSTA": (reg0_hi >> 6) & 0x03,
            "object_status": reg0_lo,
            "gDTA": reg0_lo & 0x03,
            "gDTB": (reg0_lo >> 2) & 0x03,
            "gDTC": (reg0_lo >> 4) & 0x03,
            "gDTS": (reg0_lo >> 6) & 0x03,
            "fault_status": reg1_hi,
            "position_request_echo": reg1_lo,
            "finger_a_position": reg2_hi,
            "finger_a_current": reg2_lo,
        }

    def contact_detected_while_closing(self, status: dict | None = None) -> bool:
        """
        Return True when any finger reports a closing contact condition.

        For the 3F object status bits, value 2 indicates contact while closing.
        """
        if status is None:
            status = self.get_status()
        if not status:
            return False
        return (
            status["gDTA"] == 2
            or status["gDTB"] == 2
            or status["gDTC"] == 2
            or status["gSTA"] == 2
        )

    def _wait_for_activation(self, timeout: float = _ACT_TIMEOUT) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.get_status()
            if status and status["gACT"] == 1 and status["gIMC"] == 3:
                self._active = True
                return True
            time.sleep(0.1)
        return False

    def _wait_for_motion(
        self,
        requested_position: int,
        allow_contact_stop: bool,
        timeout: float = _MOVE_TIMEOUT,
    ) -> dict | None:
        deadline = time.monotonic() + timeout
        last_status = None
        while time.monotonic() < deadline:
            status = self.get_status()
            if status:
                last_status = status
                if status["fault_status"] != 0:
                    return status

                # Grasp close can legitimately stop early on object contact.
                if allow_contact_stop and self.contact_detected_while_closing(status):
                    return status

                echo_matches = status["position_request_echo"] == requested_position
                pos_near_requested = abs(status["finger_a_position"] - requested_position) <= 20
                if status["gSTA"] == 3 and (echo_matches or pos_near_requested):
                    return status
            time.sleep(0.05)
        if last_status:
            print(f"[RobotiqGripper3F] Move timed out after {timeout}s. Last status: {last_status}")
        return None

    def _is_already_active(self) -> bool:
        status = self.get_status()
        if status and status["gACT"] == 1 and status["gIMC"] == 3:
            self._active = True
            return True
        return False

    def sync(self) -> bool:
        """
        Refresh the local controller state from the gripper status.

        This does not reset or activate the hardware. It only checks whether
        the gripper is already active.
        """
        return self._is_already_active()

    # ------------------------------------------------------------------
    # Gripper lifecycle
    # ------------------------------------------------------------------

    def activate(self, force_reinitialize: bool = False) -> bool:
        """
        Activate the gripper.

        If the gripper is already active, the controller reuses that state
        instead of forcing another reset unless `force_reinitialize` is True.

        Sequence:
          1. If already active, reuse it
          2. Otherwise reset (clear rACT)
          3. Set rACT = 1
          4. Poll until gIMC == 3 (initialisation complete)

        Returns True on success, False on timeout.
        """
        try:
            self._connect()

            if not force_reinitialize and self._is_already_active():
                return True

            # 1. Reset
            self._write_registers(_REG_OUTPUT, [0x0000, 0x0000, 0x0000])
            time.sleep(0.5)

            # 2. Activate: rACT=1 → byte0 of reg0 = 0x01
            self._write_registers(_REG_OUTPUT, [0x0100, 0x0000, 0x0000])

            # 3. Poll for activation complete (gIMC bits 4-5 of input reg 0 == 3)
            if self._wait_for_activation():
                return True

            print(f"[RobotiqGripper3F] Activation timed out after {_ACT_TIMEOUT}s")
            return False

        except Exception as e:
            print(f"[RobotiqGripper3F] Activation failed: {e}")
            return False

    def deactivate(self):
        """
        Clear rACT and put the gripper back into standby.
        """
        try:
            if self._sock is not None:
                self._write_registers(_REG_OUTPUT, [0x0000, 0x0000, 0x0000])
        except Exception:
            pass
        self._active = False

    def shutdown(self, deactivate: bool = False):
        """
        Close the TCP connection.

        By default this does not clear rACT, so repeated command runs do not
        force a full re-activation cycle. Pass `deactivate=True` only when you
        explicitly want the gripper to return to standby.
        """
        if deactivate:
            self.deactivate()
        self._disconnect()

    # ------------------------------------------------------------------
    # Motion commands
    # ------------------------------------------------------------------

    def _go_to(self, position: int, speed: int, force: int):
        """
        Send a go-to command in basic mode (all 3 fingers move together).

        position : 0 (open) … 255 (closed)
        speed    : 0 (slow) … 255 (fast)
        force    : 0 (light) … 255 (max)
        """
        position = max(0, min(255, position))
        speed    = max(0, min(255, speed))
        force    = max(0, min(255, force))

        if not self._active and not self._is_already_active():
            raise RuntimeError("Gripper must be activated before sending motion commands.")

        # Per the official Modbus TCP mapping:
        #   Register 0 = ACTION_REQUEST, GRIPPER_OPTIONS
        #   Register 1 = GRIPPER_OPTIONS_2, POSITION_REQUEST
        #   Register 2 = SPEED, FORCE
        reg0 = 0x0900
        reg1 = position
        reg2 = (speed << 8) | force

        self._write_registers(_REG_OUTPUT, [reg0, reg1, reg2])

    def move(self, position: int, speed: int = 200, force: int = 100) -> bool:
        """Move gripper to an arbitrary position in the range 0-255."""
        try:
            requested_position = max(0, min(255, position))
            self._go_to(requested_position, speed, force)
            status = self._wait_for_motion(
                requested_position,
                allow_contact_stop=requested_position > self.POS_OPEN,
            )
            if status is None:
                print(f"[RobotiqGripper3F] Move timed out after {_MOVE_TIMEOUT}s")
                return False
            if status["fault_status"] != 0:
                print(f"[RobotiqGripper3F] Move fault: 0x{status['fault_status']:02X}")
                return False
            return True
        except Exception as e:
            print(f"[RobotiqGripper3F] Move failed: {e}")
            return False

    def open(self, speed: int = 200) -> bool:
        """Open gripper fully."""
        return self.move(position=self.POS_OPEN, speed=speed, force=self.FORCE_SOFT)

    def close(self, force: int = None, speed: int = 200) -> bool:
        """Close gripper to grasp position."""
        if force is None:
            force = self.FORCE_FIRM
        return self.move(position=self.POS_CLOSE, speed=speed, force=force)

    def is_active(self) -> bool:
        return self._active

    # Context manager support
    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, *_):
        self.shutdown()
