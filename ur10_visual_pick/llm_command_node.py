import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from urllib import error, request

import rclpy
from rclpy.node import Node


class UR10LLMCommandNode(Node):
    def __init__(self):
        super().__init__("ur10_llm_command")

        self.declare_parameter("output_json_path", "/tmp/ur10_voice_command.json")
        self.declare_parameter("ollama_url", "http://localhost:11434/api/generate")
        self.declare_parameter("ollama_model", "mistral")

        self.declare_parameter("audio_device", "plughw:2,0")
        self.declare_parameter("audio_duration_sec", 4)
        self.declare_parameter("audio_sample_rate", 16000)
        self.declare_parameter("whisper_cli_path", "/home/user/whisper.cpp/build/bin/whisper-cli")
        self.declare_parameter("whisper_model_path", "/home/user/whisper.cpp/models/ggml-base.en.bin")

        self.output_json_path = str(self.get_parameter("output_json_path").value)
        self.ollama_url = str(self.get_parameter("ollama_url").value)
        self.ollama_model = str(self.get_parameter("ollama_model").value)

        self.audio_device = str(self.get_parameter("audio_device").value)
        self.audio_duration_sec = int(self.get_parameter("audio_duration_sec").value)
        self.audio_sample_rate = int(self.get_parameter("audio_sample_rate").value)
        self.whisper_cli_path = str(self.get_parameter("whisper_cli_path").value)
        self.whisper_model_path = str(self.get_parameter("whisper_model_path").value)

        self.color_names = ("blue", "red", "green", "yellow")

        self.get_logger().info("UR10 LLM command node ready.")
        self.get_logger().info("Type natural language command, or '/speak' for microphone input.")
        self.get_logger().info("Other commands: /help, /quit")

    def run_cli(self):
        while rclpy.ok():
            try:
                raw = input("llm-cmd> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not raw:
                continue

            if raw in {"/quit", "/exit"}:
                break

            if raw == "/help":
                self.print_help()
                continue

            if raw == "/speak":
                spoken = self.capture_speech_once()
                if not spoken:
                    self.get_logger().warn("No speech command captured.")
                    continue
                self.get_logger().info(f"Speech text: {spoken}")
                command = self.parse_natural_command(spoken, source="speak")
                if command is None:
                    self.get_logger().warn("Could not parse speech command.")
                    continue
                self.write_command(command)
                continue

            command = self.parse_natural_command(raw, source="text")
            if command is None:
                self.get_logger().warn("Could not parse command.")
                continue
            self.write_command(command)

    def print_help(self):
        print("Examples:")
        print("  pick the blue box")
        print("  track red object")
        print("  pause tracking")
        print("  resume")
        print("  clear selection")
        print("Special:")
        print("  /speak  -> one-shot mic capture + LLM parse")
        print("  /quit   -> exit")

    def capture_speech_once(self):
        if not os.path.exists(self.whisper_cli_path):
            self.get_logger().error(f"whisper-cli not found: {self.whisper_cli_path}")
            return None
        if not os.path.exists(self.whisper_model_path):
            self.get_logger().error(f"Whisper model not found: {self.whisper_model_path}")
            return None

        with tempfile.TemporaryDirectory(prefix="ur10_llm_") as tmp_dir:
            wav_path = os.path.join(tmp_dir, "chunk.wav")

            self.get_logger().info(
                f"Recording {self.audio_duration_sec}s from {self.audio_device} ..."
            )
            rec = subprocess.run(
                [
                    "arecord",
                    "-D",
                    self.audio_device,
                    "-d",
                    str(self.audio_duration_sec),
                    "-f",
                    "S16_LE",
                    "-r",
                    str(self.audio_sample_rate),
                    "-c",
                    "1",
                    wav_path,
                ],
                capture_output=True,
                text=True,
            )
            if rec.returncode != 0:
                self.get_logger().error(f"arecord failed: {rec.stderr.strip()}")
                return None

            self.get_logger().info("Transcribing speech ...")
            res = subprocess.run(
                [
                    self.whisper_cli_path,
                    "-m",
                    self.whisper_model_path,
                    "-f",
                    wav_path,
                    "-l",
                    "en",
                    "--no-timestamps",
                ],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                self.get_logger().error(f"whisper-cli failed: {res.stderr.strip()}")
                return None

            text = self.extract_transcript_text(res.stdout)
            return text.lower() if text else None

    def extract_transcript_text(self, stdout_text):
        lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
        if not lines:
            return None

        candidates = []
        for line in lines:
            if re.search(r"[A-Za-z]{2,}", line):
                candidates.append(line)

        if not candidates:
            return None

        return max(candidates, key=len).strip()

    def parse_natural_command(self, text, source):
        text_clean = text.strip().lower()
        heuristic = self.heuristic_parse(text_clean)
        if heuristic is not None:
            heuristic["source"] = source
            heuristic["raw_text"] = text
            return heuristic

        parsed = self.parse_with_ollama(text_clean)
        if parsed is None:
            return None
        parsed["source"] = source
        parsed["raw_text"] = text
        return parsed

    def heuristic_parse(self, text):
        if any(word in text for word in ("pause", "stop", "hold")):
            return {"action": "pause", "object": "", "location": "", "Acknowledgement": "Pausing tracking."}
        if any(word in text for word in ("resume", "continue", "start")):
            return {"action": "resume", "object": "", "location": "", "Acknowledgement": "Resuming tracking."}
        if any(word in text for word in ("clear", "reset")):
            return {"action": "clear", "object": "", "location": "", "Acknowledgement": "Clearing selection."}

        target_color = None
        for color in self.color_names:
            if color in text:
                target_color = color
                break

        if target_color and any(word in text for word in ("pick", "select", "track", "target", "follow")):
            return {
                "action": "select",
                "object": target_color,
                "location": "",
                "Acknowledgement": f"Tracking {target_color} object.",
            }
        if target_color:
            return {
                "action": "select",
                "object": target_color,
                "location": "",
                "Acknowledgement": f"Tracking {target_color} object.",
            }

        return None

    def parse_with_ollama(self, text):
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "object": {"type": "string"},
                "location": {"type": "string"},
                "Acknowledgement": {"type": "string"},
            },
            "required": ["action", "object", "location", "Acknowledgement"],
        }
        prompt = (
            "Convert this command into a robot-control JSON object.\n"
            "Allowed actions: select, pick, track, target, pause, resume, clear, reset, stop, start, continue, hold.\n"
            "Prefer color words in object when present (blue, red, green, yellow).\n"
            f'Command: "{text}"'
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
        }

        try:
            req = request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=20) as resp:
                body = resp.read().decode("utf-8")
            decoded = json.loads(body)
        except error.URLError as exc:
            self.get_logger().error(f"Ollama request failed: {exc}")
            return None
        except Exception as exc:
            self.get_logger().error(f"Ollama response error: {exc}")
            return None

        raw_out = str(decoded.get("response", "")).strip()
        parsed = self.try_parse_json_object(raw_out)
        if parsed is None:
            self.get_logger().warn(f"LLM output was not valid JSON: {raw_out}")
            return None

        action = str(parsed.get("action", "")).strip().lower()
        obj = str(parsed.get("object", "")).strip().lower()
        location = str(parsed.get("location", "")).strip()
        ack = str(parsed.get("Acknowledgement", "")).strip()

        action = self.normalize_action(action)
        if not action:
            return None

        for color in self.color_names:
            if color in obj:
                obj = color
                break

        return {
            "action": action,
            "object": obj,
            "location": location,
            "Acknowledgement": ack,
        }

    def normalize_action(self, action):
        if action in {"pause", "stop", "hold"}:
            return "pause"
        if action in {"resume", "start", "continue"}:
            return "resume"
        if action in {"clear", "reset"}:
            return "clear"
        if action in {"select", "pick", "track", "target"}:
            return "select"
        return action

    def try_parse_json_object(self, text):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
        return None

    def write_command(self, command):
        payload = {
            "action": str(command.get("action", "")).strip().lower(),
            "object": str(command.get("object", "")).strip().lower(),
            "location": str(command.get("location", "")).strip(),
            "Acknowledgement": str(command.get("Acknowledgement", "")).strip(),
            "source": str(command.get("source", "")),
            "raw_text": str(command.get("raw_text", "")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        out_dir = os.path.dirname(self.output_json_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=out_dir,
                prefix=".ur10_cmd_",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                json.dump(payload, tmp_file, ensure_ascii=True, indent=2)
                tmp_file.write("\n")
                tmp_path = tmp_file.name
            os.replace(tmp_path, self.output_json_path)
            self.get_logger().info(f"Command written: {payload}")
        except Exception as exc:
            self.get_logger().error(f"Failed writing command JSON: {exc}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def main(args=None):
    rclpy.init(args=args)
    node = UR10LLMCommandNode()
    try:
        node.run_cli()
    finally:
        node.destroy_node()
        rclpy.shutdown()

