#!/usr/bin/env python3
"""
Spin the UR3e end-effector (wrist_3) via URScript over TCP.

No ROS, no driver, no URCap required.

Usage:
  python3 scripts/spin_ee.py               # diagnostics only
  python3 scripts/spin_ee.py --test-popup  # sends a popup to the pendant to verify URScript works
  python3 scripts/spin_ee.py --execute     # send the motion script
"""
import socket
import struct
import sys
import time
import argparse

ROBOT_IP       = "192.168.0.5"
PRIMARY_PORT   = 30001
DASHBOARD_PORT = 29999

HOME_J = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
SPIN_J = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 3.1416]

APPROACH_VEL = 0.15
APPROACH_ACC = 0.3
SPIN_VEL     = 0.4
SPIN_ACC     = 0.5

MOTION_SCRIPT = f"""\
def spin_ee():
  textmsg("spin_ee: moving to home position")
  movej({HOME_J}, a={APPROACH_ACC}, v={APPROACH_VEL})
  textmsg("spin_ee: rotating wrist_3 +180 deg")
  movej({SPIN_J}, a={SPIN_ACC}, v={SPIN_VEL})
  textmsg("spin_ee: rotating wrist_3 back to 0")
  movej({HOME_J}, a={SPIN_ACC}, v={SPIN_VEL})
  textmsg("spin_ee: done")
end
spin_ee()
"""

POPUP_SCRIPT = 'popup("URScript from PC works! Now run --execute to move.", title="Test OK", blocking=False)\n'


def dashboard_query(ip: str, commands: list[str], timeout: float = 5.0) -> dict[str, str]:
    results: dict[str, str] = {}
    try:
        s = socket.create_connection((ip, DASHBOARD_PORT), timeout=timeout)
        time.sleep(0.15)
        welcome = s.recv(4096).decode("utf-8", errors="replace").strip()
        results["__welcome__"] = welcome
        for cmd in commands:
            s.sendall((cmd + "\n").encode())
            time.sleep(0.2)
            resp = s.recv(4096).decode("utf-8", errors="replace").strip()
            results[cmd] = resp
        s.close()
    except Exception as e:
        results["__error__"] = str(e)
    return results


def read_primary_port_info(ip: str, timeout: float = 4.0) -> dict:
    """
    Connect to port 30001, collect ~1 second of packets, and extract:
    - Robot mode (from type-16 state message, sub-package type 0)
    - Any robot message text (from type-20 message packets)
    Packet format: [4B big-endian length][1B type][payload...]
    """
    MODE_MAP = {
        0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING",
        3: "POWER_OFF",    4: "POWER_ON",       5: "IDLE",
        6: "BACKDRIVE",    7: "RUNNING",         8: "UPDATING_FIRMWARE",
    }
    result = {"mode": "UNKNOWN", "messages": []}
    try:
        s = socket.create_connection((ip, PRIMARY_PORT), timeout=timeout)
        s.settimeout(1.0)
        raw = b""
        deadline = time.time() + 1.5  # read for 1.5 s to collect multiple packets
        while time.time() < deadline:
            try:
                chunk = s.recv(4096)
                if not chunk:
                    break
                raw += chunk
            except socket.timeout:
                break
        s.close()

        # Parse packet stream
        pos = 0
        while pos + 5 <= len(raw):
            pkt_len  = struct.unpack_from(">I", raw, pos)[0]
            pkt_type = struct.unpack_from("B",  raw, pos + 4)[0]
            if pkt_len < 5 or pos + pkt_len > len(raw):
                break

            if pkt_type == 16:  # ROBOT_STATE — contains sub-packages
                sub_pos = pos + 5
                while sub_pos + 5 <= pos + pkt_len:
                    sub_len  = struct.unpack_from(">I", raw, sub_pos)[0]
                    sub_type = struct.unpack_from("B",  raw, sub_pos + 4)[0]
                    # Sub-pkg type 0 = Robot Mode Data
                    # Layout: [4B size][1B type][8B timestamp][bool x7][uint8 robotMode]
                    # robotMode is at offset 20 from sub-package start
                    if sub_type == 0 and sub_pos + 21 <= pos + pkt_len:
                        real_connected = struct.unpack_from("B", raw, sub_pos + 13)[0]
                        robot_powered  = struct.unpack_from("B", raw, sub_pos + 15)[0]
                        prog_running   = struct.unpack_from("B", raw, sub_pos + 18)[0]
                        mode_byte      = struct.unpack_from("B", raw, sub_pos + 20)[0]
                        result["mode"]           = MODE_MAP.get(mode_byte, f"UNKNOWN({mode_byte})")
                        result["real_connected"] = bool(real_connected)
                        result["robot_powered"]  = bool(robot_powered)
                        result["prog_running"]   = bool(prog_running)
                    if sub_len < 5:
                        break
                    sub_pos += sub_len

            elif pkt_type == 20:  # ROBOT_MESSAGE — may contain error text
                # Layout after header: 8B timestamp, 1B source, 1B msg_type, then text
                if pkt_len > 15:
                    text_start = pos + 5 + 10  # skip 8B ts + 1B src + 1B msg_type
                    text_bytes = raw[text_start: pos + pkt_len]
                    text = text_bytes.decode("utf-8", errors="replace").strip("\x00").strip()
                    if text:
                        result["messages"].append(text)

            pos += pkt_len
    except Exception as e:
        result["error"] = str(e)
    return result


def diagnose(ip: str) -> bool:
    print(f"\n--- Robot diagnostics ({ip}) ---")

    # Dashboard Server queries
    info = dashboard_query(ip, [
        "robotmode",
        "safetystatus",
        "isInRemoteControl",
        "programstate",
        "get loaded program",
    ])

    if "__error__" in info:
        print(f"  Dashboard Server ({DASHBOARD_PORT}): unreachable — {info['__error__']}")
    else:
        print(f"  Robot mode        : {info.get('robotmode', '?')}")
        print(f"  Safety status     : {info.get('safetystatus', '?')}")
        print(f"  Remote control    : {info.get('isInRemoteControl', '?')}")
        print(f"  Program state     : {info.get('programstate', '?')}")
        print(f"  Loaded program    : {info.get('get loaded program', '?')}")

    # Read mode and messages from binary primary interface
    primary_info = read_primary_port_info(ip)
    print(f"  Primary port mode : {primary_info.get('mode', 'UNKNOWN')}")
    print(f"  Real HW connected : {primary_info.get('real_connected', '?')}")
    print(f"  Robot powered     : {primary_info.get('robot_powered', '?')}")
    print(f"  Program running   : {primary_info.get('prog_running', '?')}")
    for msg in primary_info.get("messages", []):
        print(f"  Primary port msg  : {msg}")
    print()

    ready = True

    safety_val   = info.get("safetystatus", "").lower()
    mode_val     = info.get("robotmode",    "").lower()
    primary_mode = primary_info.get("mode", "").upper()

    if any(x in safety_val for x in ["estop", "protective", "fault", "violation", "reduced"]):
        print(f"  PROBLEM: Safety issue — {info.get('safetystatus')}")
        print("    Fix: release E-stop, then acknowledge on pendant")
        ready = False

    if primary_info.get("real_connected") is False:
        print("  PROBLEM: Robot reports isRealRobotConnected = False")
        print("    This means URControl is running in simulation/disconnected mode.")
        print("    The physical robot will NOT move.")
        print("    Fix: on the teach pendant check the initialization panel —")
        print("         the robot hardware (servo drives) may not be powered on.")
        print("         Press ON → START on the pendant's initialization screen.")
        ready = False

    if "power_off" in primary_mode or "power_off" in mode_val:
        print("  PROBLEM: Robot is POWERED OFF")
        print("    Fix: press ON → START on the pendant to initialize")
        ready = False

    if primary_mode in ("IDLE", "POWER_ON"):
        print(f"  PROBLEM: Robot is in {primary_mode} — motors/brakes not engaged")
        print("    Fix: on the pendant press START / Initialize to engage the motors")
        ready = False

    playing = info.get("programstate", "")
    if "playing" in playing.lower() or "paused" in playing.lower():
        print(f"  PROBLEM: Program already running ({playing})")
        print("    Fix: press STOP on the pendant")
        ready = False

    if ready:
        print("  Robot appears ready.")
        print()
        print("  *** IMPORTANT — PolyScope 5.5 note ***")
        print("  'isInRemoteControl' is not supported by this firmware.")
        print("  For URScript to execute from a remote PC on PolyScope 5.5,")
        print("  the pendant must be in AUTOMATIC mode (not Teach/Manual).")
        print()
        print("  On the teach pendant:")
        print("    1. Look at the top bar — is there a 'Manual' / person icon?")
        print("    2. If so, tap it and switch to 'Automatic' (run) mode.")
        print("    3. Alternatively: the physical key-switch on the teach pendant")
        print("       must be in the AUTO position (not MANUAL).")
        print()
        print("  Run --test-popup first to verify URScript is being received.")

    return ready


def send_script(ip: str, script: str, label: str) -> None:
    print(f"\nConnecting to {ip}:{PRIMARY_PORT} ...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect((ip, PRIMARY_PORT))

    # Drain initial binary state dump from the controller
    time.sleep(0.3)
    s.setblocking(False)
    try:
        s.recv(65536)
    except BlockingIOError:
        pass
    s.setblocking(True)

    s.sendall((script + "\n").encode("utf-8"))
    print(f"Sent: {label}")

    # Listen for text fragments in the binary stream (error messages, textmsg output)
    print("Listening for robot feedback (Ctrl+C to stop)...\n")
    s.settimeout(1.0)
    try:
        while True:
            try:
                data = s.recv(4096)
                if not data:
                    print("  [connection closed by robot]")
                    break
                # Extract printable text fragments from binary packets
                text = "".join(
                    c for c in data.decode("utf-8", errors="replace")
                    if c.isprintable() or c in "\n\r"
                ).strip()
                if text and len(text) > 3:
                    # Filter out pure-garbage binary decodes
                    printable_ratio = sum(1 for c in text if c.isascii()) / max(len(text), 1)
                    if printable_ratio > 0.6:
                        print(f"  [robot] {text}")
            except socket.timeout:
                pass
    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        s.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Spin UR3e wrist_3 via URScript")
    parser.add_argument("--execute",    action="store_true", help="Send motion script")
    parser.add_argument("--test-popup", action="store_true", help="Send a popup to verify URScript works")
    parser.add_argument("--ip", default=ROBOT_IP)
    args = parser.parse_args()

    if not args.execute and not args.test_popup:
        print("=== spin_ee ===")
        print(f"Home : {HOME_J}")
        print(f"Spin : {SPIN_J}  (wrist_3 +180°)")
        print()
        diagnose(args.ip)
        print("\nOptions:")
        print("  --test-popup   send a popup to the pendant (safest test)")
        print("  --execute      send the motion script")
        return

    if args.test_popup:
        print("Sending popup test script to robot pendant...")
        print("Watch the teach pendant — a dialog should appear.")
        send_script(args.ip, POPUP_SCRIPT, "popup test")
        return

    if args.execute:
        print("=== spin_ee EXECUTE ===")
        print(f"Home : {HOME_J}")
        print(f"Spin : {SPIN_J}  (wrist_3 +180°)")
        print()
        ready = diagnose(args.ip)
        if not ready:
            print("\nFix the issues above before executing.")
            sys.exit(1)
        print("\nSending motion script. Keep an operator at the E-stop.")
        send_script(args.ip, MOTION_SCRIPT, "spin wrist_3 ±180°")


if __name__ == "__main__":
    main()
