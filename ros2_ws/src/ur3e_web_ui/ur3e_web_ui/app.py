from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import random
from threading import Lock
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ur3e_rollout_replay.replay_core import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_ROLLOUT_PATH,
    ReplayDataError,
    SafetyLimits,
)

from . import motion
from .joint_limits import JointLimit, load_ur3e_joint_limits
from .ros_interface import ActionServerUnavailable, IKFailed, IKServiceUnavailable, MotionBusy, RosBridge
from .urdf_provider import UR_DESCRIPTION_SHARE

STATIC_DIR = Path(__file__).parent / "static"
WS_PERIOD_S = 1.0 / 15.0
STATIONARY_VELOCITY_RAD_S = 0.02
MIN_MOTION_SPEED_SCALING = 1e-3
MAX_REPLAY_ACCELERATION_RAD_S2 = 6.0
MAX_APPROACH_MIN_DURATION_S = 60.0
MIN_SEGMENT_DURATION_FLOOR_S = 0.02
MAX_SEGMENT_DURATION_FLOOR_S = 5.0
TCP_TARGET_MIN_DURATION_S = 1.0
TCP_TARGET_MAX_DURATION_S = 30.0
# MoveIt's KDL plugin restarts from uniform-random seeds when the seeded attempt
# fails (always the case near the wrist singularity at the home pose), so a
# single /compute_ik call returns an arbitrary IK branch. When the target is
# close, the seeded attempt wins deterministically; otherwise each call samples
# a quasi-random branch, so collect many samples and keep the branch needing
# the least joint motion (12 samples make missing the near branch ~0.2%).
TCP_IK_ATTEMPTS = 12
TCP_IK_CLOSE_ENOUGH_RAD = 0.35
TCP_IK_SEED_PERTURBATION_RAD = 0.08
TCP_IK_SEEDED_TIMEOUT_S = 0.05
TCP_IK_FALLBACK_TIMEOUT_S = 1.0
TCP_IK_FALLBACK_ATTEMPTS = 3
TCP_EXECUTE_MATCH_TOLERANCE_RAD = 0.05

REPLAY_PRESETS = {
    "safe": SafetyLimits(
        max_joint_velocity=0.25,
        max_joint_acceleration=0.5,
        approach_min_duration=10.0,
        min_segment_duration=0.5,
    ),
    "balanced": SafetyLimits(
        max_joint_velocity=0.5,
        max_joint_acceleration=1.0,
        approach_min_duration=3.0,
        min_segment_duration=0.15,
    ),
    "fast": SafetyLimits(
        max_joint_velocity=1.0,
        max_joint_acceleration=2.0,
        approach_min_duration=1.0,
        min_segment_duration=0.05,
    ),
}


@dataclass(frozen=True)
class Settings:
    rollout_path: str = str(DEFAULT_ROLLOUT_PATH)
    limits: SafetyLimits = SafetyLimits()
    home_positions: tuple[float, ...] = motion.HOME_POSITIONS


class JogRequest(BaseModel):
    joint: str
    direction: int
    step_rad: float | None = None


class ConfirmRequest(BaseModel):
    confirm: bool = False


class ReplaySettingsRequest(BaseModel):
    max_joint_velocity: float | None = None
    max_joint_acceleration: float | None = None
    approach_min_duration: float | None = None
    min_segment_duration: float | None = None


class TcpTargetRequest(BaseModel):
    xyz_m: list[float]
    rpy_rad: list[float]
    duration_s: float | None = motion.TCP_TARGET_MIN_DURATION
    avoid_collisions: bool = True
    confirm: bool = False
    # Joint target shown during validation; execution is rejected if the fresh
    # IK solution no longer matches what the user previewed.
    expected_joints_rad: list[float] | None = None


def create_app(bridge: RosBridge, settings: Settings) -> FastAPI:
    joint_limits = load_ur3e_joint_limits()
    jog_lock = asyncio.Lock()
    limits_lock = Lock()
    current_replay_limits = settings.limits
    finite_velocity_limits = [
        limit.max_velocity for limit in joint_limits.values() if math.isfinite(limit.max_velocity)
    ]
    max_joint_velocity_cap = min(finite_velocity_limits) if finite_velocity_limits else math.pi
    replay_setting_bounds = {
        "max_joint_velocity": {"min": 0.01, "max": max_joint_velocity_cap, "step": 0.01},
        "max_joint_acceleration": {"min": 0.05, "max": MAX_REPLAY_ACCELERATION_RAD_S2, "step": 0.05},
        "approach_min_duration": {"min": 0.0, "max": MAX_APPROACH_MIN_DURATION_S, "step": 0.1},
        "min_segment_duration": {
            "min": MIN_SEGMENT_DURATION_FLOOR_S,
            "max": MAX_SEGMENT_DURATION_FLOOR_S,
            "step": 0.01,
        },
    }
    _validate_replay_limits(current_replay_limits, replay_setting_bounds)

    def get_replay_limits() -> SafetyLimits:
        with limits_lock:
            return current_replay_limits

    def set_replay_limits(body: ReplaySettingsRequest) -> SafetyLimits:
        nonlocal current_replay_limits
        with limits_lock:
            updated = SafetyLimits(
                max_joint_velocity=(
                    current_replay_limits.max_joint_velocity
                    if body.max_joint_velocity is None
                    else body.max_joint_velocity
                ),
                max_joint_acceleration=(
                    current_replay_limits.max_joint_acceleration
                    if body.max_joint_acceleration is None
                    else body.max_joint_acceleration
                ),
                approach_min_duration=(
                    current_replay_limits.approach_min_duration
                    if body.approach_min_duration is None
                    else body.approach_min_duration
                ),
                min_segment_duration=(
                    current_replay_limits.min_segment_duration
                    if body.min_segment_duration is None
                    else body.min_segment_duration
                ),
            )
            try:
                _validate_replay_limits(updated, replay_setting_bounds)
            except ReplayDataError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            current_replay_limits = updated
            return updated

    async def ensure_motion_enabled(snapshot) -> None:
        if snapshot.program_running is False:
            raise HTTPException(
                status_code=409,
                detail="External Control program is stopped; press Play on the teach pendant or Dashboard Play",
            )
        if snapshot.speed_scaling is not None and snapshot.speed_scaling <= MIN_MOTION_SPEED_SCALING:
            raise HTTPException(
                status_code=409,
                detail="robot speed scaling is 0%; press Play on External Control and raise the speed slider",
            )
        if snapshot.controller_active is False:
            # controller_stopper can leave the controller inactive after an
            # External Control restart; reactivating only holds position.
            if not await bridge.activate_trajectory_controller():
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "scaled_joint_trajectory_controller is inactive and could not be "
                        "reactivated; stop and re-Play the External Control program"
                    ),
                )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        bridge.start(asyncio.get_running_loop())
        try:
            yield
        finally:
            bridge.shutdown()

    app = FastAPI(title="UR3e Web UI", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/pkg/ur_description", StaticFiles(directory=UR_DESCRIPTION_SHARE), name="ur_description")

    # ------------------------------------------------------------------ pages

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # ------------------------------------------------------------------ info

    @app.get("/api/health")
    def health() -> dict:
        snapshot = bridge.get_snapshot()
        return {
            "ok": True,
            "joint_states_alive": snapshot.joint_states_alive,
            "action_server_ready": snapshot.action_server_ready,
            "ik_service_ready": snapshot.ik_service_ready,
        }

    @app.get("/api/state")
    def state() -> dict:
        return _state_message(bridge)

    @app.get("/api/urdf")
    def urdf() -> Response:
        try:
            urdf_xml, source = bridge.urdf_cache.get()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"URDF unavailable: {exc}")
        return Response(content=urdf_xml, media_type="application/xml", headers={"X-Urdf-Source": source})

    @app.get("/api/limits")
    def limits() -> dict:
        replay_limits = get_replay_limits()
        return {
            "joints": {
                name: {
                    "min_position": _json_safe(limit.min_position),
                    "max_position": _json_safe(limit.max_position),
                    "max_velocity": _json_safe(limit.max_velocity),
                }
                for name, limit in joint_limits.items()
            },
            "safety": {
                "max_joint_velocity": replay_limits.max_joint_velocity,
                "max_joint_acceleration": replay_limits.max_joint_acceleration,
                "approach_min_duration": replay_limits.approach_min_duration,
                "min_segment_duration": replay_limits.min_segment_duration,
            },
            "jog": {
                "step_default": motion.JOG_STEP_DEFAULT,
                "step_max": motion.JOG_STEP_MAX,
                "velocity": motion.JOG_VELOCITY,
            },
            "home_positions": list(settings.home_positions),
        }

    @app.get("/api/replay_settings")
    def replay_settings() -> dict:
        return _replay_settings_payload(get_replay_limits(), replay_setting_bounds)

    @app.post("/api/replay_settings")
    def update_replay_settings(body: ReplaySettingsRequest) -> dict:
        return _replay_settings_payload(set_replay_limits(body), replay_setting_bounds)

    # ------------------------------------------------------------------ rollout

    @app.get("/api/rollout")
    def rollout_index() -> dict:
        replay_limits = get_replay_limits()
        try:
            metadata, summaries = motion.episode_summaries(settings.rollout_path, replay_limits)
        except ReplayDataError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {
            "path": settings.rollout_path,
            "metadata": {
                "dt_s": metadata.get("dt_s"),
                "joint_names": metadata.get("joint_names"),
                "action_scale": metadata.get("action_scale"),
                "task": metadata.get("task"),
            },
            "episodes": summaries,
        }

    @app.get("/api/rollout/{episode_index}/plan")
    def rollout_plan(episode_index: int, approach: bool = True) -> dict:
        replay_limits = get_replay_limits()
        current = None
        if approach:
            snapshot = bridge.get_snapshot()
            if not snapshot.joint_states_alive:
                raise HTTPException(status_code=409, detail="no live joint state; start the driver or use approach=false")
            current = snapshot.joint_positions
        try:
            plan, _, within = motion.build_episode_plan(
                settings.rollout_path, episode_index, replay_limits, current_positions=current
            )
        except ReplayDataError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return motion.plan_to_dict(
            plan,
            episode_index=episode_index,
            within_limits=within,
            approach_included=current is not None,
            current_positions=list(current) if current is not None else None,
        )

    @app.post("/api/rollout/{episode_index}/execute", status_code=202)
    async def rollout_execute(episode_index: int, body: ConfirmRequest) -> dict:
        if not body.confirm:
            raise HTTPException(status_code=400, detail="confirm must be true")
        replay_limits = get_replay_limits()
        snapshot = bridge.get_snapshot()
        if not snapshot.joint_states_alive:
            raise HTTPException(status_code=409, detail="no live joint state")
        await ensure_motion_enabled(snapshot)
        assert snapshot.joint_velocities is not None
        if max(abs(v) for v in snapshot.joint_velocities) > STATIONARY_VELOCITY_RAD_S:
            raise HTTPException(status_code=409, detail="robot is moving; wait for it to stop")
        try:
            plan, _, within = motion.build_episode_plan(
                settings.rollout_path, episode_index, replay_limits,
                current_positions=snapshot.joint_positions,
            )
        except ReplayDataError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not within:
            raise HTTPException(status_code=400, detail="retimed plan exceeds safety limits")
        record = await _send(plan, "rollout", episode_index)
        return {
            "accepted": record.phase == "active",
            "plan": {
                "points": len(plan.positions),
                "total_duration_s": plan.time_from_start_s[-1],
                "max_velocity_rad_s": plan.retimed_stats.max_velocity_rad_s,
                "max_acceleration_rad_s2": plan.retimed_stats.max_acceleration_rad_s2,
            },
        }

    # ------------------------------------------------------------------ control

    @app.post("/api/jog")
    async def jog(body: JogRequest) -> dict:
        async with jog_lock:
            snapshot = bridge.get_snapshot()
            if not snapshot.joint_states_alive:
                raise HTTPException(status_code=503, detail="no live joint state")
            await ensure_motion_enabled(snapshot)
            base = bridge.active_jog_base() or snapshot.joint_positions
            try:
                plan, target, clamped = motion.build_jog_target(
                    base, body.joint, body.direction,
                    body.step_rad if body.step_rad is not None else motion.JOG_STEP_DEFAULT,
                    joint_limits,
                )
            except ReplayDataError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            record = await _send(plan, "jog")
        return {
            "accepted": record.phase == "active",
            "target_rad": target,
            "duration_s": plan.time_from_start_s[-1],
            "clamped": clamped,
        }

    @app.post("/api/move_home", status_code=202)
    async def move_home(body: ConfirmRequest) -> dict:
        if not body.confirm:
            raise HTTPException(status_code=400, detail="confirm must be true")
        snapshot = bridge.get_snapshot()
        if not snapshot.joint_states_alive:
            raise HTTPException(status_code=503, detail="no live joint state")
        await ensure_motion_enabled(snapshot)
        try:
            plan = motion.build_home_plan(snapshot.joint_positions, get_replay_limits(), settings.home_positions)
        except ReplayDataError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        record = await _send(plan, "home")
        return {"accepted": record.phase == "active", "duration_s": plan.time_from_start_s[-1]}

    @app.post("/api/tcp_target/plan")
    async def tcp_target_plan(body: TcpTargetRequest) -> dict:
        snapshot = bridge.get_snapshot()
        if not snapshot.joint_states_alive:
            raise HTTPException(status_code=503, detail="no live joint state")
        plan, target_joints = await _build_tcp_target_plan(body, snapshot)
        return {
            "target_joints_rad": list(target_joints),
            "duration_s": plan.time_from_start_s[-1],
            "max_joint_delta_rad": motion.max_joint_delta(target_joints, snapshot.joint_positions),
            "plan": motion.plan_to_dict(plan),
        }

    @app.post("/api/tcp_target/execute", status_code=202)
    async def tcp_target_execute(body: TcpTargetRequest) -> dict:
        if not body.confirm:
            raise HTTPException(status_code=400, detail="confirm must be true")
        snapshot = bridge.get_snapshot()
        if not snapshot.joint_states_alive:
            raise HTTPException(status_code=503, detail="no live joint state")
        await ensure_motion_enabled(snapshot)
        assert snapshot.joint_velocities is not None
        if max(abs(v) for v in snapshot.joint_velocities) > STATIONARY_VELOCITY_RAD_S:
            raise HTTPException(status_code=409, detail="robot is moving; wait for it to stop")
        plan, target_joints = await _build_tcp_target_plan(body, snapshot)
        if body.expected_joints_rad is not None:
            expected = _coerce_vector(body.expected_joints_rad, len(DEFAULT_JOINT_NAMES), "expected_joints_rad")
            if motion.max_joint_delta(target_joints, expected) > TCP_EXECUTE_MATCH_TOLERANCE_RAD:
                raise HTTPException(
                    status_code=409,
                    detail="IK solution changed since validation; validate the target again before sending",
                )
        record = await _send(plan, "tcp")
        return {
            "accepted": record.phase == "active",
            "target_joints_rad": list(target_joints),
            "duration_s": plan.time_from_start_s[-1],
        }

    @app.post("/api/cancel")
    async def cancel() -> dict:
        return {"canceled": await bridge.cancel_active()}

    @app.post("/api/dashboard/{command}")
    async def dashboard(command: str) -> dict:
        try:
            success, message = await bridge.call_dashboard(command)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown dashboard command: {command}")
        except ActionServerUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return {"success": success, "message": message}

    async def _send(plan, kind: str, episode: int | None = None):
        try:
            return await bridge.send_plan(plan, kind, episode)
        except MotionBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ActionServerUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    async def _build_tcp_target_plan(body: TcpTargetRequest, snapshot):
        xyz = _coerce_vector(body.xyz_m, 3, "xyz_m")
        rpy = _coerce_vector(body.rpy_rad, 3, "rpy_rad")
        duration = motion.TCP_TARGET_MIN_DURATION if body.duration_s is None else float(body.duration_s)
        if not math.isfinite(duration) or duration < TCP_TARGET_MIN_DURATION_S or duration > TCP_TARGET_MAX_DURATION_S:
            raise HTTPException(
                status_code=400,
                detail=f"duration_s must be between {TCP_TARGET_MIN_DURATION_S} and {TCP_TARGET_MAX_DURATION_S}",
            )
        current = snapshot.joint_positions
        quat = _rpy_to_quat(*rpy)
        expected = None
        if body.expected_joints_rad is not None:
            expected = _coerce_vector(body.expected_joints_rad, len(DEFAULT_JOINT_NAMES), "expected_joints_rad")
        try:
            target_joints = await _solve_ik_near_current(
                xyz, quat, current, expected, body.avoid_collisions
            )
            _ensure_joint_target_within_limits(target_joints, joint_limits)
            plan = motion.build_joint_target_plan(
                current,
                target_joints,
                get_replay_limits(),
                min_duration=duration,
            )
        except IKServiceUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except IKFailed as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except ReplayDataError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return plan, target_joints

    async def _solve_ik_near_current(
        xyz,
        quat,
        current,
        expected,
        avoid_collisions: bool,
    ) -> tuple[float, ...]:
        """Solve IK preferring the branch reachable with the least joint motion.

        Short-timeout calls keep KDL on its (deterministic) seeded attempt; the
        seeds are the current joints plus growing perturbations, which also
        escapes the singular home pose where the seeded attempt cannot converge.
        A fixed RNG seed keeps repeated validations reproducible.
        """
        solutions: list[tuple[float, ...]] = []
        last_ik_error: IKFailed | None = None

        async def attempt(seed_positions, timeout_s: float) -> tuple[float, ...] | None:
            nonlocal last_ik_error
            try:
                raw = await bridge.solve_ik(
                    xyz, quat, seed_positions,
                    avoid_collisions=avoid_collisions,
                    timeout_s=timeout_s,
                )
            except IKFailed as exc:
                last_ik_error = exc
                return None
            wrapped = motion.wrap_joints_toward_seed(raw, current, joint_limits)
            solutions.append(wrapped)
            return wrapped

        # Execution re-solves IK: seeding with the previously validated joints
        # makes KDL converge back onto the exact previewed branch.
        if expected is not None:
            solution = await attempt(expected, TCP_IK_SEEDED_TIMEOUT_S)
            if solution is not None and motion.max_joint_delta(solution, expected) <= TCP_EXECUTE_MATCH_TOLERANCE_RAD:
                return solution

        rng = random.Random(1234)
        for attempt_index in range(TCP_IK_ATTEMPTS):
            scale = TCP_IK_SEED_PERTURBATION_RAD * attempt_index
            seed_positions = tuple(
                _clamp_to_limit(joint_name, value + rng.uniform(-scale, scale))
                for joint_name, value in zip(DEFAULT_JOINT_NAMES, current, strict=True)
            )
            solution = await attempt(seed_positions, TCP_IK_SEEDED_TIMEOUT_S)
            if solution is not None and motion.max_joint_delta(solution, current) <= TCP_IK_CLOSE_ENOUGH_RAD:
                break

        if not solutions:
            # Last resort: let KDL random-restart; the wrap + preview still apply.
            for _ in range(TCP_IK_FALLBACK_ATTEMPTS):
                await attempt(current, TCP_IK_FALLBACK_TIMEOUT_S)
        if not solutions:
            raise IKFailed(
                "MoveIt IK found no solution for this pose. Poses at the UR wrist "
                "singularity (wrist_2 = 0, e.g. keeping the exact home orientation) "
                "often fail: jog wrist_2 a few degrees or tilt the target slightly, "
                f"then validate again ({last_ik_error})"
            )
        return motion.select_closest_ik_solution(solutions, current)

    def _clamp_to_limit(joint_name: str, value: float) -> float:
        limit = joint_limits.get(joint_name)
        if limit is None or not limit.has_position_limits:
            return value
        return min(max(value, limit.min_position), limit.max_position)

    # ------------------------------------------------------------------ websocket

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await ws.accept()
        events: asyncio.Queue = asyncio.Queue()
        listener = events.put_nowait
        bridge.add_goal_listener(listener)
        try:
            while True:
                while not events.empty():
                    await ws.send_json({"type": "goal_event", "goal": events.get_nowait()})
                await ws.send_json(_state_message(bridge))
                await asyncio.sleep(WS_PERIOD_S)
        except WebSocketDisconnect:
            pass
        finally:
            bridge.remove_goal_listener(listener)

    return app


def _state_message(bridge: RosBridge) -> dict:
    snapshot = bridge.get_snapshot()
    joints = None
    if snapshot.joint_positions is not None:
        joints = {
            "names": list(DEFAULT_JOINT_NAMES),
            "positions_rad": list(snapshot.joint_positions),
            "velocities_rad_s": list(snapshot.joint_velocities or ()),
        }
    tcp = None
    if snapshot.tcp_xyz is not None:
        tcp = {
            "parent": "base",
            "child": "tool0",
            "xyz_m": list(snapshot.tcp_xyz),
            "rpy_rad": list(snapshot.tcp_rpy),
            "quat_xyzw": list(snapshot.tcp_quat_xyzw),
        }
    return {
        "type": "state",
        "stamp": time.time(),
        "joints": joints,
        "tcp": tcp,
        "goal": bridge.goal_to_dict(snapshot.goal),
        "driver": {
            "joint_states_alive": snapshot.joint_states_alive,
            "action_server_ready": snapshot.action_server_ready,
            "controller_active": snapshot.controller_active,
            "ik_service_ready": snapshot.ik_service_ready,
            "dashboard_available": snapshot.dashboard_available,
            "speed_scaling": snapshot.speed_scaling,
            "program_running": snapshot.program_running,
        },
    }


def _json_safe(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _coerce_vector(values: list[float], expected_len: int, label: str) -> tuple[float, ...]:
    if len(values) != expected_len:
        raise HTTPException(status_code=400, detail=f"{label} must contain {expected_len} values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise HTTPException(status_code=400, detail=f"{label} must contain finite values")
    return result


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _ensure_joint_target_within_limits(target: tuple[float, ...], limits: dict[str, JointLimit]) -> None:
    tolerance = 1.0e-6
    for joint_name, value in zip(DEFAULT_JOINT_NAMES, target, strict=True):
        limit = limits.get(joint_name)
        if limit is None or not limit.has_position_limits:
            continue
        if value < limit.min_position - tolerance or value > limit.max_position + tolerance:
            raise ReplayDataError(f"IK target for {joint_name} is outside joint position limits")


def _safety_limits_to_dict(limits: SafetyLimits) -> dict:
    return {
        "max_joint_velocity": limits.max_joint_velocity,
        "max_joint_acceleration": limits.max_joint_acceleration,
        "approach_min_duration": limits.approach_min_duration,
        "min_segment_duration": limits.min_segment_duration,
    }


def _replay_settings_payload(limits: SafetyLimits, bounds: dict[str, dict[str, float]]) -> dict:
    return {
        "limits": _safety_limits_to_dict(limits),
        "bounds": bounds,
        "presets": {name: _safety_limits_to_dict(preset) for name, preset in REPLAY_PRESETS.items()},
    }


def _validate_replay_limits(limits: SafetyLimits, bounds: dict[str, dict[str, float]]) -> None:
    limits.validate()
    for name, spec in bounds.items():
        value = getattr(limits, name)
        if not math.isfinite(value):
            raise ReplayDataError(f"{name} must be finite")
        if value < spec["min"] or value > spec["max"]:
            raise ReplayDataError(f"{name} must be between {spec['min']} and {spec['max']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Web UI for UR3e control and rollout replay.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (use 0.0.0.0 to expose on LAN)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rollout", default=str(DEFAULT_ROLLOUT_PATH), help="Path to rollouts_*.json")
    parser.add_argument("--max-joint-velocity", type=float, default=0.25, help="Safety velocity limit in rad/s")
    parser.add_argument("--max-joint-acceleration", type=float, default=0.5, help="Safety acceleration limit in rad/s^2")
    parser.add_argument("--approach-min-duration", type=float, default=10.0, help="Minimum approach duration in seconds")
    parser.add_argument("--min-segment-duration", type=float, default=0.5, help="Minimum replay segment duration in seconds")
    parser.add_argument(
        "--home-joints",
        default=",".join(str(v) for v in motion.HOME_POSITIONS),
        help="Comma-separated six-joint home pose in radians",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = build_parser().parse_args(argv)
    home = tuple(float(part.strip()) for part in args.home_joints.split(","))
    if len(home) != len(DEFAULT_JOINT_NAMES):
        print(f"ERROR: --home-joints must contain {len(DEFAULT_JOINT_NAMES)} values")
        return 2
    settings = Settings(
        rollout_path=args.rollout,
        limits=SafetyLimits(
            max_joint_velocity=args.max_joint_velocity,
            max_joint_acceleration=args.max_joint_acceleration,
            approach_min_duration=args.approach_min_duration,
            min_segment_duration=args.min_segment_duration,
        ),
        home_positions=home,
    )
    app = create_app(RosBridge(), settings)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
