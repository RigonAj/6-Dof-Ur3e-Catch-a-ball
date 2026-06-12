from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import math
import threading
import time
from typing import Any, Callable, Sequence

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)

from ur3e_rollout_replay.replay_core import DEFAULT_JOINT_NAMES
from ur3e_rollout_replay.send import build_joint_trajectory

from .urdf_provider import UrdfCache

ACTION_NAME = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
TRAJECTORY_CONTROLLER_NAME = "scaled_joint_trajectory_controller"
LIST_CONTROLLERS_SERVICE = "/controller_manager/list_controllers"
SWITCH_CONTROLLER_SERVICE = "/controller_manager/switch_controller"
CONTROLLER_POLL_TICKS = 10  # poll controller state every N 0.1s timer ticks
IK_SERVICE_NAME = "/compute_ik"
IK_GROUP_NAME = "ur_manipulator"
IK_LINK_NAME = "tool0"
IK_PARENT_FRAME = "base_link"
TCP_PARENT_FRAME = "base"
TCP_CHILD_FRAME = "tool0"
JOINT_STATE_STALE_S = 1.0

DASHBOARD_COMMANDS = ("play", "stop", "power_on", "power_off", "brake_release")

ERROR_CODE_STRINGS = {
    0: "successful",
    -1: "invalid goal",
    -2: "invalid joints",
    -3: "old header timestamp",
    -4: "path tolerance violated",
    -5: "goal tolerance violated",
}


class MotionBusy(Exception):
    """Another motion goal is active and may not be preempted by this one."""


class ActionServerUnavailable(Exception):
    """The trajectory action server is not ready."""


class IKServiceUnavailable(Exception):
    """The MoveIt inverse-kinematics service is not ready."""


class IKFailed(Exception):
    """MoveIt could not solve inverse kinematics for the requested pose."""


@dataclass
class GoalRecord:
    kind: str  # "jog" | "home" | "rollout" | "tcp" | "calibration"
    phase: str  # "sending" | "active" | "succeeded" | "aborted" | "canceled" | "rejected"
    episode: int | None = None
    started_monotonic: float = 0.0
    total_s: float = 0.0
    last_jog_target: tuple[float, ...] | None = None
    error_code: int | None = None
    error_string: str | None = None
    goal_id: bytes | None = None
    goal_handle: Any = None


@dataclass
class StateSnapshot:
    joint_positions: tuple[float, ...] | None = None
    joint_velocities: tuple[float, ...] | None = None
    joint_state_monotonic: float = 0.0
    tcp_xyz: tuple[float, float, float] | None = None
    tcp_rpy: tuple[float, float, float] | None = None
    tcp_quat_xyzw: tuple[float, float, float, float] | None = None
    speed_scaling: float | None = None
    program_running: bool | None = None
    action_server_ready: bool = False
    # The UR controller_stopper deactivates the trajectory controller whenever
    # External Control stops; the action server then still exists but rejects
    # every goal, so track the actual controller state separately.
    controller_active: bool | None = None
    ik_service_ready: bool = False
    dashboard_available: bool = False
    goal: GoalRecord | None = None

    @property
    def joint_states_alive(self) -> bool:
        return (
            self.joint_positions is not None
            and (time.monotonic() - self.joint_state_monotonic) < JOINT_STATE_STALE_S
        )


def _quat_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class RosBridge:
    """Owns the rclpy node and executor thread; exposes a thread-safe snapshot
    plus asyncio-friendly motion/dashboard calls for the FastAPI handlers."""

    def __init__(self) -> None:
        self.urdf_cache = UrdfCache()
        self._lock = threading.Lock()
        self._snapshot = StateSnapshot()
        self._goal: GoalRecord | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._goal_listeners: list[Callable[[dict], None]] = []
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._thread: threading.Thread | None = None
        self._dashboard_clients: dict[str, Any] = {}
        self._ik_client: Any = None
        self._list_controllers_client: Any = None
        self._switch_controller_client: Any = None
        self._controller_list_future: Any = None
        self._controller_poll_tick = 0

    # ---------------------------------------------------------------- lifecycle

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        rclpy.init()
        node = Node("ur3e_web_ui")
        self._node = node

        node.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        node.create_subscription(StringMsg, "/robot_description", self._on_robot_description, latched)
        node.create_subscription(
            Float64, "/speed_scaling_state_broadcaster/speed_scaling", self._on_speed_scaling, latched
        )
        node.create_subscription(
            Bool, "/io_and_status_controller/robot_program_running", self._on_program_running, latched
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        node.create_timer(0.1, self._on_poll_timer)

        self._action_client = ActionClient(node, FollowJointTrajectory, ACTION_NAME)
        self._ik_client = node.create_client(GetPositionIK, IK_SERVICE_NAME)
        self._list_controllers_client = node.create_client(ListControllers, LIST_CONTROLLERS_SERVICE)
        self._switch_controller_client = node.create_client(SwitchController, SWITCH_CONTROLLER_SERVICE)
        for command in DASHBOARD_COMMANDS:
            self._dashboard_clients[command] = node.create_client(Trigger, f"/dashboard_client/{command}")

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True, name="ros-executor")
        self._thread.start()

    def shutdown(self) -> None:
        goal = self._current_goal()
        if goal is not None and goal.phase in ("sending", "active") and goal.goal_handle is not None:
            try:
                goal.goal_handle.cancel_goal_async()
                time.sleep(0.5)
            except Exception:
                pass
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ---------------------------------------------------------------- callbacks (executor thread)

    def _on_joint_state(self, msg: JointState) -> None:
        by_name = dict(zip(msg.name, msg.position, strict=False))
        if not all(name in by_name for name in DEFAULT_JOINT_NAMES):
            return
        velocities_by_name = dict(zip(msg.name, msg.velocity, strict=False)) if msg.velocity else {}
        positions = tuple(float(by_name[name]) for name in DEFAULT_JOINT_NAMES)
        velocities = tuple(float(velocities_by_name.get(name, 0.0)) for name in DEFAULT_JOINT_NAMES)
        with self._lock:
            self._snapshot.joint_positions = positions
            self._snapshot.joint_velocities = velocities
            self._snapshot.joint_state_monotonic = time.monotonic()

    def _on_robot_description(self, msg: StringMsg) -> None:
        if msg.data:
            self.urdf_cache.set_from_topic(msg.data)

    def _on_speed_scaling(self, msg: Float64) -> None:
        with self._lock:
            self._snapshot.speed_scaling = float(msg.data)

    def _on_program_running(self, msg: Bool) -> None:
        with self._lock:
            self._snapshot.program_running = bool(msg.data)

    def _on_poll_timer(self) -> None:
        tcp_xyz = tcp_rpy = tcp_quat = None
        try:
            transform = self._tf_buffer.lookup_transform(TCP_PARENT_FRAME, TCP_CHILD_FRAME, rclpy.time.Time())
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            tcp_xyz = (translation.x, translation.y, translation.z)
            tcp_quat = (rotation.x, rotation.y, rotation.z, rotation.w)
            tcp_rpy = _quat_to_rpy(*tcp_quat)
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass

        action_ready = self._action_client.server_is_ready()
        ik_ready = self._ik_client.service_is_ready() if self._ik_client is not None else False
        dashboard_ready = self._dashboard_clients["play"].service_is_ready()
        with self._lock:
            self._snapshot.tcp_xyz = tcp_xyz
            self._snapshot.tcp_rpy = tcp_rpy
            self._snapshot.tcp_quat_xyzw = tcp_quat
            self._snapshot.action_server_ready = action_ready
            self._snapshot.ik_service_ready = ik_ready
            self._snapshot.dashboard_available = dashboard_ready

        self._controller_poll_tick += 1
        if self._controller_poll_tick >= CONTROLLER_POLL_TICKS:
            self._controller_poll_tick = 0
            self._poll_controller_state()

    def _poll_controller_state(self) -> None:
        if self._list_controllers_client is None or not self._list_controllers_client.service_is_ready():
            return
        if self._controller_list_future is not None and not self._controller_list_future.done():
            return
        future = self._list_controllers_client.call_async(ListControllers.Request())
        self._controller_list_future = future
        future.add_done_callback(self._on_controller_list)

    def _on_controller_list(self, future: Any) -> None:
        try:
            response = future.result()
        except Exception:
            return
        active = None
        for controller in response.controller:
            if controller.name == TRAJECTORY_CONTROLLER_NAME:
                active = controller.state == "active"
                break
        with self._lock:
            self._snapshot.controller_active = active

    # ---------------------------------------------------------------- snapshot access

    def get_snapshot(self) -> StateSnapshot:
        with self._lock:
            snapshot = replace(self._snapshot)
            snapshot.goal = replace(self._goal) if self._goal is not None else None
        return snapshot

    def _current_goal(self) -> GoalRecord | None:
        with self._lock:
            return replace(self._goal) if self._goal is not None else None

    def add_goal_listener(self, listener: Callable[[dict], None]) -> None:
        """listener is called on the asyncio loop thread with a goal dict."""
        self._goal_listeners.append(listener)

    def remove_goal_listener(self, listener: Callable[[dict], None]) -> None:
        if listener in self._goal_listeners:
            self._goal_listeners.remove(listener)

    def goal_to_dict(self, goal: GoalRecord | None) -> dict:
        if goal is None:
            return {"phase": "idle", "kind": None}
        elapsed = time.monotonic() - goal.started_monotonic if goal.started_monotonic else 0.0
        return {
            "phase": goal.phase,
            "kind": goal.kind,
            "episode": goal.episode,
            "elapsed_s": round(min(elapsed, goal.total_s if goal.phase == "active" else elapsed), 2),
            "total_s": round(goal.total_s, 2),
            "error_code": goal.error_code,
            "error_string": goal.error_string,
        }

    def _notify_goal_event(self) -> None:
        """Safe to call from any thread."""
        if self._loop is None:
            return
        payload = self.goal_to_dict(self._current_goal())
        for listener in list(self._goal_listeners):
            self._loop.call_soon_threadsafe(listener, payload)

    # ---------------------------------------------------------------- future bridge

    async def _await_ros_future(self, ros_future: Any) -> Any:
        assert self._loop is not None
        aio_future: asyncio.Future = self._loop.create_future()

        def _done(finished: Any) -> None:
            def _resolve() -> None:
                if aio_future.cancelled():
                    return
                exception = finished.exception()
                if exception is not None:
                    aio_future.set_exception(exception)
                else:
                    aio_future.set_result(finished.result())

            self._loop.call_soon_threadsafe(_resolve)

        ros_future.add_done_callback(_done)
        return await aio_future

    # ---------------------------------------------------------------- motion

    async def send_plan(self, plan: Any, kind: str, episode: int | None = None) -> GoalRecord:
        """Send a FollowJointTrajectory goal built from plan.

        Raises MotionBusy / ActionServerUnavailable. Returns the accepted GoalRecord.
        """
        if not self._action_client.server_is_ready():
            raise ActionServerUnavailable(f"action server not ready: {ACTION_NAME}")

        with self._lock:
            active = self._goal is not None and self._goal.phase in ("sending", "active")
            if active and not (kind == "jog" and self._goal.kind == "jog"):
                raise MotionBusy(f"a {self._goal.kind} goal is already {self._goal.phase}")
            record = GoalRecord(
                kind=kind,
                phase="sending",
                episode=episode,
                started_monotonic=time.monotonic(),
                total_s=float(plan.time_from_start_s[-1]),
                last_jog_target=tuple(plan.positions[-1]) if kind == "jog" else None,
            )
            self._goal = record

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = build_joint_trajectory(plan, JointTrajectory, JointTrajectoryPoint, Duration)

        goal_handle = await self._await_ros_future(self._action_client.send_goal_async(goal))
        goal_id = bytes(goal_handle.goal_id.uuid)

        with self._lock:
            if self._goal is not record:
                # A newer jog preempted us while the send was in flight.
                if goal_handle.accepted:
                    goal_handle.cancel_goal_async()
                return record
            if not goal_handle.accepted:
                record.phase = "rejected"
                self._goal = record
            else:
                record.phase = "active"
                record.goal_id = goal_id
                record.goal_handle = goal_handle
                record.started_monotonic = time.monotonic()

        if record.phase == "rejected":
            self._notify_goal_event()
            return record

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda finished: self._on_goal_result(goal_id, finished))
        self._notify_goal_event()
        return record

    def _on_goal_result(self, goal_id: bytes, finished: Any) -> None:
        """Runs on the executor thread."""
        try:
            wrapped = finished.result()
        except Exception as exc:  # result retrieval failed
            status, error_code, error_string = None, None, str(exc)
        else:
            status = wrapped.status
            error_code = int(getattr(wrapped.result, "error_code", 0))
            error_string = getattr(wrapped.result, "error_string", "") or ERROR_CODE_STRINGS.get(error_code, "")

        with self._lock:
            if self._goal is None or self._goal.goal_id != goal_id:
                return  # stale result from a preempted jog goal
            if status == 5:  # STATUS_CANCELED
                self._goal.phase = "canceled"
            elif status == 4 and (error_code or 0) == 0:  # STATUS_SUCCEEDED
                self._goal.phase = "succeeded"
            else:
                self._goal.phase = "aborted"
            self._goal.error_code = error_code
            self._goal.error_string = error_string
            self._goal.goal_handle = None
        self._notify_goal_event()

    async def cancel_active(self) -> bool:
        goal_handle = None
        with self._lock:
            if self._goal is not None and self._goal.phase in ("sending", "active"):
                goal_handle = self._goal.goal_handle
        if goal_handle is None:
            return False
        await self._await_ros_future(goal_handle.cancel_goal_async())
        return True

    def active_jog_base(self) -> tuple[float, ...] | None:
        """Last commanded jog target if a jog goal is in flight (for smooth hold-to-jog)."""
        with self._lock:
            if (
                self._goal is not None
                and self._goal.kind == "jog"
                and self._goal.phase in ("sending", "active")
                and self._goal.last_jog_target is not None
            ):
                return self._goal.last_jog_target
        return None

    async def activate_trajectory_controller(self) -> bool:
        """Reactivate the scaled trajectory controller (left inactive when the
        controller_stopper raced External Control start). Holds position; no motion."""
        if self._switch_controller_client is None or not self._switch_controller_client.service_is_ready():
            return False
        request = SwitchController.Request()
        request.activate_controllers = [TRAJECTORY_CONTROLLER_NAME]
        request.strictness = SwitchController.Request.BEST_EFFORT
        response = await self._await_ros_future(self._switch_controller_client.call_async(request))
        if not bool(response.ok):
            return False
        with self._lock:
            self._snapshot.controller_active = True
        return True

    # ---------------------------------------------------------------- inverse kinematics

    async def solve_ik(
        self,
        xyz_m: Sequence[float],
        quat_xyzw: Sequence[float],
        seed_positions: Sequence[float],
        avoid_collisions: bool = True,
        timeout_s: float = 0.5,
    ) -> tuple[float, ...]:
        if self._ik_client is None or not self._ik_client.service_is_ready():
            raise IKServiceUnavailable(f"MoveIt IK service not available: {IK_SERVICE_NAME}")

        request = GetPositionIK.Request()
        request.ik_request.group_name = IK_GROUP_NAME
        request.ik_request.ik_link_name = IK_LINK_NAME
        request.ik_request.avoid_collisions = bool(avoid_collisions)
        request.ik_request.timeout = _duration_from_seconds(timeout_s)

        request.ik_request.robot_state.joint_state.name = list(DEFAULT_JOINT_NAMES)
        request.ik_request.robot_state.joint_state.position = [float(value) for value in seed_positions]
        request.ik_request.robot_state.is_diff = True

        ik_xyz, ik_quat = _base_pose_to_base_link(xyz_m, quat_xyzw)
        pose = PoseStamped()
        pose.header.frame_id = IK_PARENT_FRAME
        pose.pose.position.x = float(ik_xyz[0])
        pose.pose.position.y = float(ik_xyz[1])
        pose.pose.position.z = float(ik_xyz[2])
        pose.pose.orientation.x = float(ik_quat[0])
        pose.pose.orientation.y = float(ik_quat[1])
        pose.pose.orientation.z = float(ik_quat[2])
        pose.pose.orientation.w = float(ik_quat[3])
        request.ik_request.pose_stamped = pose

        response = await self._await_ros_future(self._ik_client.call_async(request))
        error_code = int(getattr(response.error_code, "val", 0))
        if error_code != 1:  # MoveItErrorCodes.SUCCESS
            raise IKFailed(f"MoveIt IK failed with error code {error_code}")

        by_name = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position, strict=False))
        if not all(name in by_name for name in DEFAULT_JOINT_NAMES):
            raise IKFailed("MoveIt IK response did not contain all UR3e joints")
        return tuple(float(by_name[name]) for name in DEFAULT_JOINT_NAMES)

    # ---------------------------------------------------------------- dashboard

    async def call_dashboard(self, command: str) -> tuple[bool, str]:
        client = self._dashboard_clients.get(command)
        if client is None:
            raise KeyError(command)
        if not client.service_is_ready():
            raise ActionServerUnavailable(f"dashboard service not available: /dashboard_client/{command}")
        response = await self._await_ros_future(client.call_async(Trigger.Request()))
        return bool(response.success), str(response.message)


def _duration_from_seconds(seconds: float) -> Duration:
    whole = int(math.floor(seconds))
    return Duration(sec=whole, nanosec=int((seconds - whole) * 1_000_000_000))


def _base_pose_to_base_link(
    xyz_m: Sequence[float],
    quat_xyzw: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    # UR's "base" frame is rotated pi around Z from the REP-103 base_link frame.
    xyz = (-float(xyz_m[0]), -float(xyz_m[1]), float(xyz_m[2]))
    quat = _quat_multiply((0.0, 0.0, 1.0, 0.0), quat_xyzw)
    return xyz, quat


def _quat_multiply(
    lhs_xyzw: Sequence[float],
    rhs_xyzw: Sequence[float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = (float(value) for value in lhs_xyzw)
    x2, y2, z2, w2 = (float(value) for value in rhs_xyzw)
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )
