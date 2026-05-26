from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import sample_uniform

from .firsttraining_env_cfg import FirsttrainingEnvCfg


class FirsttrainingEnv(DirectRLEnv):
    cfg: FirsttrainingEnvCfg

    def __init__(self, cfg: FirsttrainingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._disk_body_idx, _ = self.robot.find_bodies(self.cfg.disk_link_name)
        self._arm_dof_idx, _ = self.robot.find_joints(self.cfg.joint_names)
        self._contact_on_arm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        force_shape = self.contact_sensor_arm.data.net_forces_w.shape[:-1]
        self._contact_force_norm = torch.zeros(force_shape, device=self.device)
        self._contact_force_hits = torch.zeros(force_shape, dtype=torch.bool, device=self.device)
        # Pre-allocate cached tensors (avoids per-step allocation)
        self._prev_disk_signed_dist = torch.zeros(self.num_envs, device=self.device)
        self.pass_through_count = torch.zeros(self.num_envs, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._passed_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_done_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._zero_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.actions = torch.zeros(self.num_envs, len(self.cfg.joint_names), device=self.device)
        self.joint_pos_target = torch.zeros_like(self.actions)
        self._prev_d = torch.zeros(self.num_envs, device=self.device)
        self._identity_quat_wxyz = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)

        # Cached per-step tensors (filled in _pre_physics_step)
        self._disk_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._disk_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._ball_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._ball_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._disk_pos_local = torch.zeros(self.num_envs, 3, device=self.device)
        self._ball_pos_local = torch.zeros(self.num_envs, 3, device=self.device)
        self._direction = torch.zeros(self.num_envs, 3, device=self.device)
        self._distance = torch.zeros(self.num_envs, 1, device=self.device)
        self._delta_d = torch.zeros(self.num_envs, device=self.device)
        self._local_pose_cache_valid = False

        disk_offset_b, disk_normal_b, disk_radius = self._read_disk_pose_in_body_frame()
        self._disk_offset_b = torch.tensor(disk_offset_b, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        self._disk_normal_b = torch.tensor(disk_normal_b, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        self._disk_radius = self.cfg.disk_radius if self.cfg.disk_radius > 0.0 else disk_radius

        print("=== Joint names ===")
        print(self.robot.joint_names)

    def _read_disk_pose_in_body_frame(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        robot_root_path = f"/World/envs/env_0/Robot/{self.cfg.robot_usd_root_name}"
        body_path = f"{robot_root_path}/{self.cfg.disk_link_name}"
        disk_path = f"/World/envs/env_0/Robot/{self.cfg.disk_prim_rel_path.strip('/')}"

        body_prim = stage.GetPrimAtPath(body_path)
        disk_prim = stage.GetPrimAtPath(disk_path)
        if not body_prim.IsValid():
            raise RuntimeError(f"Disk body prim not found: {body_path}")
        if not disk_prim.IsValid():
            raise RuntimeError(f"Disk trigger prim not found: {disk_path}")

        cache = UsdGeom.XformCache()
        body_to_world = cache.GetLocalToWorldTransform(body_prim)
        disk_to_world = cache.GetLocalToWorldTransform(disk_prim)
        world_to_body = body_to_world.GetInverse()

        mesh = UsdGeom.Mesh(disk_prim)
        points = mesh.GetPointsAttr().Get() if mesh else None
        if points:
            min_point = Gf.Vec3d(*(min(point[index] for point in points) for index in range(3)))
            max_point = Gf.Vec3d(*(max(point[index] for point in points) for index in range(3)))
            disk_center_local = (min_point + max_point) * 0.5
        else:
            disk_center_local = Gf.Vec3d(0.0, 0.0, 0.0)

        disk_center_w = disk_to_world.Transform(disk_center_local)
        disk_axis_local = disk_center_local + Gf.Vec3d(*self.cfg.disk_local_normal_axis)
        disk_axis_w = disk_to_world.Transform(disk_axis_local)
        disk_center_b = world_to_body.Transform(disk_center_w)
        disk_axis_b = world_to_body.Transform(disk_axis_w)
        disk_normal_b = disk_axis_b - disk_center_b
        normal_length = disk_normal_b.GetLength()
        if normal_length <= 1.0e-8:
            raise RuntimeError(f"Disk trigger normal is degenerate: {disk_path}")

        disk_normal_b = Gf.Vec3d(*(disk_normal_b[i] / normal_length for i in range(3)))
        radius = 0.0
        if points:
            for point in points:
                point_local = Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                point_b = world_to_body.Transform(disk_to_world.Transform(point_local))
                relative = point_b - disk_center_b
                along_normal = disk_normal_b * Gf.Dot(relative, disk_normal_b)
                radius = max(radius, (relative - along_normal).GetLength())
        if radius <= 1.0e-8:
            if self.cfg.disk_radius <= 0.0:
                raise RuntimeError(f"Unable to infer disk trigger radius from mesh: {disk_path}")
            radius = self.cfg.disk_radius

        offset = tuple(float(disk_center_b[i]) for i in range(3))
        normal = tuple(float(disk_normal_b[i]) for i in range(3))
        print(f"=== Disk trigger === path={disk_path} offset={offset} normal={normal} radius={radius}")
        return offset, normal, float(radius)

    def _update_cached_poses(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            body_quat = self.robot.data.body_quat_w[:, self._disk_body_idx].squeeze(1)
            body_pos = self.robot.data.body_pos_w[:, self._disk_body_idx].squeeze(1)
            self._disk_pos_w.copy_(body_pos + quat_rotate_wxyz(body_quat, self._disk_offset_b))
            self._disk_normal_w.copy_(quat_rotate_wxyz(body_quat, self._disk_normal_b))
            self._ball_pos_w.copy_(self.ball.data.root_pos_w)
            self._ball_vel_w.copy_(self.ball.data.root_lin_vel_w)
        else:
            body_quat = self.robot.data.body_quat_w[env_ids, self._disk_body_idx].squeeze(1)
            body_pos = self.robot.data.body_pos_w[env_ids, self._disk_body_idx].squeeze(1)
            self._disk_pos_w[env_ids].copy_(body_pos + quat_rotate_wxyz(body_quat, self._disk_offset_b[env_ids]))
            self._disk_normal_w[env_ids].copy_(quat_rotate_wxyz(body_quat, self._disk_normal_b[env_ids]))
            self._ball_pos_w[env_ids].copy_(self.ball.data.root_pos_w[env_ids])
            self._ball_vel_w[env_ids].copy_(self.ball.data.root_lin_vel_w[env_ids])

    def _update_local_pose_tensors(self) -> None:
        self._disk_pos_local.copy_(self._disk_pos_w).sub_(self.scene.env_origins)
        self._ball_pos_local.copy_(self._ball_pos_w).sub_(self.scene.env_origins)
        self._direction.copy_(self._ball_pos_local).sub_(self._disk_pos_local)
        torch.linalg.vector_norm(self._direction, dim=-1, keepdim=True, out=self._distance)
        self._local_pose_cache_valid = True

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.ball = RigidObject(self.cfg.ball_cfg)
        self.contact_sensor_arm = ContactSensor(self.cfg.contact_sensor_cfg_arm)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        self.scene.sensors["contact_sensor_arm"] = self.contact_sensor_arm

        if self.cfg.enable_markers or self.cfg.enable_disk_center_marker:
            disk_center_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/DiskCenterMarker",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=self.cfg.disk_center_marker_radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self.disk_center_marker = VisualizationMarkers(disk_center_marker_cfg)
        else:
            self.disk_center_marker = None
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions.copy_(actions)
        torch.mul(actions, self.cfg.action_scale, out=self.joint_pos_target)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.joint_pos_target,
            joint_ids=self._arm_dof_idx
        )

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, self._arm_dof_idx]
        joint_vel = self.robot.data.joint_vel[:, self._arm_dof_idx]

        if not self._local_pose_cache_valid:
            self._update_local_pose_tensors()

        obs = torch.cat([
            joint_pos,                                      # 6
            joint_vel,                                      # 6
            self._disk_pos_local,                           # 3
            self._ball_pos_local,                           # 3
            self._direction,                                # 3
            self._distance,                                 # 1
            self._ball_vel_w,                               # 3
            (self._prev_disk_signed_dist > 0.0).float().unsqueeze(-1),  # 1
            self.actions,                                   # 6
            self.pass_through_count.unsqueeze(-1),          # 1
        ], dim=-1)

        if self.disk_center_marker is not None:
            self.disk_center_marker.visualize(self._disk_pos_w)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._update_local_pose_tensors()
        self._delta_d.copy_(self._prev_d).sub_(self._distance[:, 0])
        self._prev_d.copy_(self._distance[:, 0])

        return compute_rewards(
            self._distance[:, 0],
            self.actions,
            self.reset_terminated,
            self._passed_this_step,
            self._delta_d,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_cached_poses()

        forces_arm = self.contact_sensor_arm.data.net_forces_w
        torch.linalg.vector_norm(forces_arm, dim=-1, out=self._contact_force_norm)
        torch.gt(self._contact_force_norm, 0.1, out=self._contact_force_hits)
        torch.any(self._contact_force_hits, dim=-1, out=self._contact_on_arm)

        passed, signed_dist = detect_pass_through(
            self._disk_pos_w,
            self._ball_pos_w,
            self._disk_normal_w,
            self._prev_disk_signed_dist,
            self._disk_radius,
        )
        self._passed_this_step.copy_(passed)
        self._prev_disk_signed_dist.copy_(signed_dist)
        self.pass_through_count += passed.float()
        self.episode_success |= passed

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        ball_on_ground = self._ball_pos_w[:, 2] < 0.05
        is_caught = self.pass_through_count > 0.0
        hit_arm = self._contact_on_arm
        success_done = is_caught if self.cfg.reset_on_success else self._zero_done
        terminated = success_done | hit_arm | ball_on_ground
        done = terminated | time_out
        self._last_done_success[:] = False
        self._last_done_success[done] = self.episode_success[done]
        self.extras["success"] = self._last_done_success

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot joints
        joint_pos = torch.zeros((len(env_ids), len(self._arm_dof_idx)), device=self.device)
        joint_pos[:, 0] = sample_uniform(-0.785, 0.785, len(env_ids), self.device)
        joint_pos[:, 1] = -1.57 + sample_uniform(-0.9, 0.2, len(env_ids), self.device)
        joint_pos[:, 2] = sample_uniform(-0.2, 0.2, len(env_ids), self.device)
        joint_pos[:, 3] = -1.57 + sample_uniform(-0.2, 0.2, len(env_ids), self.device)
        joint_pos[:, 4] = sample_uniform(-0.2, 0.2, len(env_ids), self.device)
        joint_pos[:, 5] = sample_uniform(-0.2, 0.2, len(env_ids), self.device)
        joint_vel = torch.zeros_like(joint_pos)

        self.actions[env_ids] = 0.0
        self.joint_pos_target[env_ids] = joint_pos
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids, joint_ids=self._arm_dof_idx)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self._arm_dof_idx, env_ids=env_ids)

        ball_state = self.ball.data.default_root_state[env_ids].clone()

        ball_state[:, 0] = sample_uniform(*self.cfg.ball_spawn_x_range, len(env_ids), self.device)
        ball_state[:, 1] = sample_uniform(*self.cfg.ball_spawn_y_range, len(env_ids), self.device)
        ball_state[:, 2] = sample_uniform(*self.cfg.ball_spawn_z_range, len(env_ids), self.device)
        if self.cfg.enable_ball_position_noise:
            ball_state[:, 0:3] += torch.randn((len(env_ids), 3), device=self.device) * self.cfg.ball_position_noise_std

        # Convert to world coords
        ball_state[:, 0:3] += self.scene.env_origins[env_ids]

        ball_state[:, 3:7] = self._identity_quat_wxyz

        ball_state[:, 7] = sample_uniform(*self.cfg.ball_velocity_x_range, len(env_ids), self.device)
        ball_state[:, 8] = sample_uniform(*self.cfg.ball_velocity_y_range, len(env_ids), self.device)
        ball_state[:, 9] = sample_uniform(*self.cfg.ball_velocity_z_range, len(env_ids), self.device)

        self.ball.write_root_state_to_sim(ball_state, env_ids=env_ids)
        self._update_cached_poses(env_ids)
        self._local_pose_cache_valid = False

        # Initialize _prev_d with the real distance at reset.
        self._prev_d[env_ids] = torch.norm(self._ball_pos_w[env_ids] - self._disk_pos_w[env_ids], dim=-1)

        # Reset tracking
        ball_relative = self._ball_pos_w[env_ids] - self._disk_pos_w[env_ids]
        self._prev_disk_signed_dist[env_ids] = (ball_relative * self._disk_normal_w[env_ids]).sum(dim=-1)
        self.pass_through_count[env_ids] = 0.0
        self.episode_success[env_ids] = False
        self._passed_this_step[env_ids] = False

        self._contact_on_arm[env_ids] = False


# JIT functions
@torch.jit.script
def quat_rotate_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_vec = quat[:, 1:4]
    quat_w = quat[:, 0:1]
    uv = torch.cross(quat_vec, vec, dim=-1)
    uuv = torch.cross(quat_vec, uv, dim=-1)
    return vec + 2.0 * (quat_w * uv + uuv)


@torch.jit.script
def detect_pass_through(
    disk_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    disk_normal: torch.Tensor,
    previous_signed_dist: torch.Tensor,
    disk_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ball_relative = ball_pos - disk_pos
    signed_dist = (ball_relative * disk_normal).sum(dim=-1)
    along_normal = signed_dist.unsqueeze(-1) * disk_normal
    radial_dist = torch.norm(ball_relative - along_normal, dim=-1)
    within_hoop = radial_dist < disk_radius
    crossed_plane_any_direction = previous_signed_dist * signed_dist < 0.0
    passed = crossed_plane_any_direction & within_hoop
    # if passed.any():
    #    num_passed = passed.sum().item()
    #    print(f"Goal! {num_passed} ball(s) passed through the hoop.")
    return passed, signed_dist


@torch.jit.script
def compute_rewards(
    distance: torch.Tensor,
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    passed: torch.Tensor,
    delta_d: torch.Tensor,
) -> torch.Tensor:
    approaching = (delta_d > 0.0).to(dtype=torch.float32)
    rew_dist = (torch.exp(-2.0 * distance) - 1.0 * distance + 5) * approaching
    rew_action = -0.5 * torch.sum(actions ** 2, dim=-1)
    rew_pass = 400.0 * passed.to(dtype=torch.float32)
    rew_termination = -100.0 * reset_terminated.to(dtype=torch.float32)
    return rew_action + rew_pass + rew_termination + rew_dist
