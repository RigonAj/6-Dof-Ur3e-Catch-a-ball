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
        self.raw_actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.joint_pos_target = torch.zeros_like(self.actions)
        self._action_penalty_coeff = torch.zeros(len(self.cfg.joint_names), device=self.device)
        self._joint_action_penalty_coeff_ranges = torch.tensor(
            self.cfg.joint_action_penalty_coeff_ranges,
            dtype=torch.float32,
            device=self.device,
        )
        if self._joint_action_penalty_coeff_ranges.shape != (len(self.cfg.joint_names), 2):
            raise ValueError(
                "joint_action_penalty_coeff_ranges must contain one (start, end) pair per joint in cfg.joint_names."
            )
        self._robot_pose_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ball_respawn_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ball_respawn_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reference_joint_pos_target = torch.zeros_like(self.actions)
        self._previous_joint_cmd_vel = torch.zeros_like(self.actions)
        self._step_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        self._joint_velocity_safe = torch.tensor(
            self.cfg.joint_velocity_safe_rad_s,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._joint_acceleration_safe = torch.tensor(
            self.cfg.joint_acceleration_safe_rad_s2,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._joint_position_lower = torch.tensor(
            self.cfg.joint_position_lower_rad,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._joint_position_upper = torch.tensor(
            self.cfg.joint_position_upper_rad,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._prev_d = torch.zeros(self.num_envs, device=self.device)
        self._identity_quat_wxyz = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)

        # Ring buffers for the delayed ball observation (perception latency model).
        delay_min, delay_max = self.cfg.ball_obs_delay_steps_range
        self._ball_obs_delay_min = max(0, int(delay_min))
        self._ball_obs_delay_max = max(self._ball_obs_delay_min, int(delay_max))
        self._ball_obs_buf_len = self._ball_obs_delay_max + 1
        self._ball_obs_pos_buf = torch.zeros(self._ball_obs_buf_len, self.num_envs, 3, device=self.device)
        self._ball_obs_vel_buf = torch.zeros_like(self._ball_obs_pos_buf)
        self._ball_obs_head = 0
        self._ball_obs_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_index = torch.arange(self.num_envs, device=self.device)

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

        # State after the last physics step and before Isaac Lab auto-resets done environments.
        self._last_step_joint_pos = torch.zeros(self.num_envs, len(self.cfg.joint_names), device=self.device)
        self._last_step_joint_vel = torch.zeros_like(self._last_step_joint_pos)
        self._last_step_joint_pos_target = torch.zeros_like(self._last_step_joint_pos)
        self._last_step_disk_pos_w = torch.zeros_like(self._disk_pos_w)
        self._last_step_disk_pos_local = torch.zeros_like(self._disk_pos_local)
        self._last_step_ball_pos_w = torch.zeros_like(self._ball_pos_w)
        self._last_step_ball_pos_local = torch.zeros_like(self._ball_pos_local)
        self._last_step_ball_vel_w = torch.zeros_like(self._ball_vel_w)
        self._last_step_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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

    def _cache_last_step_state(self) -> None:
        self._last_step_joint_pos.copy_(self.robot.data.joint_pos[:, self._arm_dof_idx])
        self._last_step_joint_vel.copy_(self.robot.data.joint_vel[:, self._arm_dof_idx])
        self._last_step_joint_pos_target.copy_(self.joint_pos_target)
        self._last_step_disk_pos_w.copy_(self._disk_pos_w)
        self._last_step_disk_pos_local.copy_(self._disk_pos_w).sub_(self.scene.env_origins)
        self._last_step_ball_pos_w.copy_(self._ball_pos_w)
        self._last_step_ball_pos_local.copy_(self._ball_pos_w).sub_(self.scene.env_origins)
        self._last_step_ball_vel_w.copy_(self._ball_vel_w)
        self._last_step_valid[:] = True

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
        self._update_ball_respawn_delays()
        self.prev_actions.copy_(self.actions)
        self.raw_actions.copy_(actions)
        self.actions.copy_(torch.clamp(actions, -1.0, 1.0))

        reference_q = self._reference_joint_pos_target.copy_(self.joint_pos_target)
        max_delta_q = self._joint_velocity_safe * self._step_dt
        desired_delta_q = torch.clamp(self.actions * max_delta_q, -max_delta_q, max_delta_q)
        desired_cmd_vel = desired_delta_q / self._step_dt
        distance_to_lower = torch.clamp(reference_q - self._joint_position_lower, min=0.0)
        distance_to_upper = torch.clamp(self._joint_position_upper - reference_q, min=0.0)
        accel_step = self._joint_acceleration_safe * self._step_dt
        max_negative_stop_vel = torch.clamp(
            -accel_step + torch.sqrt(accel_step**2 + 2.0 * self._joint_acceleration_safe * distance_to_lower),
            min=0.0,
        )
        max_positive_stop_vel = torch.clamp(
            -accel_step + torch.sqrt(accel_step**2 + 2.0 * self._joint_acceleration_safe * distance_to_upper),
            min=0.0,
        )
        desired_cmd_vel = torch.clamp(desired_cmd_vel, -max_negative_stop_vel, max_positive_stop_vel)

        max_delta_v = accel_step
        cmd_vel_delta = torch.clamp(
            desired_cmd_vel - self._previous_joint_cmd_vel,
            -max_delta_v,
            max_delta_v,
        )
        cmd_vel = torch.clamp(
            self._previous_joint_cmd_vel + cmd_vel_delta,
            -self._joint_velocity_safe,
            self._joint_velocity_safe,
        )

        self.joint_pos_target.copy_(reference_q + cmd_vel * self._step_dt)
        self.joint_pos_target.copy_(
            torch.clamp(self.joint_pos_target, self._joint_position_lower, self._joint_position_upper)
        )
        self._previous_joint_cmd_vel.copy_((self.joint_pos_target - reference_q) / self._step_dt)

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

        # The policy sees a delayed + noisy ball; direction/distance are derived
        # from that perceived ball and the true disk pose (as on the real robot).
        ball_pos_obs, ball_vel_obs = self._sample_ball_observation()
        direction_obs = ball_pos_obs - self._disk_pos_local
        distance_obs = torch.linalg.vector_norm(direction_obs, dim=-1, keepdim=True)

        obs = torch.cat([
            joint_pos,                                      # 6
            joint_vel,                                      # 6
            self._disk_pos_local,                           # 3
            ball_pos_obs,                                   # 3
            direction_obs,                                  # 3
            distance_obs,                                   # 1
            ball_vel_obs,                                   # 3
            (self._prev_disk_signed_dist > 0.0).float().unsqueeze(-1),  # 1
            self.actions,                                   # 6
            self.pass_through_count.unsqueeze(-1),          # 1
        ], dim=-1)

        if self.disk_center_marker is not None:
            self.disk_center_marker.visualize(self._disk_pos_w)

        return {"policy": obs}

    def _sample_ball_observation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Delayed and noisy view of the ball, used only by the policy observation."""
        self._ball_obs_head = (self._ball_obs_head + 1) % self._ball_obs_buf_len
        self._ball_obs_pos_buf[self._ball_obs_head].copy_(self._ball_pos_local)
        self._ball_obs_vel_buf[self._ball_obs_head].copy_(self._ball_vel_w)

        read_idx = (self._ball_obs_head - self._ball_obs_delay_steps) % self._ball_obs_buf_len
        ball_pos_obs = self._ball_obs_pos_buf[read_idx, self._env_index]
        ball_vel_obs = self._ball_obs_vel_buf[read_idx, self._env_index]

        pos_noise_std = float(self.cfg.ball_obs_position_noise_std)
        vel_noise_std = float(self.cfg.ball_obs_velocity_noise_std)
        if pos_noise_std > 0.0:
            ball_pos_obs = ball_pos_obs + torch.randn_like(ball_pos_obs) * pos_noise_std
        if vel_noise_std > 0.0:
            ball_vel_obs = ball_vel_obs + torch.randn_like(ball_vel_obs) * vel_noise_std
        return ball_pos_obs, ball_vel_obs

    def _reset_ball_observation_idx(self, env_ids: torch.Tensor) -> None:
        """Refill the delay buffers with the fresh ball state and resample per-env delays."""
        if self._ball_obs_delay_max > self._ball_obs_delay_min:
            self._ball_obs_delay_steps[env_ids] = torch.randint(
                self._ball_obs_delay_min,
                self._ball_obs_delay_max + 1,
                (len(env_ids),),
                device=self.device,
            )
        else:
            self._ball_obs_delay_steps[env_ids] = self._ball_obs_delay_min

        ball_pos_local = self._ball_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self._ball_obs_pos_buf[:, env_ids] = ball_pos_local.unsqueeze(0)
        self._ball_obs_vel_buf[:, env_ids] = self._ball_vel_w[env_ids].unsqueeze(0)

    def _get_rewards(self) -> torch.Tensor:
        self._update_local_pose_tensors()
        self._delta_d.copy_(self._prev_d).sub_(self._distance[:, 0])
        self._prev_d.copy_(self._distance[:, 0])

        warmup_progress = self._get_action_penalty_warmup_progress()
        action_penalty_coeff = self._get_action_penalty_coeff(warmup_progress)
        return compute_rewards(
            self._distance[:, 0],
            self.actions,
            self.raw_actions,
            self.prev_actions,
            self.reset_terminated,
            self._passed_this_step,
            self._delta_d,
            action_penalty_coeff,
            float(self.cfg.action_saturation_penalty_coeff),
            float(self.cfg.action_smoothness_penalty_coeff) * warmup_progress,
        )

    def _get_action_penalty_warmup_progress(self) -> float:
        warmup_steps = int(self.cfg.action_penalty_warmup_steps)
        if warmup_steps <= 0:
            return 1.0
        raw_progress = float(getattr(self, "common_step_counter", 0)) / float(warmup_steps)
        progress = min(max(raw_progress, 0.0), 1.0)
        return progress * progress * (3.0 - 2.0 * progress)

    def _get_action_penalty_coeff(self, smooth_progress: float) -> torch.Tensor:
        if self.cfg.use_per_joint_action_penalty:
            coeff_start = self._joint_action_penalty_coeff_ranges[:, 0]
            coeff_end = self._joint_action_penalty_coeff_ranges[:, 1]
            self._action_penalty_coeff.copy_(coeff_start + (coeff_end - coeff_start) * smooth_progress)
        else:
            coeff_start, coeff_end = self.cfg.action_penalty_coeff_range
            coeff = float(coeff_start) + (float(coeff_end) - float(coeff_start)) * smooth_progress
            self._action_penalty_coeff.fill_(coeff)

        return self._action_penalty_coeff

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_cached_poses()
        self._cache_last_step_state()

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
        passed &= ~self._ball_respawn_pending
        self._passed_this_step.copy_(passed)
        self._prev_disk_signed_dist.copy_(signed_dist)
        self.pass_through_count += passed.float()
        self.episode_success |= passed

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        ball_on_ground = (self._ball_pos_w[:, 2] < 0.05) & ~self._ball_respawn_pending
        is_caught = self.pass_through_count > 0.0
        hit_arm = self._contact_on_arm
        success_done = is_caught if self.cfg.reset_on_success else self._zero_done
        terminated = success_done | hit_arm | ball_on_ground
        done = terminated | time_out
        self._last_done_success[:] = False
        self._last_done_success[done] = self.episode_success[done]
        self.extras["success"] = self._last_done_success

        continuing_success = passed & ~done
        if self.cfg.reset_ball_on_success and not self.cfg.reset_on_success and continuing_success.any():
            self._queue_ball_respawn_idx(continuing_success.nonzero(as_tuple=False).squeeze(-1))

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        if self.cfg.reset_robot_on_episode_reset:
            self._reset_robot_idx(env_ids)
            self._robot_pose_initialized[env_ids] = True
        else:
            already_initialized = self._robot_pose_initialized[env_ids]
            initial_env_ids = env_ids[~already_initialized]
            if len(initial_env_ids) > 0:
                self._reset_robot_idx(initial_env_ids)
                self._robot_pose_initialized[initial_env_ids] = True
            if already_initialized.any():
                self._sync_robot_targets_to_current_pose(env_ids[already_initialized])

        self._reset_ball_idx(env_ids, reset_episode_tracking=True)

    def _reset_robot_idx(self, env_ids: Sequence[int]) -> None:
        joint_pos = torch.zeros((len(env_ids), len(self._arm_dof_idx)), device=self.device)
        joint_pos[:, 0] = sample_uniform(-0.785, 0.785, len(env_ids), self.device)
        joint_pos[:, 1] = -1.57 + sample_uniform(-0.9, 0.2, len(env_ids), self.device)
        joint_pos[:, 2] = sample_uniform(-0.2, 0.2, len(env_ids), self.device)
        joint_pos[:, 3] = -1.57 + sample_uniform(-0.2, 0.2, len(env_ids), self.device)
        joint_pos[:, 4] = sample_uniform(-0.2, 0.2, len(env_ids), self.device)
        joint_pos[:, 5] = sample_uniform(-0.2, 0.2, len(env_ids), self.device)
        joint_vel = torch.zeros_like(joint_pos)

        self.actions[env_ids] = 0.0
        self.raw_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.joint_pos_target[env_ids] = joint_pos
        self._previous_joint_cmd_vel[env_ids] = 0.0
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids, joint_ids=self._arm_dof_idx)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self._arm_dof_idx, env_ids=env_ids)

    def _sync_robot_targets_to_current_pose(self, env_ids: Sequence[int]) -> None:
        joint_pos = self.robot.data.joint_pos[env_ids][:, self._arm_dof_idx].clone()
        self.actions[env_ids] = 0.0
        self.raw_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.joint_pos_target[env_ids] = joint_pos
        self._previous_joint_cmd_vel[env_ids] = 0.0
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids, joint_ids=self._arm_dof_idx)

    def _maybe_random_reset_robot_on_ball_reset(self, env_ids: Sequence[int]) -> None:
        if self.cfg.reset_robot_on_episode_reset or not self.cfg.random_robot_reset_on_ball_reset:
            return

        reset_probability = min(max(float(self.cfg.random_robot_reset_on_ball_reset_probability), 0.0), 1.0)
        if reset_probability <= 0.0:
            return

        reset_mask = torch.rand(len(env_ids), device=self.device) < reset_probability
        if reset_mask.any():
            reset_env_ids = env_ids[reset_mask]
            self._reset_robot_idx(reset_env_ids)
            self._robot_pose_initialized[reset_env_ids] = True

    def _queue_ball_respawn_idx(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        delay_steps = self._sample_ball_respawn_delay_steps(env_ids)
        immediate_mask = delay_steps <= 0
        if immediate_mask.any():
            self._reset_ball_idx(env_ids[immediate_mask], reset_episode_tracking=False)

        delayed_mask = ~immediate_mask
        if delayed_mask.any():
            delayed_env_ids = env_ids[delayed_mask]
            self._ball_respawn_pending[delayed_env_ids] = True
            self._ball_respawn_delay_steps[delayed_env_ids] = delay_steps[delayed_mask]
            self._park_ball_idx(delayed_env_ids)

    def _sample_ball_respawn_delay_steps(self, env_ids: Sequence[int]) -> torch.Tensor:
        min_delay_s, max_delay_s = self.cfg.ball_respawn_delay_s_range
        min_delay_s = max(0.0, float(min_delay_s))
        max_delay_s = max(min_delay_s, float(max_delay_s))
        if max_delay_s <= 0.0:
            return torch.zeros(len(env_ids), dtype=torch.long, device=self.device)

        delay_s = sample_uniform(min_delay_s, max_delay_s, len(env_ids), self.device)
        return torch.ceil(delay_s / self._step_dt).to(dtype=torch.long)

    def _update_ball_respawn_delays(self) -> None:
        if not self._ball_respawn_pending.any():
            return

        pending_env_ids = self._ball_respawn_pending.nonzero(as_tuple=False).squeeze(-1)
        self._ball_respawn_delay_steps[pending_env_ids] -= 1

        due_mask = self._ball_respawn_delay_steps[pending_env_ids] <= 0
        if due_mask.any():
            self._reset_ball_idx(pending_env_ids[due_mask], reset_episode_tracking=False)

        still_pending_env_ids = self._ball_respawn_pending.nonzero(as_tuple=False).squeeze(-1)
        if len(still_pending_env_ids) > 0:
            self._park_ball_idx(still_pending_env_ids)

    def _park_ball_idx(self, env_ids: Sequence[int]) -> None:
        self._update_cached_poses(env_ids)

        ball_state = self.ball.data.default_root_state[env_ids].clone()
        ball_state[:, :] = 0.0
        if self.cfg.ball_respawn_hold_at_disk_center:
            ball_state[:, 0:3] = self._disk_pos_w[env_ids]
        else:
            ball_state[:, 0:3] = self.scene.env_origins[env_ids]
        ball_state[:, 3:7] = self._identity_quat_wxyz

        self.ball.write_root_state_to_sim(ball_state, env_ids=env_ids)
        self._update_cached_poses(env_ids)
        self._local_pose_cache_valid = False

        self._prev_d[env_ids] = torch.norm(self._ball_pos_w[env_ids] - self._disk_pos_w[env_ids], dim=-1)
        ball_relative = self._ball_pos_w[env_ids] - self._disk_pos_w[env_ids]
        self._prev_disk_signed_dist[env_ids] = (ball_relative * self._disk_normal_w[env_ids]).sum(dim=-1)

    def _reset_ball_idx(self, env_ids: Sequence[int], reset_episode_tracking: bool) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._maybe_random_reset_robot_on_ball_reset(env_ids)
        self._ball_respawn_pending[env_ids] = False
        self._ball_respawn_delay_steps[env_ids] = 0
        self.ball.reset(env_ids)
        ball_state = self.ball.data.default_root_state[env_ids].clone()
        ball_state[:, :] = 0.0

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
        self._reset_ball_observation_idx(env_ids)

        # Initialize _prev_d with the real distance at reset.
        self._prev_d[env_ids] = torch.norm(self._ball_pos_w[env_ids] - self._disk_pos_w[env_ids], dim=-1)

        # Reset tracking
        ball_relative = self._ball_pos_w[env_ids] - self._disk_pos_w[env_ids]
        self._prev_disk_signed_dist[env_ids] = (ball_relative * self._disk_normal_w[env_ids]).sum(dim=-1)
        if reset_episode_tracking:
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
    raw_actions: torch.Tensor,
    prev_actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    passed: torch.Tensor,
    delta_d: torch.Tensor,
    action_penalty_coeff: torch.Tensor,
    saturation_penalty_coeff: float,
    smoothness_penalty_coeff: float,
) -> torch.Tensor:

    rew_dist = (torch.exp(-2.0 * distance) - 1.0 * distance)
    rew_action = -torch.sum(action_penalty_coeff.unsqueeze(0) * actions ** 2, dim=-1)
    # Boundary penalty on the raw (pre-clip) action: unlike the clipped-action
    # penalty above, it stays differentiated when the policy mean saturates
    # past ±1 and is the term that can pull saturated means back toward zero.
    raw_excess = torch.relu(raw_actions.abs() - 1.0)
    rew_saturation = -saturation_penalty_coeff * torch.sum(raw_excess ** 2, dim=-1)
    rew_smooth = -smoothness_penalty_coeff * torch.sum((actions - prev_actions) ** 2, dim=-1)
    rew_pass = 400.0 * passed.to(dtype=torch.float32)
    rew_termination = -100.0 * reset_terminated.to(dtype=torch.float32)
    return rew_action + rew_saturation + rew_smooth + rew_pass + rew_termination + rew_dist
