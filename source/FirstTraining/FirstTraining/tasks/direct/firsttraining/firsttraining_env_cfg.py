import math

import numpy as np
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .ur_gripper import UR3E_HOOP_CFG


UR3E_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@configclass
class FirsttrainingEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 4.0
    action_space = spaces.Box(
        low=np.full(6, -1.0, dtype=np.float32),
        high=np.full(6, 1.0, dtype=np.float32),
        dtype=np.float32,
    )
    observation_space = 33
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.03,  # small ball relative to hoop
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.3, 0.0),  # orange ball
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 1.2, 0.75),
        ),
    )

    robot_cfg: ArticulationCfg = UR3E_HOOP_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor_cfg_arm: ContactSensorCfg = ContactSensorCfg(
        prim_path=(
            "/World/envs/env_.*/Robot/ur3e/"
            "(base_link|shoulder_link|upper_arm_link|forearm_link|wrist_1_link|wrist_2_link)"
        ),
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=4.0,
        replicate_physics=True,
    )
    joint_names = UR3E_ARM_JOINTS
    joint_velocity_safe_rad_s = (3.1416, 3.1416, 3.1416, 6.2832, 6.2832, 6.2832)
    joint_acceleration_safe_rad_s2 = (12.5664, 12.5664, 12.5664, 25.1328, 25.1328, 25.1328)
    joint_position_lower_rad = (
        -2.0 * math.pi,
        -2.0 * math.pi,
        -math.pi,
        -2.0 * math.pi,
        -2.0 * math.pi,
        -2.0 * math.pi,
    )
    joint_position_upper_rad = (
        2.0 * math.pi,
        2.0 * math.pi,
        math.pi,
        2.0 * math.pi,
        2.0 * math.pi,
        2.0 * math.pi,
    )
    robot_usd_root_name = "ur3e"
    disk_link_name = "wrist_3_link"
    disk_prim_rel_path = "ur3e/wrist_3_link/Hoop/node_/Disk"
    disk_local_normal_axis = (0.0, 0.0, 1.0)
    disk_radius = 0.1
    ball_spawn_x_range = (-0.6, -0.2)
    ball_spawn_y_range = (1.2, 2.1)
    ball_spawn_z_range = (0.5, 1.2)
    enable_ball_position_noise: bool = True
    ball_position_noise_std = 0.01
    ball_velocity_x_range = (-0.7, 0.6)
    ball_velocity_y_range = (-5.0, -3.5)
    ball_velocity_z_range = (-0.1, 1.5)
    ball_respawn_delay_s_range = (0.0, 0.0)
    ball_respawn_hold_at_disk_center: bool = True
    use_per_joint_action_penalty: bool = True
    action_penalty_coeff_range = (0.6, 1.8)
    joint_action_penalty_coeff_ranges = (
        (0.85, 2.55),  # shoulder_pan_joint
        (0.95, 2.85),  # shoulder_lift_joint
        (0.30, 1.0),  # elbow_joint
        (0.45, 1.35),  # wrist_1_joint
        (0.35, 1.05),  # wrist_2_joint
        (0.25, 0.75),  # wrist_3_joint
    )
    action_penalty_warmup_steps = 150_000
    action_scale = 1.0
    action_semantics = (
        "incremental joint target integrator: previous joint_position_target_rad + "
        "clamp(action_normalized, -1, 1) * joint_velocity_safe_rad_s * dt_s, "
        "then acceleration- and joint-limit-clamped"
    )
    enable_markers: bool = False
    enable_disk_center_marker: bool = False
    disk_center_marker_radius = 0.025
    reset_on_success: bool = False
    reset_ball_on_success: bool = True
    reset_robot_on_episode_reset: bool = False
    random_robot_reset_on_ball_reset: bool = True
    random_robot_reset_on_ball_reset_probability = 0.05
