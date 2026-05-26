import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .ur_gripper import UR3E_HOOP_CFG


@configclass
class FirsttrainingEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 4.0
    action_space = 6
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
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    robot_usd_root_name = "ur3e"
    disk_link_name = "wrist_3_link"
    disk_prim_rel_path = "ur3e/wrist_3_link/Hoop/node_/Disk"
    disk_local_normal_axis = (0.0, 0.0, 1.0)
    disk_radius = 0.1  # <= 0 means: read the trigger radius from the Disk mesh.
    ball_spawn_x_range = (-0.65, -0.35)
    ball_spawn_y_range = (1.0, 1.7)
    ball_spawn_z_range = (0.7, 0.95)
    enable_ball_position_noise: bool = True
    ball_position_noise_std = 0.05
    ball_velocity_x_range = (-0.8, 0.4)
    ball_velocity_y_range = (-10.0, -2.5)
    ball_velocity_z_range = (-0.25, 0.3)
    action_scale = 0.5
    enable_markers: bool = False
    enable_disk_center_marker: bool = False
    disk_center_marker_radius = 0.025
    reset_on_success: bool = True
