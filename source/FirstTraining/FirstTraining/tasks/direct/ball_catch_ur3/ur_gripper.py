"""Configuration for the UR3e hoop robot."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


def _find_usd_asset(filename: str) -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "USD_File" / filename
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Unable to find USD_File/{filename} from {__file__}")


UR3E_SHORT_HOOP_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_find_usd_asset("UR-with-gripper-config-2.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.74,
            "elbow_joint": 1.74,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            effort_limit_sim=23.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)