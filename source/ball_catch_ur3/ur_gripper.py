"""Configuration for the UR3e hoop robot."""

from pathlib import Path

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg


def _find_usd_asset(filename: str) -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "USD_File" / filename
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Unable to find USD_File/{filename} from {__file__}")


UR3E_HOOP_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_find_usd_asset("UR-with-grippers.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 0.0,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": 0.0,
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
            effort_limit=23.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

# Backward-compatible name used by older configs/imports.
"""UR3e robot with the a more short hoop attached to the wrist.
    And the position default more suitable for training...."""
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
            effort_limit=23.0,
            stiffness=800.0,
            damping=40.0,
            #velocity_limit=3.14,
        ),
    },
)
UR3E_GRIPPER_CFG_HIGH_PD_CFG = UR3E_SHORT_HOOP_CFG.copy()
UR3E_GRIPPER_CFG_HIGH_PD_CFG.actuators["arm"].stiffness = 100000.0
UR3E_GRIPPER_CFG_HIGH_PD_CFG.actuators["arm"].damping = 40.0
