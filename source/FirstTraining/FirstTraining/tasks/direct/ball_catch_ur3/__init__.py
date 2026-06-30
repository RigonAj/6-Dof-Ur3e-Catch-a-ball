# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from .ur_gripper import *
from . import agents


##
# Register Gym environments.
##


gym.register(
    id="Ball-Catch-UR3-Direct-v0",
    entry_point=f"{__name__}.ball_catch_env:BallCatchUR3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:BallCatchUR3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)