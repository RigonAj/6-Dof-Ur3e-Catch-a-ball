# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Firsttraining-Direct-v0",
    entry_point=f"{__name__}.firsttraining_env:FirsttrainingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.firsttraining_env_cfg:FirsttrainingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Left-hand variant: racket rotated 180 deg about wrist_3 Z, ball distribution
# mirrored across the yz plane (x -> -x). Same observation/action contract.
gym.register(
    id="Template-Firsttraining-Direct-Left-v0",
    entry_point=f"{__name__}.firsttraining_env:FirsttrainingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.firsttraining_env_cfg:FirsttrainingEnvCfgLeft",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_left.yaml",
    },
)