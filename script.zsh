#!/bin/zsh
# pip install -e source/FirstTraining  # Si Import fonctionne pas

CHECKPOINT=$(ls -dt logs/skrl/cartpole_direct/*/checkpoints/best_agent.pt 2>/dev/null | head -1)

# Safety check — warn if no checkpoint found
source ~/env_isaaclab/bin/activate
if [[ -z "$CHECKPOINT" ]]; then
  echo "[script.zsh] Warning: no checkpoint found, lunch_play and lunch_recording may fail"
else
  echo "[script.zsh] Checkpoint: $CHECKPOINT"
fi

alias train="HEADLESS=1 LIVESTREAM=0 ENABLE_CAMERAS=0 python scripts/skrl/train.py \
  --task Template-Firsttraining-Direct-v0 \
  --num_envs=12000 \
  --headless \
  --livestream 0 \
  --rendering_mode performance \
"
# --seed=42

alias play="python scripts/skrl/play.py \
  --task Template-Firsttraining-Direct-v0 \
  --num_envs=32 \
  --checkpoint=$CHECKPOINT \
"

alias evaluate="HEADLESS=1 LIVESTREAM=0 ENABLE_CAMERAS=0 python scripts/skrl/play.py \
  --task Template-Firsttraining-Direct-v0 \
  --num_envs=512 \
  --checkpoint=$CHECKPOINT \
  --headless \
  --livestream 0 \
  --rendering_mode performance \
  --eval_episodes=200000 \
"

alias record="python scripts/skrl/play.py \
  --task Template-Firsttraining-Direct-v0 \
  --num_envs=32 \
  --checkpoint=$CHECKPOINT \
  --video \
  --video_length=20 \
  --headless \
"

alias tensorboard="python -m tensorboard.main --logdir logs/skrl"

alias ur3e_stack="./scripts/launch_ur3e_stack.sh"
alias ur3e_stop="./scripts/launch_ur3e_stack.sh --stop"

# alias lunch_play_gpu="... --device cuda"  # future multi-gpu
