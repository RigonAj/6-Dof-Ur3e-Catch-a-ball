#!/bin/zsh
# pip install -e source/FirstTraining  # Si Import fonctionne pas

source ~/env_isaaclab/bin/activate

unalias train play play_latest play_best train-left train-right play-left play-right evaluate record sim2real_export sim2real_validate tensorboard checkpoint latest_checkpoint best_checkpoint 2>/dev/null || true

# Task selection: FT_TASK=Template-Firsttraining-Direct-Left-v0 train
# switches every command (train/play/evaluate/record/sim2real_export) to the
# left-hand variant. Checkpoints of the Left task land in a separate log root
# unless CHECKPOINT_ROOT is set explicitly.
_ft_task() {
  echo "${FT_TASK:-Template-Firsttraining-Direct-v0}"
}

_checkpoint_root() {
  if [[ -n "$CHECKPOINT_ROOT" ]]; then
    echo "$CHECKPOINT_ROOT"
  elif [[ "$(_ft_task)" == *-Left-* ]]; then
    echo "logs/skrl/cartpole_direct_left"
  else
    echo "logs/skrl/cartpole_direct"
  fi
}

latest_checkpoint() {
  local checkpoint
  checkpoint=$(find "$(_checkpoint_root)" \( -path '*/checkpoints/best_agent.pt' -o -path '*/checkpoints/agent_*.pt' \) -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print; exit}')
  echo "$checkpoint"
}

best_checkpoint() {
  local checkpoint
  checkpoint=$(find "$(_checkpoint_root)" -path '*/checkpoints/best_agent.pt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print; exit}')
  if [[ -z "$checkpoint" ]]; then
    checkpoint=$(latest_checkpoint)
  fi
  echo "$checkpoint"
}

latest_export_dir() {
  find "$(_checkpoint_root)" -path '*/exports/policy_metadata.json' -printf '%T@ %h\n' 2>/dev/null | sort -nr | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print; exit}'
}

_resolve_checkpoint() {
  local selector="${1:-latest}"
  case "$selector" in
    latest|last)
      latest_checkpoint
      ;;
    best)
      best_checkpoint
      ;;
    *)
      echo "$selector"
      ;;
  esac
}

_uses_checkpoint_arg() {
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "--checkpoint" || "$arg" == --checkpoint=* ]]; then
      return 0
    fi
  done
  return 1
}

_first_arg_is_checkpoint_selector() {
  [[ "$1" == "latest" || "$1" == "last" || "$1" == "best" || "$1" == *.pt || -f "$1" ]]
}

_require_checkpoint_file() {
  local checkpoint="$1"
  if [[ -z "$checkpoint" ]]; then
    echo "[script.zsh] Error: no checkpoint found in $(_checkpoint_root)"
    return 1
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "[script.zsh] Error: checkpoint file not found: $checkpoint"
    return 1
  fi
  return 0
}

checkpoint() {
  local checkpoint
  checkpoint=$(_resolve_checkpoint "${1:-latest}")
  _require_checkpoint_file "$checkpoint" || return 1
  echo "$checkpoint"
}

CHECKPOINT=$(latest_checkpoint)
if [[ -z "$CHECKPOINT" ]]; then
  echo "[script.zsh] Warning: no checkpoint found in $(_checkpoint_root); play/record/evaluate may fail"
else
  echo "[script.zsh] Checkpoint: $CHECKPOINT"
fi

train() {
  MANGOHUD=0 DISABLE_MANGOHUD=1 HEADLESS=1 LIVESTREAM=0 ENABLE_CAMERAS=0 python scripts/skrl/train.py \
  --task "$(_ft_task)" \
  --num_envs=12000 \
  --headless \
  --livestream 0 \
  --rendering_mode performance \
  "$@"
}
# --seed=42

play() {
  local checkpoint
  if _uses_checkpoint_arg "$@"; then
    MANGOHUD=0 DISABLE_MANGOHUD=1 python scripts/skrl/play.py \
    --task "$(_ft_task)" \
    --num_envs=1 \
    "$@"
    return $?
  fi
  if _first_arg_is_checkpoint_selector "$1"; then
    checkpoint=$(_resolve_checkpoint "$1")
    shift
  else
    checkpoint=$(latest_checkpoint)
  fi
  _require_checkpoint_file "$checkpoint" || return 1
  echo "[script.zsh] Checkpoint: $checkpoint"
  MANGOHUD=0 DISABLE_MANGOHUD=1 python scripts/skrl/play.py \
  --task "$(_ft_task)" \
  --num_envs=1 \
  --checkpoint="$checkpoint" \
  "$@"
}

play_latest() {
  play latest "$@"
}

play_best() {
  play best "$@"
}

# Hold-side shortcuts: same commands pinned to one task variant. They also
# select the matching checkpoint root (cartpole_direct vs cartpole_direct_left).
train-left() {
  FT_TASK=Template-Firsttraining-Direct-Left-v0 train "$@"
}

train-right() {
  FT_TASK=Template-Firsttraining-Direct-v0 train "$@"
}

play-left() {
  FT_TASK=Template-Firsttraining-Direct-Left-v0 play "$@"
}

play-right() {
  FT_TASK=Template-Firsttraining-Direct-v0 play "$@"
}

evaluate() {
  local checkpoint
  checkpoint=$(latest_checkpoint)
  _require_checkpoint_file "$checkpoint" || return 1
  echo "[script.zsh] Checkpoint: $checkpoint"
  MANGOHUD=0 DISABLE_MANGOHUD=1 HEADLESS=1 LIVESTREAM=0 ENABLE_CAMERAS=0 python scripts/skrl/play.py \
  --task "$(_ft_task)" \
  --num_envs=512 \
  --checkpoint="$checkpoint" \
  --headless \
  --livestream 0 \
  --rendering_mode performance \
  --eval_episodes=200000 \
  "$@"
}

record() {
  local checkpoint
  checkpoint=$(latest_checkpoint)
  _require_checkpoint_file "$checkpoint" || return 1
  echo "[script.zsh] Checkpoint: $checkpoint"
  MANGOHUD=0 DISABLE_MANGOHUD=1 python scripts/skrl/play.py \
  --task "$(_ft_task)" \
  --num_envs=32 \
  --checkpoint="$checkpoint" \
  --video \
  --video_length=20 \
  --headless \
  "$@"
}

sim2real_export() {
  local checkpoint
  local episodes
  checkpoint=$(latest_checkpoint)
  episodes=${SIM2REAL_EPISODES:-10}
  _require_checkpoint_file "$checkpoint" || return 1
  echo "[script.zsh] Checkpoint: $checkpoint"
  echo "[script.zsh] Recording sim2real episodes: $episodes"
  MANGOHUD=0 DISABLE_MANGOHUD=1 HEADLESS=1 LIVESTREAM=0 ENABLE_CAMERAS=0 python scripts/skrl/play.py \
  --task "$(_ft_task)" \
  --num_envs=1 \
  --checkpoint="$checkpoint" \
  --headless \
  --livestream 0 \
  --rendering_mode performance \
  --export_policy \
  --export_onnx \
  --record_actions \
  --record_episodes="$episodes" \
  "$@"
}

sim2real_validate() {
  local exports_dir
  local rollout
  exports_dir=${1:-$(latest_export_dir)}
  if [[ -z "$exports_dir" ]]; then
    echo "[script.zsh] Error: no exports directory found"
    return 1
  fi
  if [[ ! -d "$exports_dir" ]]; then
    echo "[script.zsh] Error: exports directory not found: $exports_dir"
    return 1
  fi
  rollout=$(find "$exports_dir" -maxdepth 1 -name 'rollouts_*_episodes.json' | sort | tail -n 1)
  if [[ -z "$rollout" ]]; then
    echo "[script.zsh] Error: no rollout JSON found in $exports_dir"
    return 1
  fi
  python scripts/sim2real_validate_export.py --exports "$exports_dir" && \
  python scripts/sim2real_validate_rollout_safety.py --rollout "$rollout"
}

tensorboard() {
  python -m tensorboard.main --logdir logs/skrl "$@"
}

# alias launch_play_gpu="... --device cuda"  # future multi-gpu
