#!/bin/bash
# === FULL RUN: 9-epoch fine-tune from epoch-41 baseline ===
# Baseline checkpoint: best_acc_top1_epoch_41.pth (weights-only, cannot resume)
# Strategy: load_from + 9 epochs fresh training with adjusted scheduler
# GPU: GESTURE_GPU_DEVICES=1 (GPU 0 occupied by external PID 1880123/1880124)
set -euo pipefail

export PYTHONPATH="/workspace/rightArmPerception/training/mmaction2:${PYTHONPATH:-}"

TS="20260722_115644"
WORK_DIR="/home/results/6t1/openhands_gesture/${TS}-full"
CONFIG="handoff/stgcn_hand21_3d/configs/stgcn_hand21-rgb-bs64-gated-v3-ft-rtm_add-11cls.py"
LOAD_FROM="/home/results/6t1/stgcn_hand21_rgb_gated_class11_v3_ft_rtm_add/best_acc_top1_epoch_41.pth"

mkdir -p "$WORK_DIR"

echo "=== FULL RUN START $(date -Iseconds) ==="
echo "Work dir:   $WORK_DIR"
echo "Config:     $CONFIG"
echo "Load from:  $LOAD_FROM (epoch 41, weights-only → load_from, NOT resume)"
echo "Epochs:     9 (fresh schedule)"
echo "Batch size: 32"
echo "Workers:    8"

# Step 1: save resolved config
python3 -c "
from mmengine import Config
cfg = Config.fromfile('${CONFIG}')
cfg.load_from = '${LOAD_FROM}'
cfg.resume = False
cfg.train_cfg.max_epochs = 9
cfg.train_dataloader.batch_size = 32
cfg.train_dataloader.num_workers = 8
cfg.val_dataloader.batch_size = 32
cfg.val_dataloader.num_workers = 8
cfg.param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=2, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=0, T_max=7, by_epoch=True, begin=2, end=9, convert_to_iter_based=True),
]
cfg.dump('${WORK_DIR}/resolved_config.py')
print('Resolved config saved to ${WORK_DIR}/resolved_config.py')
"

echo ""
echo "=== LAUNCHING 9-EPOCH FULL RUN ==="
# Use resolved config directly (--cfg-options cannot handle nested dicts like param_scheduler)
python3 tools/train.py \
  "${WORK_DIR}/resolved_config.py" \
  --work-dir "$WORK_DIR"

RC=$?
echo ""
echo "=== FULL RUN END $(date -Iseconds) exit=$RC ==="
exit $RC
