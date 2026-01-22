#!/usr/bin/env bash
set -euo pipefail

# =========================
# Common config
# =========================
TASK="hecktor21"
MODEL="unet"
TRAINING="default"

EPOCHS=300
BS=8
EVAL_BS=16
NUM_WORKERS=8
GPU_IDS="[1]"

SAVE_START=0
SAVE_FREQ=10

OPT="adam"
LR="5e-3"

EVAL_ON_TRAIN="true"

# =========================
# Targets (edit to your 5 centers)
# =========================
TARGET_CENTERS=("CHUS" "CHUM" "CHGJ" "CHMR" "CHUP")

# Optional: a prefix for run_name
RUN_PREFIX="hecktor21_unet"

# =========================
# Run
# =========================
for TARGET in "${TARGET_CENTERS[@]}"; do
  RUN_NAME="${RUN_PREFIX}_target-${TARGET}"

  echo "============================================================"
  echo "Running target_center=${TARGET} | run_name=${RUN_NAME}"
  echo "============================================================"

  python main.py \
    task="${TASK}" \
    task.run_name="${RUN_NAME}" \
    dataset=hecktor21 \
    dataset.target_center="${TARGET}" \
    model="${MODEL}" \
    training="${TRAINING}" \
    training.epochs="${EPOCHS}" \
    training.batch_size="${BS}" \
    training.eval_batch_size="${EVAL_BS}" \
    training.num_workers="${NUM_WORKERS}" \
    training.gpu_ids="${GPU_IDS}" \
    training.model_save_start="${SAVE_START}" \
    training.model_save_freq="${SAVE_FREQ}" \
    training.optimizer="${OPT}" \
    training.optimizers.adam.lr="${LR}" \
    training.eval_on_train="${EVAL_ON_TRAIN}"
done

