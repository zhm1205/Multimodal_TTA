#!/usr/bin/env bash
set -euo pipefail

# =========================
# Common config
# =========================
TASK="brats"
MODEL="unet"
TRAINING="default"

EPOCHS=100
BS=16
EVAL_BS=16
NUM_WORKERS=8
GPU_IDS="[6, 7]"

SAVE_START=0
SAVE_FREQ=10

OPT="adam"
LR="1e-3"

EVAL_ON_TRAIN="true"

# =========================
# Targets (edit to your 5 centers)
# =========================
TARGET_CENTERS=("brats25_glipre" "brats23_ssa" "brats24_ped")

# Optional: a prefix for run_name
RUN_PREFIX="brats_unet"

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
    dataset=brats \
    dataset.target_domain="${TARGET}" \
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

