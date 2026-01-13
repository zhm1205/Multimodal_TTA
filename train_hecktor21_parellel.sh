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

SAVE_START=0
SAVE_FREQ=10

OPT="adam"
LR="5e-3"
EVAL_ON_TRAIN="true"

# =========================
# Targets (edit to your 5 centers)
# =========================
TARGET_CENTERS=("CHUS" "CHUM" "CHGJ" "CHMR" "CHUP")
RUN_PREFIX="hecktor21_unet"

# =========================
# GPU pool (edit here)
# e.g. ("0" "1" "2") means at most 3 jobs in parallel
# =========================
GPU_POOL=("1" "2" "3")

# ---- kill all children on exit / Ctrl+C / SIGTERM ----
cleanup() {
  echo "[CLEANUP] killing all launched jobs..."
  # 1) kill background pids we started (if any)
  if [[ "${#RUN_PIDS[@]:-0}" -gt 0 ]]; then
    kill "${RUN_PIDS[@]}" 2>/dev/null || true
    # 给一点时间让它们优雅退出
    sleep 2
    # 2) still alive? force kill
    kill -9 "${RUN_PIDS[@]}" 2>/dev/null || true
  fi
  echo "[CLEANUP] done."
}
trap cleanup INT TERM EXIT

# =========================
# Internal state
# =========================
declare -A PID2GPU=()
declare -a RUN_PIDS=()

# Start one job on a given GPU (background)
start_job () {
  local target="$1"
  local gpu="$2"

  local run_name="${RUN_PREFIX}_target-${target}"
  local gpu_ids="[${gpu}]"

  echo "------------------------------------------------------------"
  echo "Launching: target_center=${target} on GPU=${gpu} | run_name=${run_name}"
  echo "------------------------------------------------------------"

  # 可选：把 stdout/stderr 单独落盘，便于排查
  local log_file="run_${run_name}.log"

  python main.py \
    task="${TASK}" \
    task.run_name="${run_name}" \
    dataset=hecktor21 \
    dataset.target_center="${target}" \
    model="${MODEL}" \
    training="${TRAINING}" \
    training.epochs="${EPOCHS}" \
    training.batch_size="${BS}" \
    training.eval_batch_size="${EVAL_BS}" \
    training.num_workers="${NUM_WORKERS}" \
    training.gpu_ids="${gpu_ids}" \
    training.model_save_start="${SAVE_START}" \
    training.model_save_freq="${SAVE_FREQ}" \
    training.optimizer="${OPT}" \
    training.optimizers.adam.lr="${LR}" \
    training.eval_on_train="${EVAL_ON_TRAIN}" \
    > "${log_file}" 2>&1 &

  local pid=$!
  PID2GPU["$pid"]="$gpu"
  RUN_PIDS+=("$pid")

  echo "Started PID=${pid} (GPU=${gpu}) log=${log_file}"
}

# Wait for any running job to finish, then free its GPU
wait_any_and_free_gpu () {
  # bash 5+ 支持 wait -n；大多数 Linux 集群都有
  local finished_pid
  if finished_pid=$(wait -n "${RUN_PIDS[@]}" 2>/dev/null; echo $?); then
    : # 这里 finished_pid 实际拿不到 pid（wait -n 返回退出码），所以下面用轮询清理
  fi

  # 清理已结束的 PID，并回收 GPU
  local new_pids=()
  local freed_gpu=""
  for pid in "${RUN_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
    else
      # pid 已结束
      freed_gpu="${PID2GPU[$pid]:-}"
      unset "PID2GPU[$pid]"
    fi
  done
  RUN_PIDS=("${new_pids[@]}")

  if [[ -z "${freed_gpu}" ]]; then
    # 极少数情况下（极快退出）可能没捕捉到，稍等再清一次
    sleep 1
    wait_any_and_free_gpu
    return
  fi

  echo "Job finished, GPU freed: ${freed_gpu}"
  echo "${freed_gpu}"
}

# Get an available GPU if any (simple: available = GPU_POOL - GPUs currently in use)
get_free_gpu () {
  local used=()
  for pid in "${RUN_PIDS[@]}"; do
    used+=("${PID2GPU[$pid]}")
  done

  for gpu in "${GPU_POOL[@]}"; do
    local in_use="false"
    for ug in "${used[@]}"; do
      if [[ "$ug" == "$gpu" ]]; then
        in_use="true"
        break
      fi
    done
    if [[ "$in_use" == "false" ]]; then
      echo "$gpu"
      return
    fi
  done

  echo ""  # none
}

# =========================
# Main scheduler loop
# =========================
for target in "${TARGET_CENTERS[@]}"; do
  while true; do
    free_gpu="$(get_free_gpu)"
    if [[ -n "${free_gpu}" ]]; then
      start_job "${target}" "${free_gpu}"
      break
    fi
    # 没空 GPU -> 等一个任务结束
    wait_any_and_free_gpu >/dev/null
  done
done

# Wait for all remaining jobs
echo "All jobs launched. Waiting for completion..."
wait
echo "All jobs completed."