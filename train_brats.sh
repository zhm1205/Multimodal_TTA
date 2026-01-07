#!/bin/bash
# BraTS-GLI 分割模型训练脚本 (使用 UNet3D)
# 直接读取 NIfTI 格式，无需预处理转换

python main.py \
  task=brats \
  task.run_name=brats_unet \
  dataset=brats \
  model=unet \
  training=default \
  training.epochs=100 \
  training.batch_size=8 \
  training.eval_batch_size=8 \
  training.num_workers=8 \
  training.gpu_ids=[6] \
  training.model_save_start=0 \
  training.model_save_freq=10 \
  training.optimizer=adam \
  training.optimizers.adam.lr=1e-3
