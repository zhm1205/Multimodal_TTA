#!/bin/bash
# BraTS-GLI 分割模型训练脚本 (使用 UNet3D)
# 直接读取 NIfTI 格式，无需预处理转换

python main.py \
  task=hecktor21 \
  task.run_name=hecktor21_unet \
  dataset=hecktor21 \
  model=unet \
  training=default \
  training.epochs=200 \
  training.batch_size=8 \
  training.eval_batch_size=16 \
  training.num_workers=8 \
  training.gpu_ids=[0] \
  training.model_save_start=0 \
  training.model_save_freq=10 \
  training.optimizer=adam \
  training.optimizers.adam.lr=3e-4 \
  training.eval_on_train=true
