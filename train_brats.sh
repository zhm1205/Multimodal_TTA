# dataset/brats_tta.yaml (use BratsMultiNiftiBuilder from brats_raw)
name: brats

# 离线预处理后应统一到固定 shape
expected_shape: [160, 192, 160]
strict_label_values: false

# 多源配置：GLIPRE 作为唯一源域；SSA/PED 仅作为测试域
sources:
  # 源域：GLIPRE，用于 train/val（以及可选的源域 test）
  - name: brats25_glipre
    profile: gli
    csv_path: /home/dengzhipeng/data/brain_processed/BraTS25-GLIPRE/processed.csv
    root_dir: null
    include_splits:
      train: ["train"]        # GLIPRE: 训练集使用 CSV 中 split=train 的样本
      val:   ["test"]         # GLIPRE: 验证集使用 CSV 中 split=test 的样本
      test:  []                # GLIPRE: 源域不再参与最终 test
    region_map:
      ET: [3]
      TC: [1, 3]
      WT: [1, 2, 3]

  # 目标域：SSA，仅作为测试集（所有 split 统一视为 test）
  - name: brats23_ssa
    profile: ssa
    csv_path: /home/dengzhipeng/data/brain_processed/BraTS23-SSA/processed.csv
    root_dir: null
    include_splits:
      train: []                 # 不参与训练
      val:   []                 # 不参与验证
      test:  ["train", "test"]  # 使用 train/test 作为测试集
    region_map:
      ET: [3]
      TC: [1, 3]
      WT: [1, 2, 3]

  # 目标域：PED，仅作为测试集（所有 split 统一视为 test）
  - name: brats24_ped
    profile: ped
    csv_path: /home/dengzhipeng/data/brain_processed/BraTS24-PED/processed.csv
    root_dir: null
    include_splits:
      train: []                 # 不参与训练
      val:   []                 # 不参与验证
      test:  ["train", "test"]  # 使用 train/test 作为测试集
    region_map:
      ET: [1]
      TC: [1, 2, 3]
      WT: [1, 2, 3, 4]
