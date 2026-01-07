"""
源代码包初始化文件
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "Unlearnable Examples Team"
__email__ = "example@example.com"

# 导入主要组件
from .registry import (
    register_model,
    register_dataset,
    register_dataset_builder,
    register_evaluation_strategy,
    get_model,
    get_dataset,
    get_dataset_builder,
    get_evaluation_strategy,
    list_models,
    list_datasets,
    list_dataset_builders,
    list_evaluation_strategies,
)


# 导入工具函数
from .utils.logger import setup_logger
from .utils.metrics import AverageMeter, set_random_seed

__all__ = [
    # 版本信息
    '__version__', '__author__', '__email__',
    
    # 注册器
    'register_model', 'register_dataset', 'register_dataset_builder', 'register_evaluation_strategy',
    'get_model', 'get_dataset', 'get_dataset_builder', 'get_evaluation_strategy',
    'list_models', 'list_datasets', 'list_dataset_builders', 'list_evaluation_strategies',

    # 工具函数
    'setup_logger', 'AverageMeter', 'set_random_seed'
]
