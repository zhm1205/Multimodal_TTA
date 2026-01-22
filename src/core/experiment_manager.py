"""
Experiment Manager
Responsible for coordinating different components during experiments
"""

import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from ..core.trainers import SegTrainer
from ..core.hooks import (
    TimerHook, CheckpointHook, MemoryMonitorHook
)
from ..utils.logger import get_logger
from ..utils.metrics import set_random_seed
from ..utils.losses import FocalLoss, TripletLoss
from ..utils.config import require_config, get_config
from ..registry import (
    get_model,
    get_dataset_builder,
    get_evaluation_strategy,
)
from .. import datasets as _datasets  # noqa: F401
from .. import evaluation as _evaluation  # noqa: F401
from .. import models as _models  # noqa: F401

class ExperimentManager:
    """Experiment Manager Class"""
    
    def __init__(self, config: DictConfig):
        """Initialize experiment manager.

        Args:
            config: Experiment configuration as a plain ``dict``.
        """
        if not isinstance(config, DictConfig):
            raise TypeError("ExperimentManager expects configuration as a DictConfig")

        self.config = config
        self.logger = get_logger()

        self.distributed = bool(get_config(self.config, 'training.distributed', False))
        self.gpu_ids: List[int] = list(require_config(self.config, 'training.gpu_ids'))
        if not self.gpu_ids:
            raise ValueError("training.gpu_ids must specify at least one GPU id")

        # Setup device
        if torch.cuda.is_available() and self.gpu_ids[0] >= 0:
            self.device = torch.device(f'cuda:{self.gpu_ids[0]}')
        else:
            self.device = torch.device('cpu')

        # Set random seed and deterministic behaviour
        seed = require_config(config, 'task.seed')
        deterministic = str(get_config(config, 'task.deterministic', "practical"))
        set_random_seed(seed, deterministic)

        # Get task name
        self.task_name = require_config(config, 'task.name')
        self.eval_strategy = get_config(config, 'task.eval_strategy')

        # Initialize components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.trainer = None
        
        # Data related
        self.train_loader = None
        self.val_loader = None
        self.perturbation = None
        
        self.logger.info(f"Experiment Manager initialized for task: {self.task_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Random seed: {seed}")
        self.logger.info(f"Deterministic: {deterministic}")
    
    def setup_model(self) -> nn.Module:
        """Setup primary model; optionally also build UE surrogates if declared."""
        if not isinstance(self.config, DictConfig):
            raise TypeError(f"`self.config` must be DictConfig, got {type(self.config).__name__}")

        model_cfg: DictConfig = require_config(self.config, "model", type_=DictConfig)
        model_name: str = require_config(model_cfg, "name", type_=str)

        model_builder = get_model(model_name)
        self.model = model_builder(model_cfg)

        self.model = self.model.to(self.device)
        if torch.cuda.is_available() and len(self.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)

        # --- Conditional: build UE surrogates if cfg.ue.surrogates exists ---
        self.surrogates: Dict[str, nn.Module] = {}
        sur_cfg = get_config(self.config, "ue.surrogates")
        if sur_cfg:
            for name, scfg in sur_cfg.items():
                model_name = get_config(scfg, "backbone", str)
                s_cls = get_model(model_name)
                sm = s_cls(scfg).to(self.device)
                if torch.cuda.is_available() and len(self.gpu_ids) > 1:
                    sm = torch.nn.DataParallel(sm, device_ids=self.gpu_ids)
                self.surrogates[str(name)] = sm
            self.logger.info(f"UE surrogates created: {list(self.surrogates.keys())}")

        self.logger.info(f"Model created: {model_name}")
        return self.model

    
    def get_dataset_builder_for_task(self):
        """
        Return an instantiated dataset builder for the current task name.
        Keeps builder selection in one place so both victim training and UE generation reuse it.
        """
        try:
            builder_cls = get_dataset_builder(self.task_name)
        except KeyError:
            builder_cls = get_dataset_builder("default")
        return builder_cls(self.config)

    def build_clean_dataset(self, split: str = "train"):
        """
        Build a CLEAN dataset for the given split (no poison wrapping, no loaders).
        Use this for UE generation (training-free or training-based) to enumerate keys/samples.
        """
        builder = self.get_dataset_builder_for_task()
        return builder.get_dataset(split)

    def setup_train_data(self) -> tuple:
        """
        Build train/val/test loaders for segmentation training.
        Uses clean datasets without any UE wrapping.
        """
        builder = self.get_dataset_builder_for_task()

        train_ds = builder.get_dataset(split="train")
        val_ds   = builder.get_dataset(split="val")
        test_ds  = builder.get_dataset(split="test")

        # Use clean datasets directly
        self.train_loader = builder.get_loader("train", dataset=train_ds)
        if val_ds is None or len(val_ds) == 0:
            self.val_loader = None
            self.logger.warning("val dataset is empty; skip validation.")
        else:
            self.val_loader   = builder.get_loader("val",   dataset=val_ds)
        self.test_loader  = builder.get_loader("test",  dataset=test_ds)

        def _safe_len(dl): 
            try: return len(dl.dataset)
            except Exception: return "?"
        self.logger.info(
            f"Data loaders created for task: {self.task_name} | "
            f"train={_safe_len(self.train_loader)} val={_safe_len(self.val_loader)} test={_safe_len(self.test_loader)}"
        )
        return self.train_loader, self.val_loader, self.test_loader


    def setup_test_data(self):
        """
        Build a clean test loader. DOES NOT perform UE generation or poison wrapping.
        Returns: test_loader.
        """
        try:
            builder_cls = get_dataset_builder(self.task_name)
        except KeyError:
            builder_cls = get_dataset_builder("default")
        builder = builder_cls(self.config)

        self.test_loader = builder.get_loader("test")

        try:
            n_test = len(self.test_loader.dataset)
        except Exception:
            n_test = "?"
        self.logger.info(f"Test loader created for task: {self.task_name} | test={n_test}")
        return self.test_loader

    def setup_data(self, mode: str = "train"):
        """
        Convenience dispatcher to keep backward-compatibility.
        - mode='train' -> returns (train_loader, val_loader)
        - mode='test'  -> returns (test_loader, None)
        """
        mode = str(mode).lower()
        if mode == "train":
            return self.setup_train_data()
        if mode == "test":
            tl = self.setup_test_data()
            return tl, None
        raise ValueError(f"Unknown mode: {mode}. Expected 'train' or 'test'.")


    def _build_optimizer_for(self, params, opt_cfg) -> torch.optim.Optimizer:
        OPTIMIZER_SPACE = {
            "sgd": (torch.optim.SGD, {"lr","momentum","weight_decay","dampening","nesterov","foreach","maximize"}),
            "adam": (torch.optim.Adam, {"lr","betas","eps","weight_decay","amsgrad","foreach","maximize","capturable","differentiable"}),
            "adamw": (torch.optim.AdamW, {"lr","betas","eps","weight_decay","amsgrad","foreach","maximize","capturable","differentiable","fused"}),
        }
        name = str(get_config(opt_cfg, "name", get_config(self.config, "training.optimizer", "sgd"))).lower()
        if name not in OPTIMIZER_SPACE:
            raise ValueError(f"Unsupported optimizer: {name}")
        opt_cls, allowed = OPTIMIZER_SPACE[name]

        # compatible param_groups（no_decay rules）, align with original setup_optimizer
        train_cfg: DictConfig = require_config(self.config, "training")
        wd = float(get_config(opt_cfg, "weight_decay", get_config(train_cfg, "weight_decay", 0.0)))

        def _split(params_iter, rules: DictConfig):
            no_decay_keys = set(get_config(rules, "no_decay_keys", []))
            treat_1d = bool(get_config(rules, "treat_1d_as_no_decay", True))
            decay, no_decay = [], []
            for n, p in (getattr(params_iter, "named_parameters", lambda: [])()):
                if not p.requires_grad: continue
                is_nd = any(k in n for k in no_decay_keys) or (treat_1d and p.ndim == 1)
                (no_decay if is_nd else decay).append(p)
            if not decay and not no_decay:
                # params_iter may not be a module, treat it as a list of params
                return [{"params": list(params_iter), "weight_decay": wd}]
            return [
                {"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0},
            ]

        param_groups = _split(params, get_config(train_cfg, "param_groups", OmegaConf.create({})))
        kwargs = {k: opt_cfg[k] for k in allowed if k in opt_cfg}
        # fill common lr/momentum
        if "lr" not in kwargs:
            kwargs["lr"] = float(get_config(opt_cfg, "lr", get_config(train_cfg, "learning_rate", 1e-3)))
        if name == "sgd" and "momentum" not in kwargs:
            kwargs["momentum"] = float(get_config(opt_cfg, "momentum", get_config(train_cfg, "momentum", 0.0)))
        return opt_cls(param_groups, **kwargs)

    
    def setup_optimizer(self) -> torch.optim.Optimizer:
        if self.model is None:
            raise ValueError("Model must be setup before optimizer")

        train_cfg: DictConfig = require_config(self.config, "training")
        optim_blocks: DictConfig = get_config(train_cfg, "optimizers", OmegaConf.create({}))
        opt_name: str = str(get_config(train_cfg, "optimizer", "sgd"))
        opt_cfg: DictConfig = get_config(optim_blocks, opt_name, OmegaConf.create({}))
        # compatible with writing lr/momentum/weight_decay directly in training.*
        if get_config(opt_cfg, "lr", None) is None and get_config(train_cfg, "learning_rate", None) is not None:
            opt_cfg.lr = get_config(train_cfg, "learning_rate")
        if get_config(opt_cfg, "weight_decay", None) is None and get_config(train_cfg, "weight_decay", None) is not None:
            opt_cfg.weight_decay = get_config(train_cfg, "weight_decay")
        if opt_name == "sgd" and get_config(opt_cfg, "momentum", None) is None and get_config(train_cfg, "momentum", None) is not None:
            opt_cfg.momentum = get_config(train_cfg, "momentum")

        # primary model optimizer
        self.optimizer = self._build_optimizer_for(self.model, opt_cfg)

        # multiple surrogates (optional)
        if hasattr(self, "surrogates"):
            self.sur_optimizers = {}
            for name, m in self.surrogates.items():
                # 从surrogate配置中获取optimizer配置
                sur_cfg = require_config(self.config, f"ue.surrogates.{name}")
                ocfg = require_config(sur_cfg, "optimizer")
                if ocfg:
                    self.sur_optimizers[name] = self._build_optimizer_for(m, ocfg)

        self.logger.info(f"Optimizer created (primary): {opt_name}")
        if hasattr(self, "sur_optimizers") and self.sur_optimizers:
            self.logger.info(f"Optimizers for surrogates: {list(self.sur_optimizers.keys())}")
        return self.optimizer


    def setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler from merged training config."""
        if self.optimizer is None:
            raise ValueError("Optimizer must be setup before scheduler")

        training_cfg = require_config(self.config, "training")
        scheduler_cfg: DictConfig = get_config(training_cfg, "scheduler", OmegaConf.create({}))
        scheduler_name: str = str(get_config(scheduler_cfg, "name", "none")).lower()

        if scheduler_name == "multistep":
            milestones = list(get_config(training_cfg, "milestones", [100, 150]))
            gamma = float(get_config(training_cfg, "gamma", 0.1))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        elif scheduler_name == "cosine":
            epochs = int(get_config(training_cfg, "epochs", 200))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        elif scheduler_name == "step":
            step_size = int(get_config(training_cfg, "step_size", 30))
            gamma = float(get_config(training_cfg, "gamma", 0.1))
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        elif scheduler_name == "reduce_on_plateau":
            rop_cfg: DictConfig = get_config(training_cfg, "reduce_on_plateau", OmegaConf.create({}))
            patience = int(get_config(rop_cfg, "patience", 10))
            factor = float(get_config(rop_cfg, "factor", 0.1))
            min_lr = float(get_config(rop_cfg, "min_lr", 1.0e-7))
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
            )

        elif scheduler_name == "none":
            self.scheduler = None

        else:
            # Keep it quiet: unknown names simply result in no scheduler.
            self.scheduler = None

        if self.scheduler:
            self.logger.info(f"Scheduler created: {scheduler_name}")
        return self.scheduler

    def setup_hooks(self):
        """Setup and register hooks"""
        hooks = []
        
        # Timer hook
        hooks.append(TimerHook())
        try:
            run_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            run_dir = get_config(self.config, 'task.save_dir', './outputs')

        # Checkpoint hook
        ckpt_dir = os.path.join(run_dir, "checkpoints")

        model_save_freq = get_config(self.config, 'training.model_save_freq', 1)
        model_save_start = get_config(self.config, 'training.model_save_start', 50)
        hooks.append(CheckpointHook(ckpt_dir, model_save_freq, model_save_start))
                
        # Memory monitor hook
        hooks.append(MemoryMonitorHook())
        
        self.trainer.register_hooks(hooks)
        self.logger.info(f"{len(hooks)} hooks registered.")

    def train(self, epochs: int) -> Dict[str, List]:
        """Execute training - delegate to trainer"""
        if self.trainer is None:
            raise ValueError("Trainer must be setup before training")
        
        self.logger.info(f"Starting training for {epochs} epochs...")
        
        eval_on_train = get_config(self.config, 'training.eval_on_train', False)
        self.logger.info(f"Eval on train: {eval_on_train}")
        # Delegate training to trainer - trainer handles all epoch loops and hooks
        training_results = self.trainer.train(
            epochs=epochs,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            eval_on_train=eval_on_train
        )
        
        self.logger.info("Training completed")
        
        return training_results

    def setup_trainer(self) -> None:
        """Setup trainer for segmentation tasks."""
        if self.eval_strategy is None:
            evaluation_strategy = None
        else:
            evaluation_cls = get_evaluation_strategy(self.eval_strategy)
            evaluation_strategy = evaluation_cls(self.config)

        # 判断是否为分割任务（支持多种命名方式）
        task_lower = self.task_name.lower()
        is_seg_task = (
            "seg" in task_lower or 
            "brats" in task_lower or
            "hecktor21" in task_lower
        )
        
        if is_seg_task:
            self.trainer = SegTrainer(self.config, self.device, evaluation_strategy)
            self.trainer.setup(
                self.model,
                self.criterion,
                self.optimizer,
                self.scheduler,
                evaluation_strategy
            )
        else:
            raise ValueError(f"Unknown trainer type: {self.task_name}")

        # Register hooks
        self.setup_hooks()

        self.logger.info(f"Trainer created: {type(self.trainer).__name__} for task: {self.task_name}")
