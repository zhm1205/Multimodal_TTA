"""
Trainer base class
Responsible for executing training loops with hooks and progress tracking
"""

import weakref
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from omegaconf import DictConfig

from ..utils.logger import get_logger
from ..utils.metrics import AverageMeter
from ..utils.config import get_config


class HookBase:
    """
    Base class for hooks that can be registered with TrainerBase.

    Each hook can implement various methods that are called at different points
    during training. The basic lifecycle is:

    hook.before_train()
    for epoch in range(epochs):
        hook.before_train_epoch()
        for batch in data_loader:
            hook.before_train_step()
            trainer.run_step()
            hook.after_train_step()
        hook.after_train_epoch()
        hook.before_val()
        # validation
        hook.after_val()
    hook.after_train()
    """

    trainer: "TrainerBase" = None
    """A weak reference to the trainer object. Set by the trainer when the hook is registered."""

    def before_train(self):
        """Called before the first epoch."""
        pass

    def after_train(self):
        """Called after the last epoch."""
        pass

    def before_train_epoch(self):
        """Called before each training epoch."""
        pass

    def after_train_epoch(self):
        """Called after each training epoch."""
        pass

    def before_train_step(self):
        """Called before each training step."""
        pass

    def after_train_step(self):
        """Called after each training step."""
        pass

    def before_val(self):
        """Called before validation."""
        pass

    def after_val(self, is_best: bool):
        """Called after validation."""
        pass

    def on_epoch_end(
        self,
        epoch: int,
        train_stats: Dict[str, float],
        eval_stats: Dict[str, float],
        is_best: bool,
    ):
        """
        Called at the end of each epoch with complete statistics.
        This is where logging, checkpointing, and other epoch-end activities should happen.
        """
        pass

    def state_dict(self):
        """Hooks are stateless by default, but can be made checkpointable."""
        return {}


class TrainerBase(ABC):
    """
    Base class for all trainers.

    Responsibilities:
    1. Execute complete training lifecycle (all epochs)
    2. Manage hooks at appropriate points
    3. Handle progress tracking with tqdm
    4. Coordinate training and evaluation
    """

    def __init__(self, config: DictConfig, device: torch.device):
        self.config = config
        self.device = device
        self.logger = get_logger()

        # Training state
        self.epoch = 0
        self.iter = 0
        self.best_metrics: Dict[str, float] = {}

        # Components (will be set by setup method)
        self.model: Optional[nn.Module] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.evaluation_strategy = None

        # Hooks
        self._hooks: List[HookBase] = []

    def setup(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        evaluation_strategy,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluation_strategy = evaluation_strategy
        self.logger.info("Trainer setup completed")

    def register_hooks(self, hooks: List[HookBase]):
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)  # avoid circular ref
        self._hooks.extend(hooks)
        self.logger.info(f"Registered {len(hooks)} hooks")

    # ---------------------------
    # Eval/Test schedule
    # ---------------------------
    def _should_run_eval_test(self, epoch: int, epochs: int) -> bool:
        """
        Shared schedule for val/test:
          - 0-based epoch
          - start_epoch: inclusive
          - every_n_epochs: interval
          - run_last: force run on last epoch
        """
        start_epoch = get_config(self.config, "training.eval_test.start_epoch", 0, int)
        every_n = get_config(self.config, "training.eval_test.every_n_epochs", 1, int)
        run_last = get_config(self.config, "training.eval_test.run_last", True, bool)

        # Defensive: avoid invalid interval
        if every_n is None or int(every_n) <= 0:
            every_n = 1
        start_epoch = int(start_epoch)

        epoch_last = (epoch == epochs - 1)
        should = (epoch >= start_epoch) and ((epoch - start_epoch) % int(every_n) == 0)
        if bool(run_last) and epoch_last:
            should = True
        return should

    def train(
        self,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader = None,
        eval_on_train: bool = False,
    ) -> Dict[str, List]:
        self.logger.info(f"Starting training for {epochs} epochs")

        train_history: List[Dict[str, float]] = []
        eval_history: List[Dict[str, float]] = []

        # Hook: before train
        for h in self._hooks:
            h.before_train()

        # schedule toggles (shared schedule; val/test can be enabled separately)
        do_val = get_config(self.config, "training.eval_test.do_val", True, bool)
        do_test = get_config(self.config, "training.eval_test.do_test", False, bool)

        pbar = tqdm(range(epochs), desc="Training Epochs", leave=True)

        try:
            for epoch in pbar:
                self.epoch = epoch

                # ---- train ----
                self.before_train_epoch()
                train_stats = self.train_epoch(epoch, train_loader)
                train_history.append(train_stats)
                self.after_train_epoch()

                # ---- scheduled eval/test ----
                should_run = self._should_run_eval_test(epoch, epochs)

                eval_stats: Dict[str, float] = {}
                is_best = False
                if should_run and bool(do_val) and (val_loader is not None):
                    eval_stats, is_best = self.evaluate(epoch, val_loader)

                # A: every epoch append (no-eval -> {})
                eval_history.append(eval_stats)

                # ---- optional debug eval on train ----
                if train_loader is not None and eval_on_train:
                    if epoch > 0 and epoch % 10 == 0:
                        self.eval_on_train(epoch, train_loader)

                # ---- scheduled test ----
                if should_run and bool(do_test) and (test_loader is not None):
                    self.test(epoch, test_loader)

                # A: on_epoch_end every epoch
                for h in self._hooks:
                    if hasattr(h, "on_epoch_end"):
                        h.on_epoch_end(epoch, train_stats, eval_stats, is_best)

                # ---- progress/log ----
                if eval_stats.get("loss") is not None:
                    pbar.set_postfix(
                        {"train_loss": train_stats.get("loss", None), "val_loss": eval_stats.get("loss", None)}
                    )
                    self.logger.info(
                        f"Epoch {epoch} completed. Train loss: {train_stats.get('loss')}, Val loss: {eval_stats.get('loss')}"
                    )
                else:
                    pbar.set_postfix({"train_loss": train_stats.get("loss", None)})
                    self.logger.info(f"Epoch {epoch} completed. Train loss: {train_stats.get('loss')}")

        except StopIteration as e:
            self.logger.info(f"Training stopped early: {e}")

        finally:
            for h in self._hooks:
                h.after_train()

        self.logger.info("Training completed")
        return {"train_history": train_history, "eval_history": eval_history}

    def train_epoch(self, epoch: int, data_loader: DataLoader) -> Dict[str, float]:
        assert self.model is not None, "model is not set; call setup() first"
        self.model.train()

        # (DDP not used now, but keeping harmless)
        if torch.distributed.is_initialized():
            sampler = getattr(data_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        metrics = self._init_epoch_metrics()

        for h in self._hooks:
            h.before_train_epoch()

        show_pbar = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
        pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]", leave=False) if show_pbar else data_loader

        for batch in pbar:
            for h in self._hooks:
                h.before_train_step()

            self.before_step()
            step_metrics = self.run_step(batch)
            self._update_metrics(metrics, step_metrics)
            self.after_step()

            if show_pbar:
                pbar.set_postfix(**self._format_progress_metrics(metrics))

            self.iter += 1

            for h in self._hooks:
                h.after_train_step()

        for h in self._hooks:
            h.after_train_epoch()

        if self.scheduler:
            self.scheduler.step()

        return self._finalize_epoch_metrics(metrics)

    def before_train_epoch(self):
        pass

    def after_train_epoch(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    @abstractmethod
    def run_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        pass

    def _init_epoch_metrics(self) -> Dict[str, Any]:
        return {"loss": AverageMeter()}

    def _update_metrics(self, metrics: Dict[str, Any], step_metrics: Dict[str, float]):
        for key, value in step_metrics.items():
            if key in metrics:
                metrics[key].update(value)

    def _format_progress_metrics(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        formatted: Dict[str, str] = {}
        for key, meter in metrics.items():
            if hasattr(meter, "avg"):
                formatted[key] = f"{meter.avg:.6f}" if key == "loss" else f"{meter.avg:.3f}"
        return formatted

    def _finalize_epoch_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        final_metrics: Dict[str, float] = {}
        for key, meter in metrics.items():
            if hasattr(meter, "avg"):
                final_metrics[key] = float(meter.avg)

        if self.optimizer:
            final_metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return final_metrics

    def evaluate(self, epoch: int, data_loader: DataLoader) -> Tuple[Dict[str, float], bool]:
        if self.evaluation_strategy is None:
            self.logger.warning("No evaluation strategy set, skipping evaluation.")
            return {}, False

        for h in self._hooks:
            h.before_val()

        eval_stats = self.evaluation_strategy.evaluate_epoch(self.model, data_loader, self.device)
        self.logger.info(f"Epoch {epoch} evaluation results: {eval_stats}")

        is_best = self._is_best_model(eval_stats)
        if is_best:
            self._update_best_metrics(eval_stats)

        for h in self._hooks:
            h.after_val(is_best)

        return eval_stats, is_best

    def eval_on_train(self, epoch: int, data_loader: DataLoader) -> Dict[str, float]:
        if self.evaluation_strategy is None:
            self.logger.warning("No evaluation strategy set, skipping evaluation.")
            return {}

        stats = self.evaluation_strategy.evaluate_epoch(self.model, data_loader, self.device)
        self.logger.info(f"Epoch {epoch} evaluation on train dataset results: {stats}")
        return stats

    def test(self, epoch: int, data_loader: DataLoader) -> Dict[str, float]:
        if self.evaluation_strategy is None:
            self.logger.warning("No evaluation strategy set, skipping evaluation.")
            return {}

        stats = self.evaluation_strategy.evaluate_epoch(self.model, data_loader, self.device)
        self.logger.info(f"Epoch {epoch} test results: {stats}")
        return stats

    def _is_best_model(self, eval_stats: Dict[str, float]) -> bool:
        # Default: no best tracking
        return False

    def _update_best_metrics(self, eval_stats: Dict[str, float]):
        self.best_metrics.update(eval_stats)