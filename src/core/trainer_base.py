"""
Trainer base class
Responsible for executing training loops with hooks and progress tracking
"""

import logging
import weakref
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logger import get_logger
from ..utils.metrics import AverageMeter
from omegaconf import DictConfig

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
    
    def on_epoch_end(self, epoch: int, train_stats: Dict[str, float], eval_stats: Dict[str, float], is_best: bool):
        """
        Called at the end of each epoch with complete statistics.
        This is where logging, checkpointing, and other epoch-end activities should happen.
        
        Args:
            epoch: Current epoch number
            train_stats: Training statistics for the epoch
            eval_stats: Evaluation statistics for the epoch  
            is_best: Whether this epoch achieved the best performance
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
        """
        Initialize trainer base
        
        Args:
            config: Training configuration
            device: Device to run training on
        """
        self.config = config
        self.device = device
        self.logger = get_logger()
        
        # Training state
        self.epoch = 0
        self.iter = 0
        self.best_metrics = {}
        
        # Components (will be set by setup method)
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.evaluation_strategy = None
        
        # Hooks
        self._hooks: List[HookBase] = []
    
    def setup(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        evaluation_strategy
    ):
        """
        Setup trainer with components from ExperimentManager
        
        Args:
            model: Neural network model
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            evaluation_strategy: Evaluation strategy for the task
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluation_strategy = evaluation_strategy
        
        self.logger.info("Trainer setup completed")
    
    def register_hooks(self, hooks: List[HookBase]):
        """
        Register hooks to the trainer
        
        Args:
            hooks: List of hooks to register
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # Use weak reference to avoid circular reference
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)
        
        self.logger.info(f"Registered {len(hooks)} hooks")
    
    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader=None, eval_on_train: bool=False) -> Dict[str, List]:
        """
        Execute complete training for specified epochs
        
        Args:
            epochs: Number of epochs to train
            train_loader: Training data loader
            val_loader: Val data loader
            
        Returns:
            Dictionary containing training and evaluation history
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        
        train_history = []
        eval_history = []
        
        # Hook: before train
        for h in self._hooks:
            h.before_train()

        # Create outer progress bar for epochs
        pbar = tqdm(range(epochs), desc="Training Epochs", leave=True)

        try:
            for epoch in pbar:
                self.epoch = epoch

                # Training phase
                self.before_train_epoch()
                train_stats = self.train_epoch(epoch, train_loader)
                train_history.append(train_stats)
                self.after_train_epoch()
                # Evaluation phase
                eval_stats = {}
                is_best = False
                if val_loader is not None:
                    eval_stats, is_best = self.evaluate(epoch, val_loader)
                eval_history.append(eval_stats)

                # Evaluation on train dataset just for debug
                if train_loader is not None and eval_on_train:
                    if epoch > 0 and epoch % 10 == 0:
                        self.eval_on_train(epoch, train_loader)

                if test_loader is not None:
                    self.test(epoch, test_loader)
                
                # Let hooks handle best model tracking and logging
                for h in self._hooks:
                    if hasattr(h, 'on_epoch_end'):
                        h.on_epoch_end(epoch, train_stats, eval_stats, is_best)

                # Update outer progress bar with current epoch results
                if eval_stats.get('loss') is not None:
                    pbar.set_postfix({"train_loss": train_stats["loss"], "val_loss": eval_stats["loss"]})
                    self.logger.info(f"Epoch {epoch} completed. Train loss: {train_stats['loss']}, Val loss: {eval_stats['loss']}")
                else:
                    pbar.set_postfix({"train_loss": train_stats["loss"]})
                    self.logger.info(f"Epoch {epoch} completed. Train loss: {train_stats['loss']}")
            
        except StopIteration as e:
            self.logger.info(f"Training stopped early: {e}")
        
        finally:
            # Hook: after train
            for h in self._hooks:
                h.after_train()
        
        self.logger.info("Training completed")
        
        return {
            'train_history': train_history,
            'eval_history': eval_history
        }
    
    def train_epoch(self, epoch: int, data_loader: DataLoader) -> Dict[str, float]:
        """
        Run one epoch of training with progress bar
        
        Args:
            epoch: Current epoch number
            data_loader: Training data loader
            
        Returns:
            Training statistics for the epoch
        """
        self.model.train()

        # Distributed sampler requires set_epoch for shuffling
        if torch.distributed.is_initialized():
            sampler = getattr(data_loader, 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

        # Initialize metric tracking - delegate to subclasses
        metrics = self._init_epoch_metrics()

        # Hook: before train epoch
        for h in self._hooks:
            h.before_train_epoch()

        # Use progress bar only on rank 0 to avoid duplication
        show_pbar = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]", leave=False) if show_pbar else data_loader

        for batch in pbar:
            # Hook: before train step
            for h in self._hooks:
                h.before_train_step()
            self.before_step()
            # Execute one training step
            step_metrics = self.run_step(batch)
            
            # Update metrics
            self._update_metrics(metrics, step_metrics)
            self.after_step()
            # Update progress bar - delegate to subclasses for formatting
            if show_pbar:
                pbar.set_postfix(**self._format_progress_metrics(metrics))
            
            self.iter += 1
            
            # Hook: after train step
            for h in self._hooks:
                h.after_train_step()
        
        # Hook: after train epoch
        for h in self._hooks:
            h.after_train_epoch()
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        # Finalize metrics for the epoch
        return self._finalize_epoch_metrics(metrics)
    
    def before_train_epoch(self):
        """Subclass hook: called before self.train_epoch()"""
        pass

    def after_train_epoch(self):
        """Subclass hook: called after self.train_epoch()"""
        pass

    def before_step(self):
        """Subclass hook: called before self.run_step()"""
        pass

    def after_step(self):
        """Subclass hook: called after self.run_step()"""
        pass

    @abstractmethod
    def run_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Run a single training step. Must be implemented by subclasses.
        
        Args:
            data: Input data for the step
            target: Target labels for the step
            
        Returns:
            A dictionary containing metrics for the step (e.g., {'loss': 0.5, 'accuracy': 0.9})
        """
        pass
    
    def _init_epoch_metrics(self) -> Dict[str, Any]:
        """
        Initialize metric tracking for an epoch. Can be overridden by subclasses.
        
        Returns:
            Dictionary to store metric accumulators
        """
        from ..utils.metrics import AverageMeter
        return {
            'loss': AverageMeter()
        }
    
    def _update_metrics(self, metrics: Dict[str, Any], step_metrics: Dict[str, float]):
        """
        Update epoch metrics with step results. Can be overridden by subclasses.
        
        Args:
            metrics: Dictionary of metric accumulators
            step_metrics: Metrics from the current step
        """
        for key, value in step_metrics.items():
            if key in metrics:
                metrics[key].update(value)
    
    def _format_progress_metrics(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Format metrics for progress bar display. Can be overridden by subclasses.
        
        Args:
            metrics: Dictionary of metric accumulators
            
        Returns:
            Dictionary of formatted metric strings for display
        """
        formatted = {}
        for key, meter in metrics.items():
            if hasattr(meter, 'avg'):
                if key == 'loss':
                    formatted[key] = f"{meter.avg:.6f}"
                else:
                    # Default formatting for other metrics
                    formatted[key] = f"{meter.avg:.3f}"
        return formatted
    
    def _finalize_epoch_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Finalize metrics for the epoch. Can be overridden by subclasses.
        
        Args:
            metrics: Dictionary of metric accumulators
            
        Returns:
            Dictionary of final metric values for the epoch
        """
        final_metrics = {}
        for key, meter in metrics.items():
            if hasattr(meter, 'avg'):
                final_metrics[key] = meter.avg
        
        # Always include learning rate
        if self.optimizer:
            final_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
        return final_metrics
    
    def evaluate(self, epoch: int, data_loader: DataLoader) -> tuple[Dict[str, float], bool]:
        """
        Run evaluation
        
        Args:
            epoch: Current epoch number
            data_loader: Evaluation data loader
            
        Returns:
            A tuple containing evaluation statistics and a boolean indicating if it's the best model
        """
        if self.evaluation_strategy is None:
            self.logger.warning("No evaluation strategy set, skipping evaluation.")
            return {}, False
        
        # Hook: before validation
        for h in self._hooks:
            h.before_val()
        
        eval_stats = self.evaluation_strategy.evaluate_epoch(self.model, data_loader, self.device)
        self.logger.info(f"Epoch {epoch} evaluation results: {eval_stats}")        
        # Determine if this is the best model - delegate to subclasses
        is_best = self._is_best_model(eval_stats)
        if is_best:
            self._update_best_metrics(eval_stats)
        
        # Hook: after validation
        for h in self._hooks:
            h.after_val(is_best)

        return eval_stats, is_best
    
    
    def eval_on_train(self, epoch: int, data_loader: DataLoader) -> Dict[str, float]:
        """
        Run test on train dataset just for debug
        
        Args:
            epoch: Current epoch number
            data_loader: Training data loader
            
        Returns:
            A tuple containing evaluation statistics
        """
        if self.evaluation_strategy is None:
            self.logger.warning("No evaluation strategy set, skipping evaluation.")
            return {}, False
        
        test_stats = self.evaluation_strategy.evaluate_epoch(self.model, data_loader, self.device)
        self.logger.info(f"Epoch {epoch} evaluation on train dataset results: {test_stats}")        
        return test_stats
    
    def test(self, epoch: int, data_loader: DataLoader) -> Dict[str, float]:
        """
        Run test
        
        Args:
            epoch: Current epoch number
            data_loader: Evaluation data loader
            
        Returns:
            A tuple containing evaluation statistics
        """
        if self.evaluation_strategy is None:
            self.logger.warning("No evaluation strategy set, skipping evaluation.")
            return {}, False
        
        test_stats = self.evaluation_strategy.evaluate_epoch(self.model, data_loader, self.device)
        self.logger.info(f"Epoch {epoch} test results: {test_stats}")        
        return test_stats
    
    def _is_best_model(self, eval_stats: Dict[str, float]) -> bool:
        """
        Determine if current evaluation results represent the best model.
        Can be overridden by subclasses to implement custom criteria.
        
        Args:
            eval_stats: Evaluation statistics
            
        Returns:
            True if this is the best model so far
        """
        # Default implementation: no best model tracking
        return False
    
    def _update_best_metrics(self, eval_stats: Dict[str, float]):
        """
        Update best metrics tracking. Can be overridden by subclasses.
        
        Args:
            eval_stats: Evaluation statistics
        """
        # Default implementation: store all metrics
        self.best_metrics.update(eval_stats)
