"""
Hook system for the training framework.
Hooks allow for decoupled components that can be plugged into the training loop.
"""

from abc import ABC, abstractmethod
import time
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .trainer_base import HookBase

class TimerHook(HookBase):
    """Measures training time."""
    
    def before_train(self):
        self.start_time = time.time()
        
    def after_train(self):
        elapsed_time = time.time() - self.start_time
        self.trainer.logger.info(f"Total training time: {elapsed_time:.2f} seconds")

    def before_train_epoch(self):
        self.epoch_start_time = time.time()
        
    def after_train_epoch(self):
        elapsed_time = time.time() - self.epoch_start_time
        self.trainer.logger.info(f"Epoch {self.trainer.epoch} took {elapsed_time:.2f} seconds")


class CheckpointHook(HookBase):
    """Saves model checkpoints."""

    def __init__(self, save_dir: str, save_freq: int = 1, save_start: int = 10):
        super().__init__()
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_start = save_start
        os.makedirs(os.path.join(self.save_dir, 'checkpoints'), exist_ok=True)

    def after_train_epoch(self):
        epoch = self.trainer.epoch
        if (epoch + 1) % self.save_freq == 0 and epoch + 1 >= self.save_start:
            self.save_checkpoint(epoch, is_best=False)
            
    def after_val(self, is_best):
        # Save best model based on validation accuracy
        if is_best:
            self.save_checkpoint(self.trainer.epoch, is_best=True)
            self.trainer.logger.info("Best model saved based on validation metrics.")

    def save_checkpoint(self, epoch: int, is_best: bool):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'best_metrics': self.trainer.best_metrics,
        }
        if self.trainer.scheduler:
            state['scheduler_state_dict'] = self.trainer.scheduler.state_dict()
        
        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"
            
        save_path = os.path.join(self.save_dir, 'checkpoints', filename)
        torch.save(state, save_path)
        self.trainer.logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint."""
        if not os.path.exists(path):
            self.trainer.logger.warning(f"Checkpoint not found at {path}, starting from scratch.")
            return 0
            
        checkpoint = torch.load(path, map_location=self.trainer.device)
        self.trainer.model.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.trainer.scheduler and 'scheduler_state_dict' in checkpoint:
            self.trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        # Load best metrics with backward compatibility
        if 'best_metrics' in checkpoint:
            self.trainer.best_metrics = checkpoint['best_metrics']
        elif 'best_acc' in checkpoint:
            # Backward compatibility
            self.trainer.best_metrics = {'eval_acc': checkpoint['best_acc']}
        
        self.trainer.logger.info(f"Checkpoint loaded from {path}, resuming from epoch {start_epoch}")
        return start_epoch


class LearningRateSchedulerHook(HookBase):
    """Updates learning rate."""
    
    def after_train_epoch(self):
        if self.trainer.scheduler:
            # Assuming scheduler is stepped per epoch
            self.trainer.scheduler.step()


class MemoryMonitorHook(HookBase):
    """Monitors GPU memory usage."""
    
    def after_train_step(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            if self.trainer.iter % 100 == 0: # Log every 100 iterations
                self.trainer.logger.debug(
                    f"GPU Memory: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB"
                )


class MetricsLoggerHook(HookBase):
    """
    Flexible metrics logging hook that can handle any metrics.
    This demonstrates how to implement task-specific logging without hardcoding metrics.
    """
    
    def __init__(self, log_every_n_epochs: int = 1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_epoch_end(self, epoch: int, train_stats: Dict[str, float], eval_stats: Dict[str, float], is_best: bool):
        """Log epoch metrics in a flexible way"""
        if epoch % self.log_every_n_epochs == 0:
            # Format training metrics
            train_str = self._format_metrics("Train", train_stats)
            eval_str = self._format_metrics("Eval", eval_stats)
            
            # Log the epoch summary
            self.trainer.logger.info(f"Epoch {epoch}: {train_str} | {eval_str}")
            
            # Log best model info if applicable
            if is_best:
                best_metric_info = self._format_best_metrics(eval_stats)
                self.trainer.logger.info(f"New best model: {best_metric_info}")
    
    def _format_metrics(self, prefix: str, metrics: Dict[str, float]) -> str:
        """Format metrics dictionary into a readable string"""
        if not metrics:
            return f"{prefix}: No metrics"
        
        formatted_metrics = []
        for key, value in metrics.items():
            if key == 'lr':
                formatted_metrics.append(f"LR: {value:.6f}")
            elif 'loss' in key.lower():
                formatted_metrics.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
            elif 'acc' in key.lower() or 'accuracy' in key.lower():
                formatted_metrics.append(f"{key.replace('_', ' ').title()}: {value:.2f}%")
            else:
                # Generic formatting for other metrics
                formatted_metrics.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
        
        return f"{prefix}: {', '.join(formatted_metrics)}"
    
    def _format_best_metrics(self, eval_stats: Dict[str, float]) -> str:
        """Format the key metric that determines 'best' model"""
        # This can be customized based on the task
        if 'eval_acc' in eval_stats:
            return f"Accuracy: {eval_stats['eval_acc']:.2f}%"
        elif 'accuracy' in eval_stats:
            return f"Accuracy: {eval_stats['accuracy']:.2f}%"
        else:
            # Fall back to first metric
            if eval_stats:
                key, value = next(iter(eval_stats.items()))
                return f"{key.replace('_', ' ').title()}: {value:.4f}"
            return "No metrics available"
