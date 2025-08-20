"""
Quantum-aware training loop for quantum-enhanced models.

This module implements training utilities specifically designed
for models with quantum-inspired components.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Callable
import time
import logging
from tqdm import tqdm

from ..utils.metrics import QuantumMetrics


class QuantumTrainer:
    """
    Quantum-aware trainer for quantum-enhanced models.
    
    Handles training with quantum-specific considerations like
    superposition collapse, entanglement training, and uncertainty
    quantification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: Optional[str] = None,
        quantum_training_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize quantum trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            loss_fn: Loss function
            device: Device to train on
            quantum_training_config: Configuration for quantum training
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Quantum training configuration
        self.quantum_config = quantum_training_config or {}
        self.superposition_schedule = self.quantum_config.get('superposition_schedule', 'linear')
        self.entanglement_training = self.quantum_config.get('entanglement_training', True)
        self.uncertainty_regularization = self.quantum_config.get('uncertainty_regularization', 0.1)
        
        # Metrics
        self.metrics = QuantumMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        collapse_probability: float = 1.0
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            collapse_probability: Probability of collapsing quantum states
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_quantum_loss = 0.0
        total_uncertainty = 0.0
        num_batches = 0
        
        # Update superposition schedule
        if self.superposition_schedule == 'linear':
            collapse_probability = min(1.0, epoch / 10.0)  # Gradually increase collapse
        elif self.superposition_schedule == 'cyclic':
            collapse_probability = 0.5 + 0.5 * torch.sin(torch.tensor(epoch * 0.1)).item()
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self._forward_pass(batch, collapse_probability)
            
            # Compute loss
            loss = self._compute_loss(batch, outputs)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            if hasattr(outputs, 'quantum_loss'):
                total_quantum_loss += outputs.quantum_loss.item()
            if hasattr(outputs, 'uncertainty'):
                total_uncertainty += outputs.uncertainty.mean().item()
            
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'collapse_prob': f'{collapse_probability:.2f}'
            })
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_quantum_loss = total_quantum_loss / num_batches if total_quantum_loss > 0 else 0.0
        avg_uncertainty = total_uncertainty / num_batches if total_uncertainty > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'quantum_loss': avg_quantum_loss,
            'uncertainty': avg_uncertainty,
            'collapse_probability': collapse_probability
        }
        
        return metrics
    
    def validate(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_quantum_loss = 0.0
        total_uncertainty = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass (always collapse for validation)
                outputs = self._forward_pass(batch, collapse_probability=1.0)
                
                # Compute loss
                loss = self._compute_loss(batch, outputs)
                
                # Update metrics
                total_loss += loss.item()
                if hasattr(outputs, 'quantum_loss'):
                    total_quantum_loss += outputs.quantum_loss.item()
                if hasattr(outputs, 'uncertainty'):
                    total_uncertainty += outputs.uncertainty.mean().item()
                
                num_batches += 1
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_quantum_loss = total_quantum_loss / num_batches if total_quantum_loss > 0 else 0.0
        avg_uncertainty = total_uncertainty / num_batches if total_uncertainty > 0 else 0.0
        
        metrics = {
            'val_loss': avg_loss,
            'val_quantum_loss': avg_quantum_loss,
            'val_uncertainty': avg_uncertainty
        }
        
        return metrics
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 5
    ) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            save_path: Path to save best model
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            val_metrics = {}
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader, epoch)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.training_history.append(epoch_metrics)
            
            # Log metrics
            self._log_metrics(epoch, epoch_metrics)
            
            # Check for best model
            if val_dataloader is not None:
                val_loss = val_metrics.get('val_loss', float('inf'))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    if save_path:
                        self.save_model(save_path)
                        self.logger.info(f"Saved best model to {save_path}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        self.logger.info("Training completed")
        return self.training_history
    
    def _forward_pass(
        self,
        batch: Dict[str, torch.Tensor],
        collapse_probability: float
    ) -> Any:
        """Perform forward pass with quantum considerations."""
        # Extract input data
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')
        
        # Create context for quantum collapse if available
        context = None
        if 'context' in batch:
            context = batch['context']
        
        # Forward pass
        if hasattr(self.model, 'forward'):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context=context,
                collapse=collapse_probability > 0.5
            )
        else:
            raise ValueError("Model must have a forward method")
        
        # Add quantum-specific outputs
        if hasattr(self.model, 'get_uncertainty'):
            uncertainty = self.model.get_uncertainty(input_ids)
            outputs.uncertainty = uncertainty
        
        return outputs
    
    def _compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Any
    ) -> torch.Tensor:
        """Compute loss with quantum regularization."""
        # Main task loss
        if 'labels' in batch:
            main_loss = self.loss_fn(outputs.logits, batch['labels'])
        else:
            main_loss = torch.tensor(0.0, device=self.device)
        
        # Quantum regularization
        quantum_reg = torch.tensor(0.0, device=self.device)
        
        if self.uncertainty_regularization > 0 and hasattr(outputs, 'uncertainty'):
            # Regularize uncertainty (encourage reasonable uncertainty levels)
            uncertainty = outputs.uncertainty
            target_uncertainty = torch.ones_like(uncertainty) * 0.5
            quantum_reg = F.mse_loss(uncertainty, target_uncertainty)
        
        # Total loss
        total_loss = main_loss + self.uncertainty_regularization * quantum_reg
        
        return total_loss
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    
    def _log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics."""
        metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f'Epoch {epoch}: {metric_str}')
    
    def save_model(self, path: str):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'quantum_config': self.quantum_config
        }, path)
    
    def load_model(self, path: str):
        """Load a saved model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        self.quantum_config = checkpoint.get('quantum_config', {})
        
        self.logger.info(f"Loaded model from {path}")
    
    def get_quantum_metrics(self) -> Dict[str, float]:
        """Get quantum-specific metrics."""
        return self.metrics.compute_metrics(self.model)
