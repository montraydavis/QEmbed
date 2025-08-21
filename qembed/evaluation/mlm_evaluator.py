"""
MLM task evaluator for QEmbed.

⚠️ CRITICAL: This evaluator integrates with existing QEmbed infrastructure
    and handles quantum BERT MLM outputs properly.

Evaluates MLM models with perplexity and accuracy metrics.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
from .base_evaluator import BaseEvaluator, EvaluationResult
from .evaluation_metrics import EvaluationMetrics

class MLMEvaluator(BaseEvaluator):
    """
    Evaluator for Masked Language Modeling (MLM) tasks.
    
    Computes perplexity, accuracy, and quantum-specific metrics
    for MLM models including quantum BERT.
    
    ⚠️ CRITICAL: Extends BaseEvaluator and integrates with existing QuantumMetrics
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        device: Optional[str] = None,
        ignore_index: int = -100
    ):
        """
        Initialize MLM evaluator.
        
        Args:
            model: MLM model to evaluate
            device: Device to run evaluation on
            ignore_index: Index to ignore in loss computation
        """
        super().__init__(model, device)
        self.ignore_index = ignore_index
        
        # ⚠️ CRITICAL: Use existing QuantumMetrics instance from parent class
        self.metrics_calculator = EvaluationMetrics()
    
    def evaluate(
        self, 
        dataloader: torch.utils.data.DataLoader,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate MLM model.
        
        Args:
            dataloader: DataLoader containing evaluation data
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with MLM metrics
        """
        all_logits = []
        all_labels = []
        all_uncertainties = []
        total_loss = 0.0
        num_batches = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self._evaluate_batch_internal(batch)
                
                # Collect results
                all_logits.append(outputs['logits'])
                all_labels.append(outputs['labels'])
                if 'uncertainty' in outputs and outputs['uncertainty'] is not None:
                    all_uncertainties.append(outputs['uncertainty'])
                
                # Accumulate loss
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                    num_batches += 1
        
        # Concatenate results
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0) if all_uncertainties else None
        
        # Compute MLM-specific metrics
        metrics = self._compute_mlm_metrics(logits, labels)
        
        # Add average loss if available
        if num_batches > 0:
            metrics['average_loss'] = total_loss / num_batches
        
        # Compute quantum metrics if available
        quantum_metrics = {}
        if uncertainties is not None:
            # ⚠️ CRITICAL: Use parent class method to avoid conflicts
            quantum_metrics = self.compute_quantum_metrics({'quantum_uncertainty': uncertainties})
        
        # Create result
        result = EvaluationResult(
            task_name="mlm",
            model_name=self.model.__class__.__name__,
            metrics=metrics,
            quantum_metrics=quantum_metrics,
            metadata={
                'ignore_index': self.ignore_index,
                'num_samples': len(logits),
                'vocab_size': logits.size(-1),
                'sequence_length': logits.size(1)
            }
        )
        
        # Store result
        self.current_result = result
        self.results.append(result)
        
        return result
    
    def _evaluate_batch_internal(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate a single batch of data.
        
        ⚠️ CRITICAL: This method handles both quantum BERT and legacy MLM models
        """
        # Extract batch components
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Get logits and loss
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        loss = None
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        
        # Get uncertainty if available
        uncertainty = None
        if hasattr(outputs, 'quantum_uncertainty'):
            uncertainty = outputs.quantum_uncertainty
        
        return {
            'logits': logits,
            'labels': labels,
            'loss': loss,
            'uncertainty': uncertainty
        }
    
    def _compute_mlm_metrics(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute MLM-specific metrics."""
        # ⚠️ CRITICAL: Use metrics calculator to avoid duplicating functionality
        metrics = self.metrics_calculator.mlm_metrics(
            logits, labels, ignore_index=self.ignore_index
        )
        
        # Add additional MLM-specific metrics
        valid_mask = labels != self.ignore_index
        if valid_mask.sum() > 0:
            valid_logits = logits[valid_mask]
            valid_labels = labels[valid_mask]
            
            # Compute top-k accuracy for different k values
            for k in [1, 3, 5, 10]:
                top_k_acc = self._compute_top_k_accuracy(valid_logits, valid_labels, k)
                metrics[f'top_{k}_accuracy'] = top_k_acc
            
            # Compute perplexity per token
            token_perplexities = []
            for i in range(valid_logits.size(0)):
                token_logits = valid_logits[i:i+1]
                token_label = valid_labels[i:i+1]
                token_loss = torch.nn.functional.cross_entropy(
                    token_logits, token_label, ignore_index=self.ignore_index
                )
                token_perplexity = torch.exp(token_loss)
                token_perplexities.append(token_perplexity.item())
            
            metrics['mean_token_perplexity'] = np.mean(token_perplexities)
            metrics['std_token_perplexity'] = np.std(token_perplexities)
        
        return metrics
    
    def _compute_top_k_accuracy(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        k: int
    ) -> float:
        """Compute top-k accuracy for MLM."""
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        correct = torch.any(top_k_indices == labels.unsqueeze(-1), dim=-1)
        return correct.float().mean().item()
    
    def evaluate_single_sample(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        masked_positions: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample for real-time inference.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            masked_positions: Optional list of masked token positions
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare input
            inputs = {'input_ids': input_ids.to(self.device)}
            if attention_mask is not None:
                inputs['attention_mask'] = attention_mask.to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Get logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Get predictions for masked positions
            predictions = {}
            if masked_positions:
                for pos in masked_positions:
                    if pos < logits.size(1):
                        pos_logits = logits[0, pos, :]  # [vocab_size]
                        top_k_logits, top_k_indices = torch.topk(pos_logits, k=5)
                        top_k_probs = torch.softmax(top_k_logits, dim=-1)
                        
                        predictions[f'position_{pos}'] = {
                            'top_tokens': top_k_indices.cpu().numpy(),
                            'top_probabilities': top_k_probs.cpu().numpy()
                        }
            
            # Get uncertainty if available
            uncertainty = None
            if hasattr(outputs, 'quantum_uncertainty'):
                uncertainty = outputs.quantum_uncertainty
            
            return {
                'predictions': predictions,
                'logits': logits.cpu().numpy(),
                'uncertainty': uncertainty.cpu().numpy() if uncertainty is not None else None
            }
    
    def get_vocabulary_performance(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze performance across different vocabulary subsets.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Ground truth labels [batch_size, seq_len]
            
        Returns:
            Dictionary with vocabulary performance metrics
        """
        valid_mask = labels != self.ignore_index
        if not valid_mask.sum():
            return {}
        
        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Analyze performance by frequency bands (assuming labels represent token frequencies)
        unique_labels, label_counts = torch.unique(valid_labels, return_counts=True)
        
        # Sort by frequency
        sorted_indices = torch.argsort(label_counts, descending=True)
        unique_labels = unique_labels[sorted_indices]
        label_counts = label_counts[sorted_indices]
        
        # Split into frequency bands
        num_bands = 5
        band_size = len(unique_labels) // num_bands
        
        band_performance = {}
        for i in range(num_bands):
            start_idx = i * band_size
            end_idx = start_idx + band_size if i < num_bands - 1 else len(unique_labels)
            
            band_labels = unique_labels[start_idx:end_idx]
            band_mask = torch.isin(valid_labels, band_labels)
            
            if band_mask.sum() > 0:
                band_logits = valid_logits[band_mask]
                band_labels_actual = valid_labels[band_mask]
                
                # Compute accuracy for this band
                band_predictions = torch.argmax(band_logits, dim=-1)
                band_accuracy = (band_predictions == band_labels_actual).float().mean()
                
                band_performance[f'frequency_band_{i+1}'] = {
                    'accuracy': band_accuracy.item(),
                    'num_tokens': band_mask.sum().item(),
                    'vocab_size': len(band_labels)
                }
        
        return band_performance
