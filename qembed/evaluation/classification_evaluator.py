"""
Classification task evaluator for QEmbed.

⚠️ CRITICAL: This evaluator integrates with existing QEmbed infrastructure
    and handles both quantum BERT and legacy quantum models.

Evaluates classification models with standard metrics and
quantum-specific analysis capabilities.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
from .base_evaluator import BaseEvaluator, EvaluationResult
from .evaluation_metrics import EvaluationMetrics

# ⚠️ CRITICAL: Import existing quantum BERT models for proper handling
from qembed.models.quantum_bert import QuantumBertForSequenceClassification

class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for classification tasks.
    
    Supports both single-label and multi-label classification
    with comprehensive metric computation and quantum analysis.
    
    ⚠️ CRITICAL: Extends BaseEvaluator and integrates with existing QuantumMetrics
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        device: Optional[str] = None,
        task_type: str = 'single_label'
    ):
        """
        Initialize classification evaluator.
        
        Args:
            model: Classification model to evaluate
            device: Device to run evaluation on
            task_type: Type of classification task ('single_label' or 'multi_label')
        """
        super().__init__(model, device)
        self.task_type = task_type
        
        # ⚠️ CRITICAL: Use existing QuantumMetrics instance from parent class
        # Don't create duplicate instances
        self.metrics_calculator = EvaluationMetrics()
    
    def evaluate(
        self, 
        dataloader: torch.utils.data.DataLoader,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate classification model.
        
        Args:
            dataloader: DataLoader containing evaluation data
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with classification metrics
        """
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        all_outputs = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self._evaluate_batch_internal(batch)
                
                # Collect results
                all_predictions.append(outputs['predictions'])
                all_labels.append(outputs['labels'])
                if 'uncertainty' in outputs and outputs['uncertainty'] is not None:
                    all_uncertainties.append(outputs['uncertainty'])
                all_outputs.append(outputs)
        
        # Concatenate results
        predictions = torch.cat(all_predictions, dim=0)
        labels = torch.cat(all_labels, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0) if all_uncertainties else None
        
        # Compute metrics
        metrics = self._compute_classification_metrics(predictions, labels)
        
        # Compute quantum metrics if available
        quantum_metrics = {}
        if uncertainties is not None:
            # ⚠️ CRITICAL: Use parent class method to avoid conflicts
            quantum_metrics = self.compute_quantum_metrics({'quantum_uncertainty': uncertainties})
        
        # Create result
        result = EvaluationResult(
            task_name=f"{self.task_type}_classification",
            model_name=self.model.__class__.__name__,
            metrics=metrics,
            quantum_metrics=quantum_metrics,
            metadata={
                'task_type': self.task_type,
                'num_samples': len(predictions),
                'num_classes': predictions.max().item() + 1 if self.task_type == 'single_label' else predictions.size(-1)
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
        
        ⚠️ CRITICAL: This method handles both quantum BERT and legacy models
        """
        # Extract batch components
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        token_type_ids = batch.get('token_type_ids')
        labels = batch.get('labels')
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        # Get predictions
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        if self.task_type == 'single_label':
            predictions = torch.argmax(logits, dim=-1)
        else:  # multi-label
            predictions = (torch.sigmoid(logits) > 0.5).long()
        
        # Get uncertainty if available
        uncertainty = None
        if hasattr(outputs, 'quantum_uncertainty'):
            uncertainty = outputs.quantum_uncertainty
        
        return {
            'predictions': predictions,
            'labels': labels,
            'logits': logits,
            'uncertainty': uncertainty
        }
    
    def _compute_classification_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute classification-specific metrics."""
        # Convert to numpy for sklearn metrics
        pred_np = predictions.cpu().numpy()
        label_np = labels.cpu().numpy()
        
        # ⚠️ CRITICAL: Use metrics calculator to avoid duplicating functionality
        metrics = self.metrics_calculator.classification_metrics(
            label_np, pred_np, average='weighted'
        )
        
        # Confusion matrix analysis
        cm_analysis = self.metrics_calculator.confusion_matrix_analysis(
            label_np, pred_np
        )
        
        # ⚠️ CRITICAL: Update metrics dict instead of replacing
        metrics.update({
            'confusion_matrix': cm_analysis['confusion_matrix'].tolist(),
            'total_samples': cm_analysis['total_samples'],
            'correct_predictions': cm_analysis['correct_predictions'],
            'incorrect_predictions': cm_analysis['incorrect_predictions'],
            'overall_accuracy': cm_analysis['overall_accuracy']
        })
        
        return metrics
    
    def get_class_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by class."""
        if not self.current_result:
            return {}
        
        # Extract per-class metrics
        per_class_metrics = {}
        if 'per_class_precision' in self.current_result.metrics:
            for i, (precision, recall, f1) in enumerate(zip(
                self.current_result.metrics['per_class_precision'],
                self.current_result.metrics['per_class_recall'],
                self.current_result.metrics['per_class_f1']
            )):
                per_class_metrics[f'class_{i}'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
        
        return per_class_metrics
    
    def evaluate_single_sample(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample for real-time inference.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            token_type_ids: Optional token type IDs
            
        Returns:
            Dictionary with prediction and uncertainty
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare input
            inputs = {'input_ids': input_ids.to(self.device)}
            if attention_mask is not None:
                inputs['attention_mask'] = attention_mask.to(self.device)
            if token_type_ids is not None:
                inputs['token_type_ids'] = token_type_ids.to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Get prediction
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            if self.task_type == 'single_label':
                prediction = torch.argmax(logits, dim=-1)
                confidence = torch.softmax(logits, dim=-1).max(dim=-1)[0]
            else:
                prediction = (torch.sigmoid(logits) > 0.5).long()
                confidence = torch.sigmoid(logits).max(dim=-1)[0]
            
            # Get uncertainty if available
            uncertainty = None
            if hasattr(outputs, 'quantum_uncertainty'):
                uncertainty = outputs.quantum_uncertainty
            
            return {
                'prediction': prediction.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'uncertainty': uncertainty.cpu().numpy() if uncertainty is not None else None,
                'logits': logits.cpu().numpy()
            }
