"""
Extended evaluation metrics for QEmbed evaluation system.

⚠️ CRITICAL: This class extends the existing QuantumMetrics class
    rather than duplicating functionality.

Provides standard NLP metrics and utility functions for
evaluating model performance across different tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import torch.nn.functional as F

# ⚠️ CRITICAL: Extend existing QuantumMetrics instead of replacing
# Note: Use absolute imports to avoid circular dependency issues
from qembed.utils.metrics import QuantumMetrics as BaseQuantumMetrics

class EvaluationMetrics(BaseQuantumMetrics):
    """
    Extended evaluation metrics that build upon existing QuantumMetrics.
    
    ⚠️ CRITICAL: Extends existing functionality rather than duplicating.
    """
    
    @staticmethod
    def classification_metrics(
        y_true: Union[torch.Tensor, np.ndarray, List],
        y_pred: Union[torch.Tensor, np.ndarray, List],
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for multi-class
            
        Returns:
            Dictionary of classification metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist()
        }
        
        return metrics
    
    @staticmethod
    def confusion_matrix_analysis(
        y_true: Union[torch.Tensor, np.ndarray, List],
        y_pred: Union[torch.Tensor, np.ndarray, List]
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Analyze confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with confusion matrix and derived metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Compute additional metrics
        total = np.sum(cm)
        correct = np.sum(np.diag(cm))
        incorrect = total - correct
        
        analysis = {
            'confusion_matrix': cm,
            'total_samples': total,
            'correct_predictions': correct,
            'incorrect_predictions': incorrect,
            'overall_accuracy': correct / total if total > 0 else 0
        }
        
        return analysis
    
    @staticmethod
    def mlm_metrics(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100
    ) -> Dict[str, float]:
        """
        Compute MLM-specific metrics.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: True labels [batch_size, seq_len]
            ignore_index: Index to ignore in loss computation
            
        Returns:
            Dictionary of MLM metrics
        """
        # Compute perplexity
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = torch.exp(loss)
        
        # Compute accuracy on non-ignored tokens
        valid_mask = labels != ignore_index
        if valid_mask.sum() > 0:
            valid_logits = logits[valid_mask]
            valid_labels = labels[valid_mask]
            predictions = torch.argmax(valid_logits, dim=-1)
            accuracy = (predictions == valid_labels).float().mean()
        else:
            accuracy = 0.0
        
        # Compute top-k accuracy
        top_1_accuracy = accuracy
        top_5_accuracy = EvaluationMetrics._compute_top_k_accuracy(
            valid_logits, valid_labels, k=5
        ) if valid_mask.sum() > 0 else 0.0
        
        return {
            'loss': loss.item(),
            'perplexity': perplexity.item(),
            'accuracy': accuracy.item(),
            'top_1_accuracy': top_1_accuracy.item(),
            'top_5_accuracy': top_5_accuracy.item()
        }
    
    @staticmethod
    def _compute_top_k_accuracy(
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        k: int
    ) -> float:
        """Compute top-k accuracy."""
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        correct = torch.any(top_k_indices == labels.unsqueeze(-1), dim=-1)
        return correct.float().mean()
    
    @staticmethod
    def embedding_metrics(
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute embedding quality metrics.
        
        Args:
            embeddings: Embedding vectors [batch_size, seq_len, embed_dim]
            labels: Optional labels for supervised metrics
            
        Returns:
            Dictionary of embedding metrics
        """
        # Compute embedding statistics
        mean_embedding = torch.mean(embeddings, dim=0)
        embedding_variance = torch.var(embeddings, dim=0)
        
        # Compute cosine similarity between embeddings
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(-2, -1))
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(similarity_matrix.size(-1), device=similarity_matrix.device)
        off_diagonal_similarities = similarity_matrix[~mask.bool()]
        
        metrics = {
            'mean_embedding_norm': torch.norm(mean_embedding).item(),
            'embedding_variance': embedding_variance.mean().item(),
            'mean_cosine_similarity': off_diagonal_similarities.mean().item(),
            'std_cosine_similarity': off_diagonal_similarities.std().item()
        }
        
        return metrics
    
    @staticmethod
    def statistical_analysis(
        values: Union[torch.Tensor, np.ndarray, List]
    ) -> Dict[str, float]:
        """
        Perform statistical analysis on a set of values.
        
        Args:
            values: Values to analyze
            
        Returns:
            Dictionary of statistical measures
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        elif isinstance(values, list):
            values = np.array(values)
        
        analysis = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
        }
        
        return analysis
