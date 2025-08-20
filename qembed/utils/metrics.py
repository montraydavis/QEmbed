"""
Quantum-specific metrics for evaluating quantum-enhanced models.

This module provides various metrics and evaluation functions
specifically designed for quantum-inspired models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, average_precision_score


class QuantumMetrics:
    """
    Metrics for evaluating quantum-enhanced models.
    
    Provides various evaluation metrics including quantum-specific
    measures like superposition quality, entanglement strength,
    and uncertainty calibration.
    """
    
    def __init__(self):
        """Initialize quantum metrics."""
        pass
    
    def compute_metrics(
        self,
        model: torch.nn.Module,
        test_data: Optional[Dict[str, torch.Tensor]] = None,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics for quantum model.
        
        Args:
            model: Quantum model to evaluate
            test_data: Test data dictionary
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Basic task metrics
        if predictions is not None and targets is not None:
            task_metrics = self._compute_task_metrics(predictions, targets)
            metrics.update(task_metrics)
        
        # Quantum-specific metrics
        if hasattr(model, 'get_uncertainty'):
            quantum_metrics = self._compute_quantum_metrics(model, test_data)
            metrics.update(quantum_metrics)
        
        # Model complexity metrics
        complexity_metrics = self._compute_complexity_metrics(model)
        metrics.update(complexity_metrics)
        
        return metrics
    
    def _compute_task_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute basic task performance metrics."""
        metrics = {}
        
        # Convert to numpy for sklearn metrics
        if isinstance(predictions, torch.Tensor):
            pred_np = predictions.detach().cpu().numpy()
        else:
            pred_np = predictions
        
        if isinstance(targets, torch.Tensor):
            target_np = targets.detach().cpu().numpy()
        else:
            target_np = targets
        
        # Handle different prediction formats
        if pred_np.ndim == 3:  # [batch, seq_len, vocab_size]
            pred_labels = np.argmax(pred_np, axis=-1)
        else:
            pred_labels = pred_np
        
        # Flatten for metrics
        pred_flat = pred_labels.flatten()
        target_flat = target_np.flatten()
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(target_flat, pred_flat)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_flat, pred_flat, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # ROC AUC (for binary classification)
        if len(np.unique(target_flat)) == 2:
            try:
                if pred_np.ndim == 3:
                    # Use softmax probabilities for ROC AUC
                    prob_positive = F.softmax(torch.tensor(pred_np), dim=-1)[:, :, 1].numpy()
                    prob_flat = prob_positive.flatten()
                else:
                    prob_flat = pred_np.flatten()
                
                roc_auc = roc_auc_score(target_flat, prob_flat)
                metrics['roc_auc'] = roc_auc
            except:
                metrics['roc_auc'] = np.nan
        
        return metrics
    
    def _compute_quantum_metrics(
        self,
        model: torch.nn.Module,
        test_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """Compute quantum-specific metrics."""
        metrics = {}
        
        # Get uncertainty estimates
        if test_data is not None and 'input_ids' in test_data:
            input_ids = test_data['input_ids']
            uncertainty = model.get_uncertainty(input_ids)
            
            # Uncertainty statistics
            metrics['uncertainty_mean'] = uncertainty.mean().item()
            metrics['uncertainty_std'] = uncertainty.std().item()
            metrics['uncertainty_min'] = uncertainty.min().item()
            metrics['uncertainty_max'] = uncertainty.max().item()
            
            # Uncertainty distribution metrics
            uncertainty_flat = uncertainty.flatten()
            metrics['uncertainty_entropy'] = self._compute_entropy(uncertainty_flat)
            metrics['uncertainty_gini'] = self._compute_gini_coefficient(uncertainty_flat)
        
        # Superposition quality metrics
        if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'quantum_embeddings'):
            superposition_metrics = self._compute_superposition_metrics(model)
            metrics.update(superposition_metrics)
        
        # Entanglement metrics
        if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'quantum_embeddings'):
            entanglement_metrics = self._compute_entanglement_metrics(model)
            metrics.update(entanglement_metrics)
        
        return metrics
    
    def _compute_superposition_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute superposition quality metrics."""
        metrics = {}
        
        try:
            quantum_embeddings = model.embeddings.quantum_embeddings
            
            # Get superposition matrix
            if hasattr(quantum_embeddings, 'superposition_matrix'):
                superposition_matrix = quantum_embeddings.superposition_matrix.data
                
                # Orthogonality of superposition states
                orthogonality = self._compute_orthogonality(superposition_matrix)
                metrics['superposition_orthogonality'] = orthogonality
                
                # Superposition strength
                identity = torch.eye(superposition_matrix.size(0), device=superposition_matrix.device)
                superposition_strength = torch.norm(superposition_matrix - identity).item()
                metrics['superposition_strength'] = superposition_strength
                
                # Superposition diversity
                diversity = self._compute_diversity(superposition_matrix)
                metrics['superposition_diversity'] = diversity
            
            # State variance metrics
            if hasattr(quantum_embeddings, 'state_embeddings'):
                state_embeddings = quantum_embeddings.state_embeddings.data
                
                # Compute variance across states for each token
                state_variance = torch.var(state_embeddings, dim=1)
                avg_variance = torch.mean(state_variance).item()
                metrics['state_variance_mean'] = avg_variance
                
                # State separation
                separation = self._compute_state_separation(state_embeddings)
                metrics['state_separation'] = separation
                
        except Exception as e:
            # If any error occurs, set metrics to NaN
            metrics['superposition_orthogonality'] = np.nan
            metrics['superposition_strength'] = np.nan
            metrics['superposition_diversity'] = np.nan
            metrics['state_variance_mean'] = np.nan
            metrics['state_separation'] = np.nan
        
        return metrics
    
    def _compute_entanglement_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute entanglement correlation metrics."""
        metrics = {}
        
        try:
            # This would require access to entanglement matrices
            # For now, return placeholder metrics
            metrics['entanglement_strength'] = np.nan
            metrics['entanglement_correlation'] = np.nan
            metrics['entanglement_entropy'] = np.nan
            
        except Exception as e:
            metrics['entanglement_strength'] = np.nan
            metrics['entanglement_correlation'] = np.nan
            metrics['entanglement_entropy'] = np.nan
        
        return metrics
    
    def _compute_complexity_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute model complexity metrics."""
        metrics = {}
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metrics['total_parameters'] = total_params
        metrics['trainable_parameters'] = trainable_params
        
        # Model depth (approximate)
        if hasattr(model, 'encoder_layers'):
            metrics['model_depth'] = len(model.encoder_layers)
        elif hasattr(model, 'layers'):
            metrics['model_depth'] = len(model.layers)
        else:
            metrics['model_depth'] = np.nan
        
        # Embedding dimension
        if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'hidden_size'):
            metrics['embedding_dim'] = model.embeddings.hidden_size
        elif hasattr(model, 'embeddings') and hasattr(model.embeddings, 'd_model'):
            metrics['embedding_dim'] = model.embeddings.d_model
        else:
            metrics['embedding_dim'] = np.nan
        
        return metrics
    
    def _compute_entropy(self, values: torch.Tensor) -> float:
        """Compute entropy of a distribution."""
        # Normalize to probabilities
        values_norm = F.softmax(values, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(values_norm * torch.log(values_norm + 1e-8))
        return entropy.item()
    
    def _compute_gini_coefficient(self, values: torch.Tensor) -> float:
        """Compute Gini coefficient of inequality."""
        # Sort values
        sorted_values, _ = torch.sort(values)
        n = len(sorted_values)
        
        # Compute Gini coefficient
        cumsum = torch.cumsum(sorted_values, dim=0)
        gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
        return gini.item()
    
    def _compute_orthogonality(self, matrix: torch.Tensor) -> float:
        """Compute orthogonality of matrix columns."""
        # Normalize columns
        matrix_norm = F.normalize(matrix, p=2, dim=0)
        
        # Compute correlation matrix
        correlation = torch.mm(matrix_norm.t(), matrix_norm)
        
        # Remove diagonal
        mask = torch.eye(correlation.size(0), device=correlation.device)
        off_diagonal = correlation * (1 - mask)
        
        # Compute average off-diagonal correlation
        avg_correlation = torch.mean(torch.abs(off_diagonal))
        
        # Orthogonality is 1 - average correlation
        orthogonality = 1 - avg_correlation.item()
        return orthogonality
    
    def _compute_diversity(self, matrix: torch.Tensor) -> float:
        """Compute diversity of matrix elements."""
        # Compute standard deviation of matrix elements
        diversity = torch.std(matrix).item()
        return diversity
    
    def _compute_state_separation(self, state_embeddings: torch.Tensor) -> float:
        """Compute separation between quantum states."""
        # Compute pairwise distances between states
        states_flat = state_embeddings.view(-1, state_embeddings.size(-1))
        
        # Compute cosine distances
        states_norm = F.normalize(states_flat, p=2, dim=-1)
        distances = 1 - torch.mm(states_norm, states_norm.t())
        
        # Remove diagonal
        mask = torch.eye(distances.size(0), device=distances.device)
        off_diagonal_distances = distances * (1 - mask)
        
        # Compute average separation
        separation = torch.mean(off_diagonal_distances).item()
        return separation
    
    def compute_uncertainty_calibration(
        self,
        uncertainty: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_bins: int = 10
    ) -> Dict[str, float]:
        """
        Compute uncertainty calibration metrics.
        
        Args:
            uncertainty: Uncertainty estimates
            predictions: Model predictions
            targets: Ground truth targets
            num_bins: Number of calibration bins
            
        Returns:
            Dictionary of calibration metrics
        """
        # Convert to numpy
        if isinstance(uncertainty, torch.Tensor):
            unc_np = uncertainty.detach().cpu().numpy()
        else:
            unc_np = uncertainty
        
        if isinstance(predictions, torch.Tensor):
            pred_np = predictions.detach().cpu().numpy()
        else:
            pred_np = predictions
        
        if isinstance(targets, torch.Tensor):
            target_np = targets.detach().cpu().numpy()
        else:
            target_np = targets
        
        # Handle different prediction formats
        if pred_np.ndim == 3:
            pred_labels = np.argmax(pred_np, axis=-1)
        else:
            pred_labels = pred_np
        
        # Flatten
        unc_flat = unc_np.flatten()
        pred_flat = pred_labels.flatten()
        target_flat = target_np.flatten()
        
        # Create calibration bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Compute calibration metrics
        calibration_error = 0.0
        reliability = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (unc_flat >= bin_lower) & (unc_flat < bin_upper)
            
            if np.sum(in_bin) > 0:
                # Predicted probability for this bin
                pred_prob = (bin_lower + bin_upper) / 2
                
                # Actual accuracy in this bin
                actual_acc = np.mean(pred_flat[in_bin] == target_flat[in_bin])
                
                # Calibration error
                bin_error = np.abs(pred_prob - actual_acc)
                calibration_error += bin_error * np.sum(in_bin)
                
                # Reliability
                reliability += np.sum(in_bin) * actual_acc
        
        # Normalize
        total_samples = len(unc_flat)
        calibration_error /= total_samples
        reliability /= total_samples
        
        # Expected calibration error (ECE)
        ece = calibration_error
        
        # Maximum calibration error (MCE)
        mce = np.max(np.abs(bin_boundaries[:-1] - np.array([
            np.mean(pred_flat[(unc_flat >= bin_boundaries[i]) & (unc_flat < bin_boundaries[i+1])] == 
                   target_flat[(unc_flat >= bin_boundaries[i]) & (unc_flat < bin_boundaries[i+1])])
            for i in range(num_bins)
        ])))
        
        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'reliability': reliability,
            'calibration_error': calibration_error
        }
    
    def compute_quantum_fidelity(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor
    ) -> float:
        """
        Compute quantum fidelity between two states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Fidelity value
        """
        # Normalize states
        state1_norm = F.normalize(state1, p=2, dim=-1)
        state2_norm = F.normalize(state2, p=2, dim=-1)
        
        # Compute overlap
        overlap = torch.abs(torch.sum(state1_norm * state2_norm, dim=-1))
        
        # Fidelity is the square of the overlap
        fidelity = torch.mean(overlap ** 2).item()
        
        return fidelity
    
    def generate_metrics_report(
        self,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a formatted metrics report.
        
        Args:
            metrics: Dictionary of computed metrics
            save_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        report = "=" * 60 + "\n"
        report += "QUANTUM MODEL EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Task Performance
        report += "TASK PERFORMANCE:\n"
        report += "-" * 20 + "\n"
        task_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in task_metrics:
            if metric in metrics:
                report += f"{metric.replace('_', ' ').title()}: {metrics[metric]:.4f}\n"
        report += "\n"
        
        # Quantum Metrics
        report += "QUANTUM METRICS:\n"
        report += "-" * 20 + "\n"
        quantum_metrics = ['uncertainty_mean', 'uncertainty_std', 'superposition_orthogonality', 
                          'superposition_strength', 'state_variance_mean']
        for metric in quantum_metrics:
            if metric in metrics:
                report += f"{metric.replace('_', ' ').title()}: {metrics[metric]:.4f}\n"
        report += "\n"
        
        # Model Complexity
        report += "MODEL COMPLEXITY:\n"
        report += "-" * 20 + "\n"
        complexity_metrics = ['total_parameters', 'trainable_parameters', 'model_depth', 'embedding_dim']
        for metric in complexity_metrics:
            if metric in metrics:
                if metric.endswith('_parameters'):
                    value = f"{metrics[metric]:,}"
                else:
                    value = f"{metrics[metric]:.2f}"
                report += f"{metric.replace('_', ' ').title()}: {value}\n"
        
        report += "\n" + "=" * 60
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
