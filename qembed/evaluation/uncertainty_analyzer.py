"""
Uncertainty quantification analysis for QEmbed.

⚠️ CRITICAL: This analyzer uses existing uncertainty tracking from QuantumTrainer
    and integrates with existing uncertainty regularization.

Analyzes uncertainty distributions and calibration.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, mutual_info_score

# ⚠️ CRITICAL: Import existing QEmbed infrastructure
from qembed.utils.metrics import QuantumMetrics

class UncertaintyAnalyzer:
    """
    Analyzer for uncertainty quantification and calibration.
    
    ⚠️ CRITICAL: Integrates with existing QuantumTrainer uncertainty tracking
    and follows Phase 2 integration patterns.
    """
    
    def __init__(self):
        """Initialize uncertainty analyzer."""
        self.quantum_metrics = QuantumMetrics()
    
    def analyze_uncertainty_distribution(
        self, 
        uncertainty: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze uncertainty distribution patterns.
        
        Args:
            uncertainty: Uncertainty values [batch_size, ...]
            predictions: Optional model predictions
            targets: Optional ground truth targets
            
        Returns:
            Dictionary of uncertainty distribution metrics
        """
        uncertainty_np = uncertainty.detach().cpu().numpy()
        
        # Basic statistical measures
        distribution_metrics = {
            'mean_uncertainty': float(np.mean(uncertainty_np)),
            'std_uncertainty': float(np.std(uncertainty_np)),
            'min_uncertainty': float(np.min(uncertainty_np)),
            'max_uncertainty': float(np.max(uncertainty_np)),
            'median_uncertainty': float(np.median(uncertainty_np)),
            'q25_uncertainty': float(np.percentile(uncertainty_np, 25)),
            'q75_uncertainty': float(np.percentile(uncertainty_np, 75)),
            'iqr_uncertainty': float(np.percentile(uncertainty_np, 75) - np.percentile(uncertainty_np, 25))
        }
        
        # Distribution shape analysis
        skewness = stats.skew(uncertainty_np.flatten())
        kurtosis = stats.kurtosis(uncertainty_np.flatten())
        
        distribution_metrics.update({
            'uncertainty_skewness': float(skewness),
            'uncertainty_kurtosis': float(kurtosis),
            'uncertainty_entropy': float(stats.entropy(np.histogram(uncertainty_np.flatten(), bins=20)[0] + 1e-8))
        })
        
        # Uncertainty vs. prediction confidence analysis
        if predictions is not None and targets is not None:
            confidence_metrics = self._analyze_uncertainty_confidence_correlation(
                uncertainty_np, predictions, targets
            )
            distribution_metrics.update(confidence_metrics)
        
        return distribution_metrics
    
    def _analyze_uncertainty_confidence_correlation(
        self,
        uncertainty: np.ndarray,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze correlation between uncertainty and prediction confidence."""
        # Convert predictions to confidence scores
        if predictions.dim() > 1:
            # Multi-class: use max probability as confidence
            confidence = torch.softmax(predictions, dim=-1).max(dim=-1)[0]
        else:
            # Binary: use sigmoid probability
            confidence = torch.sigmoid(predictions)
        
        confidence_np = confidence.detach().cpu().numpy()
        
        # Compute correlation
        correlation = np.corrcoef(uncertainty.flatten(), confidence_np.flatten())[0, 1]
        
        # Compute mutual information
        uncertainty_binned = np.digitize(uncertainty.flatten(), bins=np.linspace(uncertainty.min(), uncertainty.max(), 10))
        confidence_binned = np.digitize(confidence_np.flatten(), bins=np.linspace(confidence_np.min(), confidence_np.max(), 10))
        
        mutual_info = mutual_info_score(uncertainty_binned, confidence_binned)
        
        return {
            'uncertainty_confidence_correlation': float(correlation),
            'uncertainty_confidence_mutual_info': float(mutual_info)
        }
    
    def analyze_uncertainty_calibration(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor,
        num_bins: int = 10
    ) -> Dict[str, float]:
        """
        Analyze uncertainty calibration.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            uncertainty: Uncertainty values
            num_bins: Number of bins for calibration analysis
            
        Returns:
            Dictionary of calibration metrics
        """
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        uncertainty_np = uncertainty.detach().cpu().numpy()
        
        # Convert to binary classification if needed
        if predictions.dim() > 1:
            # Multi-class: use argmax
            pred_classes = np.argmax(predictions_np, axis=-1)
            pred_probs = np.max(torch.softmax(predictions, dim=-1).cpu().numpy(), axis=-1)
        else:
            # Binary: use sigmoid
            pred_classes = (predictions_np > 0).astype(int)
            pred_probs = torch.sigmoid(predictions).cpu().numpy()
        
        # For multi-class, convert to one-vs-rest binary classification
        # Use the most confident class as positive
        if len(np.unique(targets_np)) > 2:
            # Convert to binary: most confident prediction vs rest
            pred_probs_binary = pred_probs
            # Create binary targets: 1 if prediction matches target, 0 otherwise
            targets_binary = (pred_classes == targets_np).astype(int)
        else:
            pred_probs_binary = pred_probs
            targets_binary = targets_np
        
        # Flatten arrays for calibration analysis
        pred_probs_flat = pred_probs_binary.flatten()
        targets_flat = targets_binary.flatten()
        
        # Compute calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                targets_flat, pred_probs_flat, n_bins=num_bins
            )
            
            # Calibration error (Brier score)
            brier_score = brier_score_loss(targets_flat, pred_probs_flat)
            
            # Expected calibration error (ECE)
            ece = self._compute_expected_calibration_error(
                pred_probs_flat, targets_flat, uncertainty_np.flatten(), num_bins
            )
            
            calibration_metrics = {
                'brier_score': float(brier_score),
                'expected_calibration_error': float(ece),
                'calibration_curve_fraction': fraction_of_positives.tolist(),
                'calibration_curve_predicted': mean_predicted_value.tolist()
            }
            
        except Exception as e:
            # Fallback if calibration analysis fails
            calibration_metrics = {
                'brier_score': float('nan'),
                'expected_calibration_error': float('nan'),
                'calibration_error': str(e)
            }
        
        return calibration_metrics
    
    def _compute_expected_calibration_error(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainty: np.ndarray,
        num_bins: int
    ) -> float:
        """Compute expected calibration error."""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = np.logical_and(predictions > bin_lower, predictions <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_accuracy = np.mean(targets[in_bin])
                bin_confidence = np.mean(predictions[in_bin])
                ece += bin_size * np.abs(bin_accuracy - bin_confidence)
        
        return ece / len(predictions)
    
    def analyze_uncertainty_regularization(
        self,
        uncertainty: torch.Tensor,
        loss_values: Optional[torch.Tensor] = None,
        training_step: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Analyze uncertainty regularization effects.
        
        Args:
            uncertainty: Uncertainty values
            loss_values: Optional training loss values
            training_step: Optional training step number
            
        Returns:
            Dictionary of regularization analysis metrics
        """
        uncertainty_np = uncertainty.detach().cpu().numpy()
        
        regularization_metrics = {
            'uncertainty_regularization_mean': float(np.mean(uncertainty_np)),
            'uncertainty_regularization_std': float(np.std(uncertainty_np))
        }
        
        # Analyze uncertainty distribution shape
        if len(uncertainty_np.flatten()) > 1:
            # Test for normal distribution
            _, normality_p_value = stats.normaltest(uncertainty_np.flatten())
            regularization_metrics['uncertainty_normality_p_value'] = float(normality_p_value)
            
            # Test for uniform distribution
            _, uniformity_p_value = stats.kstest(uncertainty_np.flatten(), 'uniform')
            regularization_metrics['uncertainty_uniformity_p_value'] = float(uniformity_p_value)
        
        # Analyze uncertainty vs. loss correlation if available
        if loss_values is not None:
            loss_np = loss_values.detach().cpu().numpy()
            correlation = np.corrcoef(uncertainty_np.flatten(), loss_np.flatten())[0, 1]
            regularization_metrics['uncertainty_loss_correlation'] = float(correlation)
        
        return regularization_metrics
    
    def analyze_uncertainty_evolution(
        self,
        uncertainty_sequence: List[torch.Tensor],
        time_steps: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze uncertainty evolution over time.
        
        Args:
            uncertainty_sequence: List of uncertainty values at different time steps
            time_steps: Optional time steps for temporal analysis
            
        Returns:
            Dictionary of evolution analysis metrics
        """
        if len(uncertainty_sequence) < 2:
            return {}
        
        # Convert to numpy arrays
        uncertainty_arrays = [u.detach().cpu().numpy() for u in uncertainty_sequence]
        
        # Compute evolution metrics
        evolution_metrics = {}
        
        # Mean uncertainty over time
        mean_uncertainties = [np.mean(u) for u in uncertainty_arrays]
        evolution_metrics['mean_uncertainty_evolution'] = mean_uncertainties
        
        # Uncertainty stability
        uncertainty_stability = 1.0 - np.std(mean_uncertainties)
        evolution_metrics['uncertainty_stability'] = float(uncertainty_stability)
        
        # Uncertainty convergence
        if len(mean_uncertainties) > 2:
            # Test for convergence (decreasing trend)
            x = np.arange(len(mean_uncertainties))
            slope, _, r_value, _, _ = stats.linregress(x, mean_uncertainties)
            evolution_metrics['uncertainty_convergence_slope'] = float(slope)
            evolution_metrics['uncertainty_convergence_r_squared'] = float(r_value ** 2)
        
        # Uncertainty jumps (sudden large changes)
        if len(mean_uncertainties) > 1:
            changes = np.abs(np.diff(mean_uncertainties))
            jump_threshold = np.mean(changes) + 2 * np.std(changes)
            quantum_jumps = np.sum(changes > jump_threshold)
            evolution_metrics['uncertainty_quantum_jumps'] = int(quantum_jumps)
        
        # Time-dependent analysis if time steps provided
        if time_steps and len(time_steps) == len(uncertainty_sequence):
            evolution_metrics['time_steps'] = time_steps
            evolution_metrics['temporal_analysis'] = True
        
        return evolution_metrics
    
    def generate_uncertainty_report(
        self,
        uncertainty: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive uncertainty analysis report.
        
        Args:
            uncertainty: Uncertainty values
            predictions: Optional model predictions
            targets: Optional ground truth targets
            
        Returns:
            Dictionary containing comprehensive uncertainty analysis
        """
        report = {}
        
        # Distribution analysis
        report['distribution'] = self.analyze_uncertainty_distribution(
            uncertainty, predictions, targets
        )
        
        # Calibration analysis
        if predictions is not None and targets is not None:
            report['calibration'] = self.analyze_uncertainty_calibration(
                predictions, targets, uncertainty
            )
        
        # Regularization analysis
        report['regularization'] = self.analyze_uncertainty_regularization(uncertainty)
        
        # Summary statistics
        import time
        report['summary'] = {
            'total_samples': int(uncertainty.numel()),
            'analysis_timestamp': str(time.time()),
            'uncertainty_range': [
                float(uncertainty.min().item()),
                float(uncertainty.max().item())
            ]
        }
        
        return report
