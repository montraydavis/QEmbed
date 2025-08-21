"""
Base evaluator class for QEmbed evaluation system.

⚠️ CRITICAL: This class integrates with existing QEmbed infrastructure
    rather than duplicating functionality.

Provides common evaluation functionality for all task-specific evaluators
including result storage, reporting, and quantum-specific hooks.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import json
import time
from pathlib import Path

# ⚠️ CRITICAL: Import existing QEmbed infrastructure
# Note: Use absolute imports to avoid circular dependency issues
from qembed.utils.metrics import QuantumMetrics

# ⚠️ CRITICAL: Avoid circular import - don't import QuantumTrainer here
# Instead, pass trainer instance as parameter or use lazy imports

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    task_name: str
    model_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    quantum_metrics: Dict[str, float] = field(default_factory=dict)
    uncertainty_analysis: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    ⚠️ CRITICAL: Integrates with existing QuantumMetrics and QuantumTrainer
    rather than duplicating functionality.
    
    Provides common evaluation functionality and defines
    the interface that all evaluators must implement.
    """
    
    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        """
        Initialize base evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # ⚠️ CRITICAL: Use existing QuantumMetrics instead of duplicating
        self.quantum_metrics = QuantumMetrics()
        
        # ⚠️ CRITICAL: Avoid circular imports - don't store QuantumTrainer reference
        # Instead, pass trainer instance when needed for evaluation
        
        # Results storage
        self.results: List[EvaluationResult] = []
        self.current_result: Optional[EvaluationResult] = None
        
        # Configuration
        self.include_quantum_analysis = True
        self.save_intermediate_results = True
    
    @abstractmethod
    def evaluate(
        self, 
        data: Any, 
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate the model on given data.
        
        Args:
            data: Data to evaluate on
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult containing all metrics
        """
        pass
    
    def evaluate_batch(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate a single batch of data.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of batch-level results
        """
        with torch.no_grad():
            return self._evaluate_batch_internal(batch)
    
    @abstractmethod
    def _evaluate_batch_internal(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Internal batch evaluation method to be implemented by subclasses."""
        pass
    
    def compute_quantum_metrics(
        self, 
        outputs: Any
    ) -> Dict[str, float]:
        """
        Compute quantum-specific metrics.
        
        Args:
            outputs: Model outputs
            
        Returns:
            Dictionary of quantum metrics
        """
        if not self.include_quantum_analysis:
            return {}
        
        quantum_metrics = {}
        
        # Uncertainty analysis
        if hasattr(outputs, 'quantum_uncertainty') and outputs.quantum_uncertainty is not None:
            uncertainty = outputs.quantum_uncertainty
            quantum_metrics.update(self._analyze_uncertainty(uncertainty))
        
        # Superposition analysis
        if hasattr(outputs, 'last_hidden_state'):
            quantum_metrics.update(self._analyze_superposition(outputs.last_hidden_state))
        
        return quantum_metrics
    
    def _analyze_uncertainty(self, uncertainty: torch.Tensor) -> Dict[str, float]:
        """Analyze uncertainty patterns."""
        return {
            'mean_uncertainty': uncertainty.mean().item(),
            'std_uncertainty': uncertainty.std().item(),
            'max_uncertainty': uncertainty.max().item(),
            'min_uncertainty': uncertainty.min().item(),
            'uncertainty_entropy': self._compute_entropy(uncertainty)
        }
    
    def _analyze_superposition(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """Analyze superposition state characteristics."""
        # Compute variance across sequence dimension as superposition measure
        variance = torch.var(hidden_states, dim=1)
        return {
            'mean_superposition': variance.mean().item(),
            'superposition_entropy': self._compute_entropy(variance)
        }
    
    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Compute entropy of a tensor."""
        # Convert to probabilities and compute entropy
        probs = torch.softmax(tensor.flatten(), dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item()
    
    def save_results(self, filepath: Union[str, Path]):
        """Save evaluation results to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'results': [result.__dict__ for result in self.results],
            'metadata': {
                'model_name': self.model.__class__.__name__,
                'total_evaluations': len(self.results),
                'timestamp': time.time()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
    
    def load_results(self, filepath: Union[str, Path]):
        """Load evaluation results from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct results
        self.results = []
        for result_dict in data['results']:
            result = EvaluationResult(**result_dict)
            self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation results."""
        if not self.results:
            return {}
        
        summary = {
            'total_evaluations': len(self.results),
            'average_metrics': {},
            'best_performance': {},
            'quantum_analysis': {}
        }
        
        # Compute average metrics across all results
        metric_keys = set()
        for result in self.results:
            metric_keys.update(result.metrics.keys())
        
        for key in metric_keys:
            values = [result.metrics.get(key, 0) for result in self.results]
            summary['average_metrics'][key] = np.mean(values)
            summary['best_performance'][key] = max(values)
        
        return summary
