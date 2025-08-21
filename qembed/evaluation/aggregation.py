"""
Result aggregation utilities for QEmbed.

⚠️ CRITICAL: This aggregator ensures all results follow Phase 2
    EvaluationResult patterns while incorporating Phase 3 data.

Aggregates results from different phases and standardizes output format.
"""

import numpy as np
import time
from typing import Dict, Any, List

# ⚠️ CRITICAL: Import existing QEmbed infrastructure
from .base_evaluator import EvaluationResult

class ResultAggregator:
    """
    Aggregates results from different phases and standardizes output format.
    
    ⚠️ CRITICAL: This class ensures all results follow Phase 2
    EvaluationResult patterns while incorporating Phase 3 data.
    """
    
    def __init__(self):
        """Initialize the result aggregator."""
        self.aggregation_cache = {}
    
    def aggregate_evaluation_results(
        self,
        results: List[EvaluationResult],
        aggregation_strategy: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """
        Aggregate multiple evaluation results into comprehensive summary.
        
        Args:
            results: List of evaluation results to aggregate
            aggregation_strategy: Strategy for aggregation
            
        Returns:
            Aggregated results summary
        """
        if not results:
            return {}
        
        # Basic aggregation
        aggregated = {
            'total_evaluations': len(results),
            'evaluation_tasks': list(set(r.task_name for r in results)),
            'models_evaluated': list(set(r.model_name for r in results)),
            'aggregation_timestamp': time.time(),
            'aggregation_strategy': aggregation_strategy
        }
        
        # Metric aggregation
        aggregated['metrics_summary'] = self._aggregate_metrics(results)
        aggregated['quantum_metrics_summary'] = self._aggregate_quantum_metrics(results)
        aggregated['metadata_summary'] = self._aggregate_metadata(results)
        
        # Performance ranking
        aggregated['performance_ranking'] = self._compute_performance_ranking(results)
        
        return aggregated
    
    def _aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate standard metrics across results."""
        metric_keys = set()
        for result in results:
            metric_keys.update(result.metrics.keys())
        
        aggregated_metrics = {}
        for key in metric_keys:
            values = []
            for result in results:
                if key in result.metrics and isinstance(result.metrics[key], (int, float)):
                    values.append(result.metrics[key])
            
            if values:
                aggregated_metrics[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return aggregated_metrics
    
    def _aggregate_quantum_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate quantum metrics across results."""
        quantum_keys = set()
        for result in results:
            quantum_keys.update(result.quantum_metrics.keys())
        
        aggregated_quantum = {}
        for key in quantum_keys:
            values = []
            for result in results:
                if key in result.quantum_metrics and isinstance(result.quantum_metrics[key], (int, float)):
                    values.append(result.quantum_metrics[key])
            
            if values:
                aggregated_quantum[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return aggregated_quantum
    
    def _aggregate_metadata(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate metadata across results."""
        metadata_keys = set()
        for result in results:
            metadata_keys.update(result.metadata.keys())
        
        aggregated_metadata = {}
        for key in metadata_keys:
            values = []
            for result in results:
                if key in result.metadata:
                    values.append(result.metadata[key])
            
            if values:
                # Handle different metadata types
                if all(isinstance(v, (int, float)) for v in values):
                    aggregated_metadata[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
                else:
                    # For non-numeric metadata, just collect unique values
                    aggregated_metadata[key] = list(set(str(v) for v in values))
        
        return aggregated_metadata
    
    def _compute_performance_ranking(self, results: List[EvaluationResult]) -> Dict[str, List[str]]:
        """Compute performance ranking across different metrics."""
        ranking = {}
        
        # Get all metric keys
        metric_keys = set()
        for result in results:
            metric_keys.update(result.metrics.keys())
        
        for metric in metric_keys:
            # Filter results that have this metric
            valid_results = [(r, r.metrics.get(metric, 0)) for r in results 
                           if metric in r.metrics and isinstance(r.metrics[metric], (int, float))]
            
            if valid_results:
                # Sort by metric value (higher is better for most metrics)
                sorted_results = sorted(valid_results, key=lambda x: x[1], reverse=True)
                ranking[metric] = [r[0].model_name for r in sorted_results]
        
        return ranking
