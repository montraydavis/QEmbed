"""
Evaluation pipeline for QEmbed.

⚠️ CRITICAL: This pipeline integrates with existing QuantumTrainer
    evaluation loops rather than building parallel systems.

⚠️ CRITICAL: Must bridge Phase 2 evaluators with Phase 3 analyzers
    while maintaining seamless integration and hiding complexity.

Orchestrates comprehensive evaluation across multiple
models, tasks, and evaluation strategies.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable, Type
from pathlib import Path
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging

# ⚠️ CRITICAL: Import existing QEmbed infrastructure
# Note: Use absolute imports to avoid circular dependency issues
from qembed.training.quantum_trainer import QuantumTrainer

# ⚠️ CRITICAL: Import Phase 2 evaluators and Phase 3 analyzers
from .base_evaluator import BaseEvaluator, EvaluationResult
from .classification_evaluator import ClassificationEvaluator
from .mlm_evaluator import MLMEvaluator
from .embedding_evaluator import EmbeddingEvaluator
from .quantum_evaluation import QuantumEvaluation
from .uncertainty_analyzer import UncertaintyAnalyzer
from .superposition_analyzer import SuperpositionAnalyzer
from .aggregation import ResultAggregator
from .reporting import EvaluationReporter

class EvaluationCoordinator:
    """
    Bridge class that coordinates Phase 2 evaluators with Phase 3 analyzers.
    
    ⚠️ CRITICAL: This class reconciles the different integration patterns
    from Phase 2 (self.metrics_calculator) and Phase 3 (self.quantum_metrics).
    """
    
    def __init__(self):
        """Initialize the evaluation coordinator."""
        # Phase 3 analyzers
        self.quantum_evaluator = QuantumEvaluation()
        self.uncertainty_analyzer = UncertaintyAnalyzer()
        self.superposition_analyzer = SuperpositionAnalyzer()
        
        # Integration state
        self.analysis_cache = {}
    
    def enhance_evaluation_result(
        self,
        base_result: EvaluationResult,
        model_outputs: Any,
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> EvaluationResult:
        """
        Enhance Phase 2 evaluation result with Phase 3 quantum analysis.
        
        Args:
            base_result: Phase 2 evaluation result
            model_outputs: Model outputs for quantum analysis
            embeddings: Optional embeddings for superposition analysis
            attention_mask: Optional attention mask
            
        Returns:
            Enhanced EvaluationResult with Phase 3 quantum metrics
        """
        enhanced_result = EvaluationResult(
            task_name=base_result.task_name,
            model_name=base_result.model_name,
            metrics=base_result.metrics.copy(),
            quantum_metrics=base_result.quantum_metrics.copy(),
            uncertainty_analysis=base_result.uncertainty_analysis.copy(),
            timestamp=base_result.timestamp,
            metadata=base_result.metadata.copy()
        )
        
        # Add Phase 3 quantum analysis
        quantum_enhancements = self._compute_quantum_enhancements(
            model_outputs, embeddings, attention_mask
        )
        
        # Merge quantum metrics
        enhanced_result.quantum_metrics.update(quantum_enhancements)
        
        # Add quantum analysis metadata
        enhanced_result.metadata['phase_3_analysis'] = True
        enhanced_result.metadata['quantum_analysis_timestamp'] = time.time()
        
        return enhanced_result
    
    def _compute_quantum_enhancements(
        self,
        model_outputs: Any,
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute quantum enhancements using Phase 3 analyzers."""
        enhancements = {}
        
        # Quantum coherence analysis if embeddings available
        if embeddings is not None:
            coherence_metrics = self.quantum_evaluator.coherence_metrics(
                embeddings, attention_mask
            )
            enhancements.update(coherence_metrics)
        
        # Uncertainty analysis if available
        if hasattr(model_outputs, 'quantum_uncertainty') and model_outputs.quantum_uncertainty is not None:
            uncertainty_metrics = self.uncertainty_analyzer.analyze_uncertainty_distribution(
                model_outputs.quantum_uncertainty
            )
            enhancements.update(uncertainty_metrics)
        
        # Superposition analysis if embeddings available
        if embeddings is not None:
            superposition_metrics = self.superposition_analyzer.analyze_superposition_states(
                embeddings, attention_mask
            )
            enhancements.update(superposition_metrics)
        
        return enhancements

class EvaluationPipeline:
    """
    Orchestrates comprehensive evaluation workflows.
    
    ⚠️ CRITICAL: This pipeline coordinates all three phases:
    - Phase 1: Infrastructure (BaseEvaluator, EvaluationResult)
    - Phase 2: Task-specific evaluators
    - Phase 3: Quantum-specific analyzers
    
    Supports parallel evaluation, result caching, and
    automated evaluation across multiple configurations.
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        output_dir: Union[str, Path] = "evaluation_results"
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            base_config: Base configuration for evaluation
            output_dir: Directory to save results
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ⚠️ CRITICAL: Initialize bridge components for three-phase integration
        self.coordinator = EvaluationCoordinator()
        self.aggregator = ResultAggregator()
        
        # Pipeline state
        self.evaluations: List[Dict[str, Any]] = []
        self.results: List[EvaluationResult] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_evaluation(
        self,
        name: str,
        model: torch.nn.Module,
        evaluator_class: Type[BaseEvaluator],
        data: Any,
        config: Optional[Dict[str, Any]] = None,
        enable_quantum_analysis: bool = True
    ):
        """
        Add an evaluation to the pipeline.
        
        Args:
            name: Name of the evaluation
            model: Model to evaluate
            evaluator_class: Class of evaluator to use (must extend BaseEvaluator)
            data: Data for evaluation
            config: Evaluation-specific configuration
            enable_quantum_analysis: Whether to enable Phase 3 quantum analysis
        """
        evaluation_config = {
            'name': name,
            'model': model,
            'evaluator_class': evaluator_class,
            'data': data,
            'config': config or {},
            'enable_quantum_analysis': enable_quantum_analysis
        }
        
        self.evaluations.append(evaluation_config)
        self.logger.info(f"Added evaluation: {name} (quantum analysis: {enable_quantum_analysis})")
    
    def run_evaluation(
        self,
        name: str,
        parallel: bool = False,
        max_workers: int = 4
    ) -> List[EvaluationResult]:
        """
        Run a specific evaluation.
        
        Args:
            name: Name of the evaluation to run
            parallel: Whether to run in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of evaluation results
        """
        evaluation = next((e for e in self.evaluations if e['name'] == name), None)
        if not evaluation:
            raise ValueError(f"Evaluation '{name}' not found")
        
        if parallel:
            return self._run_parallel_evaluation([evaluation], max_workers)
        else:
            return self._run_sequential_evaluation([evaluation])
    
    def run_all_evaluations(
        self,
        parallel: bool = False,
        max_workers: int = 4
    ) -> List[EvaluationResult]:
        """
        Run all evaluations in the pipeline.
        
        Args:
            parallel: Whether to run evaluations in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of all evaluation results
        """
        if parallel:
            return self._run_parallel_evaluation(self.evaluations, max_workers)
        else:
            return self._run_sequential_evaluation(self.evaluations)
    
    def _run_sequential_evaluation(
        self, 
        evaluations: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """Run evaluations sequentially with three-phase integration."""
        results = []
        
        for evaluation in evaluations:
            self.logger.info(f"Running evaluation: {evaluation['name']}")
            
            try:
                # Create evaluator (Phase 2)
                evaluator = evaluation['evaluator_class'](
                    evaluation['model'],
                    **evaluation['config']
                )
                
                # Run evaluation (Phase 2)
                start_time = time.time()
                result = evaluator.evaluate(evaluation['data'])
                end_time = time.time()
                
                # Add timing information
                result.metadata['evaluation_time'] = end_time - start_time
                result.metadata['evaluation_name'] = evaluation['name']
                
                # ⚠️ CRITICAL: Enhance with Phase 3 quantum analysis if enabled
                if evaluation.get('enable_quantum_analysis', True):
                    result = self._enhance_with_quantum_analysis(result, evaluation)
                
                results.append(result)
                self.logger.info(f"Completed evaluation: {evaluation['name']} in {end_time - start_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error in evaluation {evaluation['name']}: {str(e)}")
                continue
        
        self.results.extend(results)
        return results
    
    def _enhance_with_quantum_analysis(
        self,
        result: EvaluationResult,
        evaluation: Dict[str, Any]
    ) -> EvaluationResult:
        """Enhance Phase 2 result with Phase 3 quantum analysis."""
        try:
            # Extract model outputs and embeddings for quantum analysis
            model_outputs = self._extract_model_outputs(evaluation['model'], evaluation['data'])
            embeddings = self._extract_embeddings(evaluation['model'], evaluation['data'])
            attention_mask = self._extract_attention_mask(evaluation['data'])
            
            # Use coordinator to enhance result
            enhanced_result = self.coordinator.enhance_evaluation_result(
                result, model_outputs, embeddings, attention_mask
            )
            
            self.logger.info(f"Enhanced {evaluation['name']} with quantum analysis")
            return enhanced_result
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance {evaluation['name']} with quantum analysis: {str(e)}")
            # Return original result if enhancement fails
            return result
    
    def _extract_model_outputs(self, model: torch.nn.Module, data: Any) -> Any:
        """Extract model outputs for quantum analysis."""
        try:
            # Handle different data types
            if hasattr(data, 'dataset') and hasattr(data.dataset, '__getitem__'):
                # DataLoader - get first batch
                sample_batch = next(iter(data))
            elif isinstance(data, (list, tuple)) and len(data) > 0:
                # List/tuple of data
                sample_batch = data[0]
            else:
                # Single data item
                sample_batch = data
            
            # Extract input data for forward pass
            if isinstance(sample_batch, dict):
                inputs = {k: v for k, v in sample_batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
            elif isinstance(sample_batch, (list, tuple)):
                inputs = {'input_ids': sample_batch[0]}
            else:
                inputs = {'input_ids': sample_batch}
            
            # Ensure inputs are on the same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Run forward pass to get model outputs
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract quantum-specific outputs if available
            if hasattr(outputs, 'quantum_uncertainty'):
                quantum_uncertainty = outputs.quantum_uncertainty
            elif hasattr(outputs, 'uncertainty'):
                quantum_uncertainty = outputs.uncertainty
            else:
                quantum_uncertainty = None
            
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden_state = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
                last_hidden_state = outputs.hidden_states[-1]
            else:
                last_hidden_state = None
            
            # Create structured output object
            class ModelOutputs:
                def __init__(self, uncertainty=None, hidden_state=None):
                    self.quantum_uncertainty = uncertainty
                    self.last_hidden_state = hidden_state
            
            return ModelOutputs(quantum_uncertainty, last_hidden_state)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract model outputs: {str(e)}")
            # Return mock outputs as fallback
            class MockOutputs:
                def __init__(self):
                    self.quantum_uncertainty = None
                    self.last_hidden_state = None
            return MockOutputs()
    
    def _extract_embeddings(self, model: torch.nn.Module, data: Any) -> Optional[torch.Tensor]:
        """Extract embeddings for quantum analysis."""
        try:
            # Get model outputs first
            outputs = self._extract_model_outputs(model, data)
            
            # Try to extract embeddings from outputs
            if outputs.last_hidden_state is not None:
                return outputs.last_hidden_state
            
            # If no hidden state, try to get embeddings from model
            if hasattr(model, 'embeddings'):
                # Handle different embedding types
                if hasattr(model.embeddings, 'word_embeddings'):
                    # BERT-style embeddings
                    sample_batch = self._get_sample_batch(data)
                    input_ids = self._extract_input_ids(sample_batch)
                    if input_ids is not None:
                        device = next(model.parameters()).device
                        input_ids = input_ids.to(device)
                        with torch.no_grad():
                            embeddings = model.embeddings.word_embeddings(input_ids)
                        return embeddings
                elif hasattr(model.embeddings, 'forward'):
                    # Generic embedding layer
                    sample_batch = self._get_sample_batch(data)
                    input_ids = self._extract_input_ids(sample_batch)
                    if input_ids is not None:
                        device = next(model.parameters()).device
                        input_ids = input_ids.to(device)
                        with torch.no_grad():
                            embeddings = model.embeddings(input_ids)
                        return embeddings
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract embeddings: {str(e)}")
            return None
    
    def _extract_attention_mask(self, data: Any) -> Optional[torch.Tensor]:
        """Extract attention mask from data."""
        try:
            sample_batch = self._get_sample_batch(data)
            
            if isinstance(sample_batch, dict) and 'attention_mask' in sample_batch:
                attention_mask = sample_batch['attention_mask']
                if hasattr(attention_mask, 'to'):
                    device = next(iter(self.evaluations[0]['model'].parameters())).device
                    attention_mask = attention_mask.to(device)
                return attention_mask
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract attention mask: {str(e)}")
            return None
    
    def _get_sample_batch(self, data: Any) -> Any:
        """Helper method to get a sample batch from various data types."""
        if hasattr(data, 'dataset') and hasattr(data.dataset, '__getitem__'):
            # DataLoader
            return next(iter(data))
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            # List/tuple
            return data[0]
        else:
            # Single item
            return data
    
    def _extract_input_ids(self, sample_batch: Any) -> Optional[torch.Tensor]:
        """Helper method to extract input_ids from a sample batch."""
        if isinstance(sample_batch, dict) and 'input_ids' in sample_batch:
            return sample_batch['input_ids']
        elif isinstance(sample_batch, (list, tuple)) and len(sample_batch) > 0:
            return sample_batch[0]
        elif hasattr(sample_batch, 'shape'):
            # Assume it's already input_ids
            return sample_batch
        return None
    
    def _run_parallel_evaluation(
        self, 
        evaluations: List[Dict[str, Any]],
        max_workers: int
    ) -> List[EvaluationResult]:
        """Run evaluations in parallel."""
        # For now, use threading since models may not be pickleable
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_eval = {
                executor.submit(self._run_single_evaluation, eval_config): eval_config
                for eval_config in evaluations
            }
            
            results = []
            for future in future_to_eval:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    eval_config = future_to_eval[future]
                    self.logger.error(f"Error in parallel evaluation {eval_config['name']}: {str(e)}")
                    continue
        
        self.results.extend(results)
        return results
    
    def _run_single_evaluation(self, evaluation: Dict[str, Any]) -> Optional[EvaluationResult]:
        """Run a single evaluation (for parallel execution)."""
        try:
            evaluator = evaluation['evaluator_class'](
                evaluation['model'],
                **evaluation['config']
            )
            
            start_time = time.time()
            result = evaluator.evaluate(evaluation['data'])
            end_time = time.time()
            
            result.metadata['evaluation_time'] = end_time - start_time
            result.metadata['evaluation_name'] = evaluation['name']
            
            # Enhance with quantum analysis if enabled
            if evaluation.get('enable_quantum_analysis', True):
                result = self._enhance_with_quantum_analysis(result, evaluation)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in evaluation {evaluation['name']}: {str(e)}")
            return None
    
    def save_pipeline_results(self, filename: str = "pipeline_results.json"):
        """Save all pipeline results to file."""
        filepath = self.output_dir / filename
        
        pipeline_data = {
            'pipeline_config': self.base_config,
            'evaluations': [e['name'] for e in self.evaluations],
            'results': [result.__dict__ for result in self.results],
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(pipeline_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved pipeline results to {filepath}")
    
    def generate_report(self, output_format: str = "html") -> str:
        """Generate comprehensive evaluation report."""
        reporter = EvaluationReporter(self.results)
        return reporter.generate_report(output_format)
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare performance across different models."""
        # Use aggregator for consistent result comparison
        return self.aggregator.aggregate_evaluation_results(
            [r for r in self.results if r.model_name in model_names]
        )
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all evaluation results."""
        return self.aggregator.aggregate_evaluation_results(self.results)
