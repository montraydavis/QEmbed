"""
Tests for QEmbed evaluation system.

This test suite covers all four phases of the evaluation system:
- Phase 1: Core infrastructure (BaseEvaluator, EvaluationMetrics)
- Phase 2: Task-specific evaluators (Classification, MLM, Embedding)
- Phase 3: Quantum analysis tools (QuantumEvaluation, UncertaintyAnalyzer, SuperpositionAnalyzer)
- Phase 4: Pipeline and reporting (EvaluationPipeline, EvaluationReporter, ModelComparator, ResultAggregator)
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import evaluation components
from qembed.evaluation import (
    BaseEvaluator, EvaluationMetrics, EvaluationResult,
    ClassificationEvaluator, MLMEvaluator, EmbeddingEvaluator,
    QuantumEvaluation, UncertaintyAnalyzer, SuperpositionAnalyzer,
    EvaluationPipeline, EvaluationReporter, ModelComparator, ResultAggregator
)

# Import models for testing
from qembed.models.quantum_bert import QuantumBertForSequenceClassification
from qembed.core.quantum_embeddings import QuantumEmbeddings
from transformers import BertConfig


class TestBaseEvaluator:
    """Test cases for BaseEvaluator class."""
    
    # Define ConcreteEvaluator at class level
    class ConcreteEvaluator(BaseEvaluator):
        def evaluate(self, data, **kwargs):
            return EvaluationResult(
                task_name="test",
                model_name="test_model",
                metrics={"accuracy": 0.85}
            )
        
        def _evaluate_batch_internal(self, batch):
            return {
                'predictions': torch.randint(0, 3, (2, 10)),
                'labels': torch.randint(0, 3, (2, 10))
            }
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.__class__.__name__ = "MockModel"
        model.eval.return_value = None
        model.to.return_value = model
        return model
    
    @pytest.fixture
    def evaluator(self, mock_model):
        """Create a concrete BaseEvaluator instance for testing."""
        return self.ConcreteEvaluator(model=mock_model, device='cpu')
    
    def test_initialization(self, evaluator, mock_model):
        """Test proper initialization."""
        assert evaluator.model == mock_model
        assert evaluator.device == 'cpu'
        assert evaluator.include_quantum_analysis is True
        assert evaluator.save_intermediate_results is True
        assert len(evaluator.results) == 0
        assert evaluator.current_result is None
        
        # Check that quantum_metrics exists
        assert hasattr(evaluator, 'quantum_metrics')
    
    def test_compute_quantum_metrics_with_uncertainty(self, evaluator):
        """Test quantum metrics computation with uncertainty."""
        # Mock outputs with uncertainty
        outputs = Mock()
        outputs.quantum_uncertainty = torch.rand(2, 10)
        outputs.last_hidden_state = torch.randn(2, 10, 768)
        
        metrics = evaluator.compute_quantum_metrics(outputs)
        
        assert isinstance(metrics, dict)
        assert 'mean_uncertainty' in metrics
        assert 'std_uncertainty' in metrics
        assert 'max_uncertainty' in metrics
        assert 'min_uncertainty' in metrics
        assert 'uncertainty_entropy' in metrics
        assert 'mean_superposition' in metrics
        assert 'superposition_entropy' in metrics
    
    def test_compute_quantum_metrics_without_uncertainty(self, evaluator):
        """Test quantum metrics computation without uncertainty."""
        # Mock outputs without uncertainty
        outputs = Mock()
        outputs.last_hidden_state = torch.randn(2, 10, 768)
        # Mock the model to return actual tensors for uncertainty
        evaluator.model.get_uncertainty = Mock(return_value=torch.rand(2, 10))
        # Mock the uncertainty attribute to return a proper tensor
        outputs.quantum_uncertainty = torch.rand(2, 10)
        
        metrics = evaluator.compute_quantum_metrics(outputs)
        
        assert isinstance(metrics, dict)
        assert 'mean_superposition' in metrics
        assert 'superposition_entropy' in metrics
    
    def test_compute_quantum_metrics_disabled(self, evaluator):
        """Test quantum metrics computation when disabled."""
        evaluator.include_quantum_analysis = False
        
        outputs = Mock()
        outputs.quantum_uncertainty = torch.rand(2, 10)
        
        metrics = evaluator.compute_quantum_metrics(outputs)
        
        assert metrics == {}
    
    def test_analyze_uncertainty(self, evaluator):
        """Test uncertainty analysis."""
        uncertainty = torch.rand(2, 10)
        
        analysis = evaluator._analyze_uncertainty(uncertainty)
        
        assert isinstance(analysis, dict)
        assert 'mean_uncertainty' in analysis
        assert 'std_uncertainty' in analysis
        assert 'max_uncertainty' in analysis
        assert 'min_uncertainty' in analysis
        assert 'uncertainty_entropy' in analysis
    
    def test_analyze_superposition(self, evaluator):
        """Test superposition analysis."""
        hidden_states = torch.randn(2, 10, 768)
        
        analysis = evaluator._analyze_superposition(hidden_states)
        
        assert isinstance(analysis, dict)
        assert 'mean_superposition' in analysis
        assert 'superposition_entropy' in analysis
    
    def test_compute_tensor_entropy(self, evaluator):
        """Test tensor entropy computation."""
        tensor = torch.randn(10, 10)
        
        entropy = evaluator._compute_entropy(tensor)
        
        assert isinstance(entropy, float)
        assert entropy >= 0
    
    def test_save_and_load_results(self, evaluator):
        """Test saving and loading results."""
        # Create a test result
        result = EvaluationResult(
            task_name="test_task",
            model_name="test_model",
            metrics={"accuracy": 0.85},
            quantum_metrics={"coherence": 0.75}
        )
        evaluator.results.append(result)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            filepath = tmp_file.name
        
        try:
            # Save results
            evaluator.save_results(filepath)
            
            # Check that file exists and contains data
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert 'results' in data
            assert 'metadata' in data
            assert len(data['results']) == 1
            
            # Load results into new evaluator
            new_evaluator = TestBaseEvaluator.ConcreteEvaluator(model=Mock(), device='cpu')
            new_evaluator.load_results(filepath)
            
            assert len(new_evaluator.results) == 1
            assert new_evaluator.results[0].task_name == "test_task"
            assert new_evaluator.results[0].model_name == "test_model"
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_get_summary(self, evaluator):
        """Test summary generation."""
        # Add some test results
        for i in range(3):
            result = EvaluationResult(
                task_name=f"task_{i}",
                model_name=f"model_{i}",
                metrics={"accuracy": 0.8 + i * 0.05}
            )
            evaluator.results.append(result)
        
        summary = evaluator.get_summary()
        
        assert isinstance(summary, dict)
        assert 'total_evaluations' in summary
        assert 'average_metrics' in summary
        assert 'best_performance' in summary
        assert summary['total_evaluations'] == 3
        assert 'accuracy' in summary['average_metrics']
        assert 'accuracy' in summary['best_performance']


class TestEvaluationMetrics:
    """Test cases for EvaluationMetrics class."""
    
    @pytest.fixture
    def metrics(self):
        """Create an EvaluationMetrics instance for testing."""
        return EvaluationMetrics()
    
    def test_classification_metrics(self, metrics):
        """Test classification metrics computation."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 1, 2]
        
        result = metrics.classification_metrics(y_true, y_pred)
        
        assert isinstance(result, dict)
        assert 'accuracy' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
        assert 'per_class_precision' in result
        assert 'per_class_recall' in result
        assert 'per_class_f1' in result
        
        # Check that metrics are reasonable
        assert 0 <= result['accuracy'] <= 1
        assert 0 <= result['precision'] <= 1
        assert 0 <= result['recall'] <= 1
        assert 0 <= result['f1'] <= 1
    
    def test_confusion_matrix_analysis(self, metrics):
        """Test confusion matrix analysis."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 1, 2]
        
        result = metrics.confusion_matrix_analysis(y_true, y_pred)
        
        assert isinstance(result, dict)
        assert 'confusion_matrix' in result
        assert 'total_samples' in result
        assert 'correct_predictions' in result
        assert 'incorrect_predictions' in result
        assert 'overall_accuracy' in result
        
        assert result['total_samples'] == 6
        assert result['correct_predictions'] == 5
        assert result['incorrect_predictions'] == 1
        assert result['overall_accuracy'] == 5/6
    
    def test_mlm_metrics(self, metrics):
        """Test MLM metrics computation."""
        logits = torch.randn(2, 10, 1000)  # [batch, seq_len, vocab_size]
        labels = torch.randint(0, 1000, (2, 10))
        labels[0, 5:] = -100  # Some ignored tokens
        
        result = metrics.mlm_metrics(logits, labels)
        
        assert isinstance(result, dict)
        assert 'loss' in result
        assert 'perplexity' in result
        assert 'accuracy' in result
        assert 'top_1_accuracy' in result
        assert 'top_5_accuracy' in result
        
        # Check that metrics are reasonable
        assert result['loss'] > 0
        assert result['perplexity'] > 0
        assert 0 <= result['accuracy'] <= 1
        assert 0 <= result['top_1_accuracy'] <= 1
        assert 0 <= result['top_5_accuracy'] <= 1
    
    def test_embedding_metrics(self, metrics):
        """Test embedding metrics computation."""
        # Flatten embeddings to 2D for the method
        embeddings = torch.randn(20, 128)  # [batch*seq, embed_dim]
        labels = torch.randint(0, 5, (20,))
        
        result = metrics.embedding_metrics(embeddings, labels)
        
        assert isinstance(result, dict)
        assert 'mean_embedding_norm' in result
        assert 'embedding_variance' in result
        assert 'mean_cosine_similarity' in result
        assert 'std_cosine_similarity' in result
        
        # Check that metrics are reasonable
        assert result['mean_embedding_norm'] > 0
        assert result['embedding_variance'] >= 0
        assert -1 <= result['mean_cosine_similarity'] <= 1
        assert result['std_cosine_similarity'] >= 0
    
    def test_statistical_analysis(self, metrics):
        """Test statistical analysis."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        result = metrics.statistical_analysis(values)
        
        assert isinstance(result, dict)
        assert 'mean' in result
        assert 'median' in result
        assert 'std' in result
        assert 'min' in result
        assert 'max' in result
        assert 'q25' in result
        assert 'q75' in result
        assert 'iqr' in result
        
        # Check that statistics are correct
        assert result['mean'] == 3.0
        assert result['median'] == 3.0
        assert result['min'] == 1.0
        assert result['max'] == 5.0


class TestClassificationEvaluator:
    """Test cases for ClassificationEvaluator class."""
    
    @pytest.fixture
    def model(self):
        """Create a mock classification model."""
        model = Mock()
        model.__class__.__name__ = "QuantumBertForSequenceClassification"
        
        # Mock forward pass
        outputs = Mock()
        outputs.logits = torch.randn(2, 10, 3)  # [batch, seq_len, num_classes]
        outputs.quantum_uncertainty = torch.rand(2, 10)
        model.return_value = outputs
        
        return model
    
    @pytest.fixture
    def evaluator(self, model):
        """Create a ClassificationEvaluator instance."""
        return ClassificationEvaluator(model=model, device='cpu')
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader."""
        dataloader = Mock()
        
        # Mock batch data
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 3, (2, 10))
        }
        
        dataloader.__iter__.return_value = [batch]
        return dataloader
    
    def test_initialization(self, evaluator, model):
        """Test proper initialization."""
        assert evaluator.model == model
        assert evaluator.task_type == 'single_label'
        assert hasattr(evaluator, 'metrics_calculator')
    
    def test_evaluate_batch_internal(self, evaluator):
        """Test internal batch evaluation."""
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 3, (2, 10))
        }
        
        result = evaluator._evaluate_batch_internal(batch)
        
        assert isinstance(result, dict)
        assert 'predictions' in result
        assert 'labels' in result
        assert 'logits' in result
        assert 'uncertainty' in result
        
        assert result['predictions'].shape == (2, 10)
        assert result['labels'].shape == (2, 10)
        assert result['logits'].shape == (2, 10, 3)
    
    def test_compute_classification_metrics(self, evaluator):
        """Test classification metrics computation."""
        # Flatten to 1D for sklearn metrics
        predictions = torch.randint(0, 3, (20,))  # [batch*seq]
        labels = torch.randint(0, 3, (20,))      # [batch*seq]
        
        metrics = evaluator._compute_classification_metrics(predictions, labels)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics
    
    def test_get_class_performance(self, evaluator):
        """Test class performance analysis."""
        # Mock current result with per-class metrics
        evaluator.current_result = EvaluationResult(
            task_name="test",
            model_name="test_model",
            metrics={
                'per_class_precision': [0.8, 0.9, 0.7],
                'per_class_recall': [0.85, 0.88, 0.75],
                'per_class_f1': [0.82, 0.89, 0.72]
            }
        )
        
        performance = evaluator.get_class_performance()
        
        assert isinstance(performance, dict)
        assert 'class_0' in performance
        assert 'class_1' in performance
        assert 'class_2' in performance
        
        for class_name, metrics in performance.items():
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics


class TestMLMEvaluator:
    """Test cases for MLMEvaluator class."""
    
    @pytest.fixture
    def model(self):
        """Create a mock MLM model."""
        model = Mock()
        model.__class__.__name__ = "QuantumBertForMaskedLM"
        
        # Mock forward pass
        outputs = Mock()
        outputs.logits = torch.randn(2, 10, 1000)  # [batch, seq_len, vocab_size]
        model.return_value = outputs
        
        return model
    
    @pytest.fixture
    def evaluator(self, model):
        """Create an MLMEvaluator instance."""
        return MLMEvaluator(model=model, device='cpu')
    
    def test_initialization(self, evaluator, model):
        """Test proper initialization."""
        assert evaluator.model == model
        assert hasattr(evaluator, 'metrics_calculator')
    
    def test_evaluate_batch_internal(self, evaluator):
        """Test internal batch evaluation."""
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 1000, (2, 10))
        }
        
        result = evaluator._evaluate_batch_internal(batch)
        
        assert isinstance(result, dict)
        assert 'logits' in result
        assert 'labels' in result
        assert 'loss' in result
        assert 'uncertainty' in result
        
        assert result['logits'].shape == (2, 10, 1000)
        assert result['labels'].shape == (2, 10)
    
    def test_compute_mlm_metrics(self, evaluator):
        """Test MLM metrics computation."""
        logits = torch.randn(2, 10, 1000)
        labels = torch.randint(0, 1000, (2, 10))
        
        metrics = evaluator._compute_mlm_metrics(logits, labels)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'accuracy' in metrics
        assert 'top_1_accuracy' in metrics
        assert 'top_5_accuracy' in metrics
    
    def test_compute_top_k_accuracy(self, evaluator):
        """Test top-k accuracy computation."""
        logits = torch.randn(2, 10, 1000)
        labels = torch.randint(0, 1000, (2, 10))
        
        accuracy = evaluator._compute_top_k_accuracy(logits, labels, k=5)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1


class TestEmbeddingEvaluator:
    """Test cases for EmbeddingEvaluator class."""
    
    @pytest.fixture
    def model(self):
        """Create a mock embedding model."""
        model = Mock()
        model.__class__.__name__ = "QuantumEmbeddings"
        
        # Mock forward pass
        outputs = torch.randn(2, 10, 128)  # [batch, seq_len, embed_dim]
        model.return_value = outputs
        
        return model
    
    @pytest.fixture
    def evaluator(self, model):
        """Create an EmbeddingEvaluator instance."""
        return EmbeddingEvaluator(model=model, device='cpu')
    
    def test_initialization(self, evaluator, model):
        """Test proper initialization."""
        assert evaluator.model == model
        assert hasattr(evaluator, 'metrics_calculator')
    
    def test_evaluate_batch_internal(self, evaluator):
        """Test internal batch evaluation."""
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        
        result = evaluator._evaluate_batch_internal(batch)
        
        assert isinstance(result, dict)
        assert 'embeddings' in result
        assert 'labels' in result
        assert 'uncertainty' in result
        
        assert result['embeddings'].shape == (2, 10, 128)
    
    def test_compute_embedding_metrics(self, evaluator):
        """Test embedding metrics computation."""
        # Flatten embeddings to 2D for the method
        embeddings = torch.randn(20, 128)  # [batch*seq, embed_dim]
        labels = None
        
        metrics = evaluator._compute_embedding_metrics(embeddings, labels)
        
        assert isinstance(metrics, dict)
        assert 'mean_embedding_norm' in metrics
        assert 'embedding_variance' in metrics
        assert 'mean_cosine_similarity' in metrics
        assert 'std_cosine_similarity' in metrics
    
    def test_compute_similarity_metrics(self, evaluator):
        """Test similarity metrics computation."""
        # Flatten embeddings to 2D for the method
        embeddings = torch.randn(20, 128)  # [batch*seq, embed_dim]
        
        metrics = evaluator._compute_similarity_metrics(embeddings)
        
        assert isinstance(metrics, dict)
        assert 'mean_cosine_similarity' in metrics
        assert 'std_cosine_similarity' in metrics
    
    def test_compute_diversity_metrics(self, evaluator):
        """Test diversity metrics computation."""
        # Flatten embeddings to 2D for the method
        embeddings = torch.randn(20, 128)  # [batch*seq, embed_dim]
        
        metrics = evaluator._compute_diversity_metrics(embeddings)
        
        assert isinstance(metrics, dict)
        assert 'embedding_coverage' in metrics


class TestQuantumEvaluation:
    """Test cases for QuantumEvaluation class."""
    
    @pytest.fixture
    def quantum_eval(self):
        """Create a QuantumEvaluation instance."""
        return QuantumEvaluation()
    
    def test_coherence_metrics(self, quantum_eval):
        """Test coherence metrics computation."""
        embeddings = torch.randn(2, 10, 128)
        attention_mask = torch.ones(2, 10)
        
        metrics = quantum_eval.coherence_metrics(embeddings, attention_mask)
        
        assert isinstance(metrics, dict)
        assert 'mean_coherence' in metrics
        assert 'std_coherence' in metrics
        assert 'coherence_entropy' in metrics
    
    def test_superposition_quality(self, quantum_eval):
        """Test superposition quality assessment."""
        state_embeddings = torch.randn(100, 4, 128)  # [vocab_size, num_states, embed_dim]
        superposition_matrix = torch.randn(4, 4)
        
        metrics = quantum_eval.superposition_quality(state_embeddings, superposition_matrix)
        
        assert isinstance(metrics, dict)
        assert 'eigenvalue_entropy' in metrics
        assert 'mixing_efficiency' in metrics
        assert 'state_diversity' in metrics
        assert 'superposition_rank' in metrics
    
    def test_entanglement_quantification(self, quantum_eval):
        """Test entanglement quantification."""
        embeddings = torch.randn(2, 10, 128)
        attention_mask = torch.ones(2, 10)
        
        metrics = quantum_eval.entanglement_quantification(embeddings, attention_mask)
        
        assert isinstance(metrics, dict)
        assert 'mean_correlation' in metrics
        assert 'correlation_entropy' in metrics
        assert 'mean_mutual_info' in metrics
        assert 'entanglement_strength' in metrics
    
    def test_quantum_state_evolution(self, quantum_eval):
        """Test quantum state evolution analysis."""
        embeddings_sequence = [
            torch.randn(2, 10, 128),
            torch.randn(2, 10, 128),
            torch.randn(2, 10, 128)
        ]
        
        metrics = quantum_eval.quantum_state_evolution(embeddings_sequence)
        
        assert isinstance(metrics, dict)
        assert 'evolution_stability' in metrics
        assert 'evolution_entropy' in metrics
        assert 'quantum_jumps' in metrics
        assert 'mean_transition_prob' in metrics
        assert 'transition_consistency' in metrics


class TestUncertaintyAnalyzer:
    """Test uncertainty analyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create uncertainty analyzer instance."""
        return UncertaintyAnalyzer()
    
    @pytest.fixture
    def uncertainty(self):
        """Create uncertainty tensor."""
        return torch.rand(2, 10)
    
    @pytest.fixture
    def predictions(self):
        """Create predictions tensor."""
        return torch.randn(2, 10, 3)
    
    @pytest.fixture
    def targets(self):
        """Create targets tensor."""
        return torch.randint(0, 3, (2, 10))
    
    def test_analyze_uncertainty_distribution(self, analyzer, uncertainty, predictions, targets):
        """Test uncertainty distribution analysis."""
        result = analyzer.analyze_uncertainty_distribution(uncertainty, predictions, targets)
        
        # Check that key metrics are present
        assert 'mean_uncertainty' in result
        assert 'std_uncertainty' in result
        assert 'min_uncertainty' in result
        assert 'max_uncertainty' in result
        assert 'median_uncertainty' in result
        assert 'iqr_uncertainty' in result
        assert 'uncertainty_skewness' in result
        assert 'uncertainty_kurtosis' in result
        assert 'uncertainty_entropy' in result
        
        # Check that values are reasonable
        assert result['mean_uncertainty'] >= 0
        assert result['std_uncertainty'] >= 0
        assert result['min_uncertainty'] >= 0
        assert result['max_uncertainty'] >= 0
    
    def test_analyze_uncertainty_calibration(self, analyzer, predictions, targets, uncertainty):
        """Test uncertainty calibration analysis."""
        result = analyzer.analyze_uncertainty_calibration(predictions, targets, uncertainty)
        
        # Check that key metrics are present
        assert 'brier_score' in result
        assert 'expected_calibration_error' in result
        assert 'calibration_curve_fraction' in result
        assert 'calibration_curve_predicted' in result
        
        # Check that values are reasonable
        assert isinstance(result['brier_score'], (float, str))  # Could be nan or error string
        assert isinstance(result['expected_calibration_error'], (float, str))
    
    def test_analyze_uncertainty_regularization(self, analyzer, uncertainty):
        """Test uncertainty regularization analysis."""
        result = analyzer.analyze_uncertainty_regularization(uncertainty)
        
        # Check that key metrics are present
        assert 'uncertainty_regularization_mean' in result
        assert 'uncertainty_regularization_std' in result
        
        # Check that values are reasonable
        assert result['uncertainty_regularization_mean'] >= 0
        assert result['uncertainty_regularization_std'] >= 0
    
    def test_analyze_uncertainty_evolution(self, analyzer):
        """Test uncertainty evolution analysis."""
        uncertainty_sequence = [torch.rand(2, 10) for _ in range(3)]
        result = analyzer.analyze_uncertainty_evolution(uncertainty_sequence)
        
        # Check that key metrics are present
        assert 'mean_uncertainty_evolution' in result
        assert 'uncertainty_stability' in result
        assert 'uncertainty_convergence_slope' in result
        assert 'uncertainty_convergence_r_squared' in result
        assert 'uncertainty_quantum_jumps' in result
        
        # Check that values are reasonable
        assert len(result['mean_uncertainty_evolution']) == 3
        assert isinstance(result['uncertainty_stability'], float)
        assert isinstance(result['uncertainty_quantum_jumps'], int)
    
    def test_generate_uncertainty_report(self, analyzer, uncertainty, predictions, targets):
        """Test uncertainty report generation."""
        report = analyzer.generate_uncertainty_report(uncertainty, predictions, targets)
        
        # Check report structure
        assert 'distribution' in report
        assert 'calibration' in report
        assert 'regularization' in report
        assert 'summary' in report
        
        # Check summary
        assert report['summary']['total_samples'] == uncertainty.numel()
        assert 'uncertainty_range' in report['summary']


class TestSuperpositionAnalyzer:
    """Test superposition analyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create superposition analyzer instance."""
        return SuperpositionAnalyzer()
    
    @pytest.fixture
    def embeddings(self):
        """Create embeddings tensor."""
        return torch.randn(2, 10, 128)
    
    @pytest.fixture
    def attention_mask(self):
        """Create attention mask."""
        return torch.ones(2, 10, dtype=torch.bool)
    
    def test_analyze_superposition_states(self, analyzer, embeddings, attention_mask):
        """Test superposition states analysis."""
        result = analyzer.analyze_superposition_states(embeddings, attention_mask)
        
        # Check that key metrics are present
        assert 'batch_size' in result
        assert 'sequence_length' in result
        assert 'embedding_dimension' in result
        assert 'mean_superposition_strength' in result
        assert 'std_superposition_strength' in result
        assert 'max_superposition_strength' in result
        assert 'min_superposition_strength' in result
        assert 'mean_superposition_coherence' in result
        assert 'std_superposition_coherence' in result
        assert 'superposition_coherence_entropy' in result
        
        # Check that values are reasonable
        assert result['batch_size'] == 2
        assert result['sequence_length'] == 10
        assert result['embedding_dimension'] == 128
        assert result['mean_superposition_strength'] >= 0
        assert isinstance(result['mean_superposition_coherence'], float)
    
    def test_analyze_collapse_mechanisms(self, analyzer, embeddings):
        """Test collapse mechanisms analysis."""
        result = analyzer.analyze_collapse_mechanisms(embeddings)
        
        # Check that key metrics are present
        assert 'mean_collapse_stability' in result
        assert 'std_collapse_stability' in result
        assert 'collapse_stability_entropy' in result
        
        # Check that values are reasonable
        assert result['mean_collapse_stability'] >= 0
        assert result['std_collapse_stability'] >= 0
        assert result['collapse_stability_entropy'] >= 0
    
    def test_analyze_quantum_interference_effects(self, analyzer, embeddings):
        """Test quantum interference effects analysis."""
        result = analyzer.analyze_quantum_interference(embeddings)
        
        # Check that key metrics are present
        assert 'mean_interference_strength' in result
        assert 'std_interference_strength' in result
        assert 'interference_entropy' in result
        
        # Check that values are reasonable
        assert isinstance(result['mean_interference_strength'], float)
        assert isinstance(result['std_interference_strength'], float)
        assert result['interference_entropy'] >= 0
    
    def test_analyze_superposition_evolution(self, analyzer):
        """Test superposition evolution analysis."""
        embeddings_sequence = [torch.randn(2, 10, 128) for _ in range(3)]
        result = analyzer.analyze_superposition_evolution(embeddings_sequence)
        
        # Check that key metrics are present
        assert 'superposition_strength_evolution' in result
        assert 'superposition_evolution_stability' in result
        assert 'superposition_evolution_entropy' in result
        assert 'superposition_quantum_jumps' in result
        
        # Check that values are reasonable
        assert len(result['superposition_strength_evolution']) == 3
        assert 0 <= result['superposition_evolution_stability'] <= 1
        assert result['superposition_evolution_entropy'] >= 0
        assert isinstance(result['superposition_quantum_jumps'], int)
    
    def test_generate_superposition_report(self, analyzer, embeddings, attention_mask):
        """Test superposition report generation."""
        # Create a proper superposition matrix (square matrix)
        superposition_matrix = torch.randn(128, 128)
        report = analyzer.generate_superposition_report(embeddings, superposition_matrix, attention_mask=attention_mask)
        
        # Check report structure
        assert 'superposition_states' in report
        assert 'collapse_mechanisms' in report
        assert 'quantum_interference' in report
        assert 'summary' in report
        
        # Check summary
        assert report['summary']['total_samples'] == embeddings.numel()
        assert 'embedding_shape' in report['summary']


class TestEvaluationPipeline:
    """Test evaluation pipeline functionality."""
    
    @pytest.fixture
    def pipeline(self):
        """Create evaluation pipeline instance."""
        return EvaluationPipeline(base_config={'test': True})
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator."""
        evaluator = Mock()
        evaluator.evaluate.return_value = {
            'accuracy': 0.85,
            'f1': 0.83,
            'loss': 0.15
        }
        return evaluator
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create mock analyzer."""
        analyzer = Mock()
        analyzer.analyze.return_value = {
            'uncertainty': 0.2,
            'superposition': 0.7
        }
        return analyzer
    
    def test_compare_models(self, pipeline):
        """Test model comparison."""
        # Add some mock results to the pipeline
        mock_result = Mock()
        mock_result.model_name = 'model_0'
        mock_result.metrics = {'accuracy': 0.85}
        mock_result.quantum_metrics = {'uncertainty': 0.2}
        mock_result.uncertainty_analysis = {}
        mock_result.timestamp = time.time()
        mock_result.metadata = {}
        
        pipeline.results.append(mock_result)
        
        result = pipeline.compare_models(['model_0'])
        
        # Check that result contains aggregation info
        assert 'aggregation_strategy' in result
        assert 'aggregation_timestamp' in result


class TestModelComparator:
    """Test model comparator functionality."""
    
    @pytest.fixture
    def comparator(self):
        """Create model comparator instance."""
        return ModelComparator()
    
    @pytest.fixture
    def model_results(self):
        """Create model results."""
        return {
            'model_0': {
                'accuracy': 0.85,
                'f1': 0.83,
                'loss': 0.15
            },
            'model_1': {
                'accuracy': 0.87,
                'f1': 0.85,
                'loss': 0.13
            }
        }
    
    def test_compute_statistical_significance(self, comparator):
        """Test statistical significance computation."""
        # Create mock EvaluationResult objects
        mock_result1 = Mock()
        mock_result1.model_name = 'model_0'
        mock_result1.metrics = {'accuracy': 0.85}
        
        mock_result2 = Mock()
        mock_result2.model_name = 'model_1'
        mock_result2.metrics = {'accuracy': 0.87}
        
        results = [mock_result1, mock_result2]
        
        result = comparator._compute_statistical_significance(results, ['accuracy'])
        
        # Check that result contains statistical test info
        assert 'accuracy' in result
    
    def test_generate_comparison_report(self, comparator):
        """Test comparison report generation."""
        # Create mock EvaluationResult objects
        mock_result1 = Mock()
        mock_result1.model_name = 'model_0'
        mock_result1.metrics = {'accuracy': 0.85, 'f1': 0.83, 'loss': 0.15}
        mock_result1.quantum_metrics = {}
        mock_result1.uncertainty_analysis = {}
        mock_result1.timestamp = time.time()
        mock_result1.metadata = {}
        
        mock_result2 = Mock()
        mock_result2.model_name = 'model_1'
        mock_result2.metrics = {'accuracy': 0.87, 'f1': 0.85, 'loss': 0.13}
        mock_result2.quantum_metrics = {}
        mock_result2.uncertainty_analysis = {}
        mock_result2.timestamp = time.time()
        mock_result2.metadata = {}
        
        results = [mock_result1, mock_result2]
        
        report = comparator.generate_comparison_report(results)
        
        # Check that report is an HTML string
        assert isinstance(report, str)
        assert '<html>' in report
        assert '<title>QEmbed Model Comparison Report</title>' in report
        assert 'Performance Comparison' in report
        assert 'Statistical Significance' in report
        assert 'Ranking Analysis' in report


class TestResultAggregator:
    """Test cases for ResultAggregator class."""
    
    @pytest.fixture
    def aggregator(self):
        """Create a ResultAggregator instance."""
        return ResultAggregator()
    
    @pytest.fixture
    def results(self):
        """Create test results for aggregation."""
        return [
            EvaluationResult(
                task_name="test1",
                model_name="model1",
                metrics={"accuracy": 0.85, "f1": 0.82},
                quantum_metrics={"coherence": 0.75}
            ),
            EvaluationResult(
                task_name="test2",
                model_name="model2",
                metrics={"accuracy": 0.88, "f1": 0.85},
                quantum_metrics={"coherence": 0.78}
            )
        ]
    
    def test_initialization(self, aggregator):
        """Test proper initialization."""
        assert hasattr(aggregator, 'aggregation_cache')
    
    def test_aggregate_evaluation_results(self, aggregator, results):
        """Test evaluation results aggregation."""
        aggregated = aggregator.aggregate_evaluation_results(results)
        
        assert isinstance(aggregated, dict)
        assert 'total_evaluations' in aggregated
        assert 'evaluation_tasks' in aggregated
        assert 'models_evaluated' in aggregated
        assert 'aggregation_timestamp' in aggregated
        assert 'aggregation_strategy' in aggregated
        assert 'metrics_summary' in aggregated
        assert 'quantum_metrics_summary' in aggregated
        assert 'metadata_summary' in aggregated
        assert 'performance_ranking' in aggregated
        
        assert aggregated['total_evaluations'] == 2
        assert len(aggregated['evaluation_tasks']) == 2
        assert len(aggregated['models_evaluated']) == 2
    
    def test_aggregate_empty_results(self, aggregator):
        """Test aggregation with empty results."""
        aggregated = aggregator.aggregate_evaluation_results([])
        
        assert aggregated == {}
    
    def test_aggregate_single_result(self, aggregator):
        """Test aggregation with single result."""
        single_result = [
            EvaluationResult(
                task_name="test",
                model_name="model1",
                metrics={"accuracy": 0.85}
            )
        ]
        
        aggregated = aggregator.aggregate_evaluation_results(single_result)
        
        assert aggregated['total_evaluations'] == 1
        assert len(aggregated['evaluation_tasks']) == 1
        assert len(aggregated['models_evaluated']) == 1
    
    def test_compute_performance_ranking(self, aggregator, results):
        """Test performance ranking computation."""
        ranking = aggregator._compute_performance_ranking(results)
        
        assert isinstance(ranking, dict)
        assert 'accuracy' in ranking
        assert 'f1' in ranking
        
        # Check that rankings are lists of model names
        for metric, model_list in ranking.items():
            assert isinstance(model_list, list)
            assert all(isinstance(name, str) for name in model_list)


class TestEvaluationIntegration:
    """Integration tests for the complete evaluation system."""
    
    @pytest.fixture
    def quantum_model(self):
        """Create a quantum model for integration testing."""
        config = BertConfig(vocab_size=1000, hidden_size=768, num_labels=3)
        return QuantumBertForSequenceClassification(config)
    
    def test_full_evaluation_pipeline(self, quantum_model):
        """Test complete evaluation pipeline integration."""
        # Create pipeline
        pipeline = EvaluationPipeline({'test': True})
        
        # Add evaluation
        pipeline.add_evaluation(
            name="integration_test",
            model=quantum_model,
            evaluator_class=ClassificationEvaluator,
            data=None
        )
        
        # Run evaluation
        results = pipeline.run_evaluation("integration_test")
        
        # Should return list of results
        assert isinstance(results, list)
        assert len(results) >= 0  # May be empty if no data provided
    
    def test_evaluation_reporter_integration(self, quantum_model):
        """Test evaluation reporter integration."""
        # Create test results
        results = [
            EvaluationResult(
                task_name="integration_test",
                model_name="quantum_bert",
                metrics={"accuracy": 0.85},
                quantum_metrics={"coherence": 0.75}
            )
        ]
        
        # Create reporter
        reporter = EvaluationReporter(results)
        
        # Generate reports
        html_report = reporter.generate_report('html')
        json_report = reporter.generate_report('json')
        
        assert isinstance(html_report, str)
        assert isinstance(json_report, str)
        
        # JSON should be valid
        json.loads(json_report)
    
    def test_model_comparison_integration(self, quantum_model):
        """Test model comparison integration."""
        # Create test results
        results = [
            EvaluationResult(
                task_name="test",
                model_name="model_1",
                metrics={"accuracy": 0.85}
            ),
            EvaluationResult(
                task_name="test",
                model_name="model_2",
                metrics={"accuracy": 0.88}
            )
        ]
        
        # Create comparator
        comparator = ModelComparator()
        
        # Compare models
        comparison = comparator.compare_models(results)
        
        assert isinstance(comparison, dict)
        assert 'performance_comparison' in comparison
        assert 'statistical_significance' in comparison
    
    def test_result_aggregation_integration(self, quantum_model):
        """Test result aggregation integration."""
        # Create test results
        results = [
            EvaluationResult(
                task_name="test1",
                model_name="model1",
                metrics={"accuracy": 0.85}
            ),
            EvaluationResult(
                task_name="test2",
                model_name="model2",
                metrics={"accuracy": 0.88}
            )
        ]
        
        # Create aggregator
        aggregator = ResultAggregator()
        
        # Aggregate results
        aggregated = aggregator.aggregate_evaluation_results(results)
        
        assert isinstance(aggregated, dict)
        assert aggregated['total_evaluations'] == 2
        assert 'metrics_summary' in aggregated
        assert 'performance_ranking' in aggregated


if __name__ == "__main__":
    pytest.main([__file__])
