"""
QEmbed Evaluation Package

This package provides comprehensive evaluation tools for quantum-enhanced
embeddings and models, including standard NLP metrics and quantum-specific
analysis capabilities.

⚠️ IMPORTANT: This package extends existing QEmbed infrastructure
    rather than duplicating functionality.
"""

from .base_evaluator import BaseEvaluator, EvaluationResult
from .evaluation_metrics import EvaluationMetrics  # Renamed to avoid conflict
from .classification_evaluator import ClassificationEvaluator
from .mlm_evaluator import MLMEvaluator
from .embedding_evaluator import EmbeddingEvaluator
from .quantum_evaluation import QuantumEvaluation  # Renamed to avoid conflict
from .uncertainty_analyzer import UncertaintyAnalyzer
from .superposition_analyzer import SuperpositionAnalyzer
from .pipeline import EvaluationPipeline
from .reporting import EvaluationReporter
from .comparison import ModelComparator
from .aggregation import ResultAggregator

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluationMetrics", 
    "ClassificationEvaluator",
    "MLMEvaluator",
    "EmbeddingEvaluator",
    "QuantumEvaluation",  # Renamed
    "UncertaintyAnalyzer",
    "SuperpositionAnalyzer",
    "EvaluationPipeline",
    "EvaluationReporter",
    "ModelComparator",
    "ResultAggregator",
]

__version__ = "0.1.0"
