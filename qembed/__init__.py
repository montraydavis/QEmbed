"""
QEmbed: Quantum-Enhanced Embeddings for Natural Language Processing

A Python library for creating quantum-inspired embeddings that capture
contextual uncertainty and polysemy in natural language.
"""

__version__ = "0.1.0"
__author__ = "QEmbed Team"
__email__ = "contact@qembed.ai"

from .core.quantum_embeddings import QuantumEmbeddings
from .core.collapse_layers import ContextCollapseLayer
from .core.entanglement import EntanglementCorrelation
from .core.measurement import QuantumMeasurement

from .models.quantum_bert import QuantumBERT
from .models.quantum_transformer import QuantumTransformer
from .models.hybrid_models import HybridModel

from .training.quantum_trainer import QuantumTrainer
from .training.losses import QuantumLoss
from .training.optimizers import QuantumOptimizer

__all__ = [
    "QuantumEmbeddings",
    "ContextCollapseLayer", 
    "EntanglementCorrelation",
    "QuantumMeasurement",
    "QuantumBERT",
    "QuantumTransformer",
    "HybridModel",
    "QuantumTrainer",
    "QuantumLoss",
    "QuantumOptimizer",
]
