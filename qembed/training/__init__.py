"""
Training utilities for quantum-enhanced embeddings.

This module contains training loops, loss functions, and optimizers
specifically designed for quantum-inspired models.
"""

from .quantum_trainer import QuantumTrainer
from .losses import QuantumLoss
from .optimizers import QuantumOptimizer

__all__ = [
    "QuantumTrainer",
    "QuantumLoss",
    "QuantumOptimizer",
]
