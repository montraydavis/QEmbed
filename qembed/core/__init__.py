"""
Core quantum embedding implementations.

This module contains the fundamental quantum-inspired components for
creating and manipulating embeddings with quantum properties.
"""

from .quantum_embeddings import QuantumEmbeddings
from .collapse_layers import ContextCollapseLayer
from .entanglement import EntanglementCorrelation
from .measurement import QuantumMeasurement

__all__ = [
    "QuantumEmbeddings",
    "ContextCollapseLayer",
    "EntanglementCorrelation", 
    "QuantumMeasurement",
]
