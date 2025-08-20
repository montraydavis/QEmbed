"""
Quantum measurement operators for quantum embeddings.

This module implements various quantum measurement operators
that can be used to collapse superposition states and extract
classical information from quantum embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List


class QuantumMeasurement(nn.Module):
    """
    Base quantum measurement operator.
    
    This class provides the foundation for implementing
    various quantum measurement strategies on embeddings.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        measurement_basis: str = "computational",
        noise_level: float = 0.0
    ):
        """
        Initialize quantum measurement operator.
        
        Args:
            embedding_dim: Dimension of input embeddings
            measurement_basis: Basis for measurement ('computational', 'bell', 'custom')
            noise_level: Level of measurement noise to add
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.measurement_basis = measurement_basis
        self.noise_level = noise_level
        
        # Measurement basis matrices
        if measurement_basis == "computational":
            self.basis_matrix = nn.Parameter(torch.eye(embedding_dim))
        elif measurement_basis == "bell":
            self.basis_matrix = nn.Parameter(self._create_bell_basis())
        else:
            self.basis_matrix = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        
        # Normalize basis matrix
        self.basis_matrix.data = F.normalize(self.basis_matrix.data, p=2, dim=1)
        
    def _create_bell_basis(self) -> torch.Tensor:
        """Create Bell state measurement basis."""
        # Simple Bell-like basis (could be more sophisticated)
        basis = torch.eye(self.embedding_dim)
        
        # Add some off-diagonal elements for entanglement
        for i in range(0, self.embedding_dim - 1, 2):
            if i + 1 < self.embedding_dim:
                basis[i, i+1] = 0.7071
                basis[i+1, i] = 0.7071
                basis[i, i] = 0.7071
                basis[i+1, i+1] = 0.7071
        
        return basis
    
    def forward(
        self,
        embeddings: torch.Tensor,
        collapse_probability: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform quantum measurement on embeddings.
        
        Args:
            embeddings: Input quantum embeddings [batch_size, seq_len, embedding_dim]
            collapse_probability: Probability of collapsing superposition
            
        Returns:
            Tuple of (measured_embeddings, measurement_results)
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Project onto measurement basis
        projected = torch.einsum('bse,ef->bsf', embeddings, self.basis_matrix)
        
        # Add measurement noise if specified
        if self.noise_level > 0 and self.training:
            noise = torch.randn_like(projected) * self.noise_level
            projected = projected + noise
        
        # Determine which embeddings to collapse
        if collapse_probability < 1.0:
            collapse_mask = torch.rand(batch_size, seq_len) < collapse_probability
            collapse_mask = collapse_mask.unsqueeze(-1).expand_as(projected)
        else:
            collapse_mask = torch.ones_like(projected, dtype=torch.bool)
        
        # Apply collapse
        measured = torch.where(collapse_mask, projected, embeddings)
        
        # Compute measurement results (e.g., probabilities)
        measurement_results = self._compute_measurement_results(projected)
        
        return measured, measurement_results
    
    def _compute_measurement_results(self, projected: torch.Tensor) -> torch.Tensor:
        """Compute measurement results and statistics."""
        # Compute measurement probabilities
        probabilities = F.softmax(projected, dim=-1)
        
        # Compute measurement uncertainty
        uncertainty = torch.var(probabilities, dim=-1)
        
        # Combine into measurement results
        results = torch.cat([
            probabilities,
            uncertainty.unsqueeze(-1)
        ], dim=-1)
        
        return results


class AdaptiveMeasurement(nn.Module):
    """
    Adaptive quantum measurement operator.
    
    This operator learns to choose the optimal measurement
    basis for each embedding based on context and uncertainty.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_bases: int = 4,
        temperature: float = 1.0
    ):
        """
        Initialize adaptive measurement operator.
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_bases: Number of measurement bases to choose from
            temperature: Temperature for basis selection
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_bases = num_bases
        self.temperature = temperature
        
        # Basis selection network
        self.basis_selector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_bases)
        )
        
        # Multiple measurement bases
        self.measurement_bases = nn.ModuleList([
            QuantumMeasurement(embedding_dim, "computational"),
            QuantumMeasurement(embedding_dim, "bell"),
            QuantumMeasurement(embedding_dim, "custom"),
            QuantumMeasurement(embedding_dim, "custom")
        ])
        
        # Ensure we have enough bases
        while len(self.measurement_bases) < num_bases:
            self.measurement_bases.append(
                QuantumMeasurement(embedding_dim, "custom")
            )
        
    def forward(
        self,
        embeddings: torch.Tensor,
        collapse_probability: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform adaptive quantum measurement.
        
        Returns:
            Tuple of (measured_embeddings, measurement_results, basis_weights)
        """
        # Select measurement basis for each embedding
        basis_logits = self.basis_selector(embeddings)
        basis_weights = F.softmax(basis_logits / self.temperature, dim=-1)
        
        # Apply each measurement basis
        basis_outputs = []
        measurement_results = []
        
        for i, basis in enumerate(self.measurement_bases):
            measured, results = basis(embeddings, collapse_probability)
            basis_outputs.append(measured)
            measurement_results.append(results)
        
        # Combine outputs based on basis weights
        basis_outputs = torch.stack(basis_outputs, dim=2)
        measured = torch.sum(
            basis_outputs * basis_weights.unsqueeze(-1),
            dim=2
        )
        
        # Combine measurement results
        measurement_results = torch.stack(measurement_results, dim=2)
        combined_results = torch.sum(
            measurement_results * basis_weights.unsqueeze(-1),
            dim=2
        )
        
        return measured, combined_results, basis_weights


class WeakMeasurement(nn.Module):
    """
    Weak quantum measurement operator.
    
    This operator performs weak measurements that don't
    completely collapse the quantum state, preserving
    some quantum properties.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        measurement_strength: float = 0.5,
        decoherence_rate: float = 0.1
    ):
        """
        Initialize weak measurement operator.
        
        Args:
            embedding_dim: Dimension of input embeddings
            measurement_strength: Strength of measurement (0 = no collapse, 1 = full collapse)
            decoherence_rate: Rate of decoherence during measurement
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.measurement_strength = measurement_strength
        self.decoherence_rate = decoherence_rate
        
        # Weak measurement parameters
        self.weak_measurement_matrix = nn.Parameter(
            torch.eye(embedding_dim) * measurement_strength
        )
        
        # Decoherence matrix
        self.decoherence_matrix = nn.Parameter(
            torch.eye(embedding_dim) * decoherence_rate
        )
        
    def forward(
        self,
        embeddings: torch.Tensor,
        measurement_strength: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform weak quantum measurement.
        
        Args:
            embeddings: Input quantum embeddings
            measurement_strength: Override default measurement strength
            
        Returns:
            Tuple of (weakly_measured_embeddings, measurement_strength_used)
        """
        if measurement_strength is None:
            measurement_strength = self.measurement_strength
        
        # Apply weak measurement
        weak_measured = torch.einsum(
            'bse,ef->bsf',
            embeddings,
            self.weak_measurement_matrix * measurement_strength
        )
        
        # Add decoherence
        decoherence = torch.einsum(
            'bse,ef->bsf',
            embeddings,
            self.decoherence_matrix
        )
        
        # Combine original state with weak measurement and decoherence
        result = (1 - measurement_strength) * embeddings + weak_measured - decoherence
        
        # Track measurement strength used
        measurement_strength_tensor = torch.full_like(
            embeddings[:, :, 0],
            measurement_strength
        )
        
        return result, measurement_strength_tensor


class POVMMeasurement(nn.Module):
    """
    Positive Operator-Valued Measure (POVM) measurement.
    
    Implements POVM measurements which are more general than
    projective measurements and can handle mixed states.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_povm_elements: int = 4
    ):
        """
        Initialize POVM measurement operator.
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_povm_elements: Number of POVM elements
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_povm_elements = num_povm_elements
        
        # POVM elements (positive operators that sum to identity)
        self.povm_elements = nn.ParameterList([
            nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.1)
            for _ in range(num_povm_elements)
        ])
        
        # Ensure POVM elements are positive and sum to identity
        self._normalize_povm_elements()
        
    def _normalize_povm_elements(self):
        """Normalize POVM elements to ensure they form a valid POVM."""
        # Make elements positive semi-definite
        for element in self.povm_elements:
            element.data = torch.mm(element.data, element.data.t())
        
        # Normalize to sum to identity
        total = sum(element.data for element in self.povm_elements)
        for element in self.povm_elements:
            element.data = element.data / total
    
    def forward(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform POVM measurement.
        
        Returns:
            Tuple of (measured_embeddings, povm_probabilities)
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Compute POVM probabilities
        povm_probs = []
        
        for element in self.povm_elements:
            # Compute probability for this POVM element
            prob = torch.einsum('bse,ef,bsf->bs', embeddings, element, embeddings)
            povm_probs.append(prob)
        
        # Stack probabilities
        povm_probs = torch.stack(povm_probs, dim=-1)
        
        # Normalize probabilities
        povm_probs = F.softmax(povm_probs, dim=-1)
        
        # Apply POVM measurement
        measured = torch.zeros_like(embeddings)
        
        for i, element in enumerate(self.povm_elements):
            # Weight by POVM probability
            weight = povm_probs[:, :, i:i+1]
            element_contribution = torch.einsum('bse,ef->bsf', embeddings, element)
            measured += weight * element_contribution
        
        return measured, povm_probs
