"""
Quantum loss functions for training quantum-enhanced models.

This module implements loss functions that incorporate quantum
concepts like superposition, entanglement, and uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple


class QuantumLoss(nn.Module):
    """
    Base quantum loss function.
    
    Combines classical task loss with quantum-specific
    regularization terms.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        quantum_weight: float = 0.1,
        uncertainty_weight: float = 0.05,
        entanglement_weight: float = 0.02
    ):
        """
        Initialize quantum loss.
        
        Args:
            base_loss: Base loss function (e.g., CrossEntropyLoss)
            quantum_weight: Weight for quantum regularization
            uncertainty_weight: Weight for uncertainty regularization
            entanglement_weight: Weight for entanglement regularization
        """
        super().__init__()
        
        self.base_loss = base_loss
        self.quantum_weight = quantum_weight
        self.uncertainty_weight = uncertainty_weight
        self.entanglement_weight = entanglement_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantum_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute quantum loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            quantum_outputs: Additional quantum model outputs
            
        Returns:
            Total loss value
        """
        # Base task loss
        base_loss = self.base_loss(predictions, targets)
        
        # Quantum regularization terms
        quantum_reg = self._compute_quantum_regularization(quantum_outputs)
        
        # Total loss
        total_loss = base_loss + self.quantum_weight * quantum_reg
        
        return total_loss
    
    def _compute_quantum_regularization(
        self,
        quantum_outputs: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute quantum regularization terms."""
        if quantum_outputs is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        reg_terms = []
        
        # Uncertainty regularization
        if self.uncertainty_weight > 0 and 'uncertainty' in quantum_outputs:
            uncertainty = quantum_outputs['uncertainty']
            # Encourage reasonable uncertainty levels (not too high, not too low)
            target_uncertainty = torch.ones_like(uncertainty) * 0.5
            uncertainty_reg = F.mse_loss(uncertainty, target_uncertainty)
            reg_terms.append(self.uncertainty_weight * uncertainty_reg)
        
        # Entanglement regularization
        if self.entanglement_weight > 0 and 'entanglement' in quantum_outputs:
            entanglement = quantum_outputs['entanglement']
            # Encourage meaningful entanglement patterns
            entanglement_reg = self._entanglement_regularization(entanglement)
            reg_terms.append(self.entanglement_weight * entanglement_reg)
        
        # Superposition regularization
        if 'superposition' in quantum_outputs:
            superposition = quantum_outputs['superposition']
            # Encourage diverse superposition states
            superposition_reg = self._superposition_regularization(superposition)
            reg_terms.append(superposition_reg)
        
        if reg_terms:
            return torch.stack(reg_terms).sum()
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def _entanglement_regularization(self, entanglement: torch.Tensor) -> torch.Tensor:
        """Regularize entanglement patterns."""
        # Encourage non-trivial entanglement (not just identity)
        identity = torch.eye(entanglement.size(-1), device=entanglement.device)
        identity = identity.unsqueeze(0).expand_as(entanglement)
        
        # Penalize too much similarity to identity
        identity_penalty = F.mse_loss(entanglement, identity)
        
        # Encourage some structure (not random)
        structure_penalty = -torch.std(entanglement)
        
        return identity_penalty + structure_penalty
    
    def _superposition_regularization(self, superposition: torch.Tensor) -> torch.Tensor:
        """Regularize superposition states."""
        # Encourage diverse superposition weights
        weights = F.softmax(superposition, dim=-1)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        
        # Penalize low entropy (too deterministic)
        target_entropy = torch.log(torch.tensor(weights.size(-1), dtype=torch.float))
        entropy_penalty = F.mse_loss(entropy.mean(), target_entropy)
        
        return entropy_penalty


class SuperpositionLoss(nn.Module):
    """
    Loss function specifically for training superposition states.
    
    Encourages models to maintain meaningful superposition
    during training while allowing for controlled collapse.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        superposition_weight: float = 0.1,
        collapse_weight: float = 0.05
    ):
        """
        Initialize superposition loss.
        
        Args:
            base_loss: Base loss function
            superposition_weight: Weight for superposition regularization
            collapse_weight: Weight for collapse regularization
        """
        super().__init__()
        
        self.base_loss = base_loss
        self.superposition_weight = superposition_weight
        self.collapse_weight = collapse_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        superposition_states: torch.Tensor,
        collapsed_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute superposition loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            superposition_states: States before collapse
            collapsed_states: States after collapse
            
        Returns:
            Total loss value
        """
        # Base task loss
        base_loss = self.base_loss(predictions, targets)
        
        # Superposition regularization
        superposition_reg = self._superposition_regularization(superposition_states)
        
        # Collapse regularization
        collapse_reg = self._collapse_regularization(superposition_states, collapsed_states)
        
        # Total loss
        total_loss = (
            base_loss +
            self.superposition_weight * superposition_reg +
            self.collapse_weight * collapse_reg
        )
        
        return total_loss
    
    def _superposition_regularization(self, superposition_states: torch.Tensor) -> torch.Tensor:
        """Regularize superposition states."""
        # Encourage diverse superposition
        batch_size, seq_len, num_states, embed_dim = superposition_states.shape
        
        # Compute variance across states
        state_variance = torch.var(superposition_states, dim=2)
        
        # Encourage reasonable variance (not too low, not too high)
        target_variance = torch.ones_like(state_variance) * 0.1
        variance_reg = F.mse_loss(state_variance, target_variance)
        
        # Encourage orthogonality between states
        states_flat = superposition_states.view(-1, num_states, embed_dim)
        orthogonality_reg = self._orthogonality_regularization(states_flat)
        
        return variance_reg + orthogonality_reg
    
    def _orthogonality_regularization(self, states: torch.Tensor) -> torch.Tensor:
        """Encourage orthogonality between quantum states."""
        # Compute pairwise cosine similarities
        states_norm = F.normalize(states, p=2, dim=-1)
        similarities = torch.bmm(states_norm, states_norm.transpose(1, 2))
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(similarities.size(1), device=similarities.device)
        similarities = similarities * (1 - mask)
        
        # Penalize high similarities (encourage orthogonality)
        orthogonality_reg = torch.mean(torch.abs(similarities))
        
        return orthogonality_reg
    
    def _collapse_regularization(
        self,
        superposition_states: torch.Tensor,
        collapsed_states: torch.Tensor
    ) -> torch.Tensor:
        """Regularize the collapse process."""
        # Ensure collapsed states are close to one of the superposition states
        batch_size, seq_len, num_states, embed_dim = superposition_states.shape
        
        # Find closest superposition state for each collapsed state
        collapsed_expanded = collapsed_states.unsqueeze(2).expand(-1, -1, num_states, -1)
        
        # Compute distances to each superposition state
        distances = torch.norm(superposition_states - collapsed_expanded, dim=-1)
        
        # Find minimum distance for each position
        min_distances, _ = torch.min(distances, dim=-1)
        
        # Penalize large distances (collapse should be close to superposition)
        collapse_reg = torch.mean(min_distances)
        
        return collapse_reg


class EntanglementLoss(nn.Module):
    """
    Loss function for training entanglement correlations.
    
    Encourages meaningful entanglement patterns between
    different positions in sequences.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        entanglement_weight: float = 0.1,
        correlation_weight: float = 0.05
    ):
        """
        Initialize entanglement loss.
        
        Args:
            base_loss: Base loss function
            entanglement_weight: Weight for entanglement regularization
            correlation_weight: Weight for correlation regularization
        """
        super().__init__()
        
        self.base_loss = base_loss
        self.entanglement_weight = entanglement_weight
        self.correlation_weight = correlation_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        entanglement_matrix: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entanglement loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            entanglement_matrix: Entanglement correlation matrix
            attention_weights: Attention weights for correlation
            
        Returns:
            Total loss value
        """
        # Base task loss
        base_loss = self.base_loss(predictions, targets)
        
        # Entanglement regularization
        entanglement_reg = self._entanglement_regularization(entanglement_matrix)
        
        # Correlation regularization
        correlation_reg = self._correlation_regularization(entanglement_matrix, attention_weights)
        
        # Total loss
        total_loss = (
            base_loss +
            self.entanglement_weight * entanglement_reg +
            self.correlation_weight * correlation_reg
        )
        
        return total_loss
    
    def _entanglement_regularization(self, entanglement_matrix: torch.Tensor) -> torch.Tensor:
        """Regularize entanglement patterns."""
        # Encourage non-trivial entanglement
        identity = torch.eye(entanglement_matrix.size(-1), device=entanglement_matrix.device)
        identity = identity.unsqueeze(0).expand_as(entanglement_matrix)
        
        # Penalize too much similarity to identity
        identity_penalty = F.mse_loss(entanglement_matrix, identity)
        
        # Encourage some structure (not random)
        random_matrix = torch.randn_like(entanglement_matrix) * 0.1
        random_penalty = F.mse_loss(entanglement_matrix, random_matrix)
        
        # Balance between structure and randomness
        structure_weight = 0.7
        entanglement_reg = (
            structure_weight * identity_penalty +
            (1 - structure_weight) * random_penalty
        )
        
        return entanglement_reg
    
    def _correlation_regularization(
        self,
        entanglement_matrix: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """Regularize correlation with attention patterns."""
        # Ensure entanglement correlates with attention patterns
        batch_size, num_heads, seq_len, seq_len = attention_weights.shape
        
        # Average attention across heads
        avg_attention = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
        
        # Reshape entanglement matrix to match attention
        if entanglement_matrix.dim() == 3:
            entanglement_matrix = entanglement_matrix.unsqueeze(0)
        
        # Compute correlation between entanglement and attention
        correlation = F.cosine_similarity(
            entanglement_matrix.view(-1),
            avg_attention.view(-1),
            dim=0
        )
        
        # Encourage positive correlation
        correlation_reg = -torch.mean(correlation)
        
        return correlation_reg


class UncertaintyLoss(nn.Module):
    """
    Loss function for uncertainty quantification.
    
    Encourages models to provide meaningful uncertainty
    estimates for their predictions.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        uncertainty_weight: float = 0.1,
        calibration_weight: float = 0.05
    ):
        """
        Initialize uncertainty loss.
        
        Args:
            base_loss: Base loss function
            uncertainty_weight: Weight for uncertainty regularization
            calibration_weight: Weight for calibration regularization
        """
        super().__init__()
        
        self.base_loss = base_loss
        self.uncertainty_weight = uncertainty_weight
        self.calibration_weight = calibration_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute uncertainty loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            uncertainty: Uncertainty estimates
            confidence: Confidence estimates (optional)
            
        Returns:
            Total loss value
        """
        # Base task loss
        base_loss = self.base_loss(predictions, targets)
        
        # Uncertainty regularization
        uncertainty_reg = self._uncertainty_regularization(uncertainty)
        
        # Calibration regularization
        calibration_reg = torch.tensor(0.0, device=uncertainty.device)
        if confidence is not None:
            calibration_reg = self._calibration_regularization(uncertainty, confidence, targets)
        
        # Total loss
        total_loss = (
            base_loss +
            self.uncertainty_weight * uncertainty_reg +
            self.calibration_weight * calibration_reg
        )
        
        return total_loss
    
    def _uncertainty_regularization(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Regularize uncertainty estimates."""
        # Encourage reasonable uncertainty levels
        target_uncertainty = torch.ones_like(uncertainty) * 0.5
        
        # Penalize extreme uncertainty values
        extreme_penalty = torch.mean(
            torch.where(
                uncertainty < 0.1,
                (0.1 - uncertainty) ** 2,
                torch.where(uncertainty > 0.9, (uncertainty - 0.9) ** 2, torch.zeros_like(uncertainty))
            )
        )
        
        # Encourage smooth uncertainty patterns
        smoothness_penalty = torch.mean(torch.abs(torch.diff(uncertainty, dim=-1)))
        
        # Target uncertainty regularization
        target_reg = F.mse_loss(uncertainty, target_uncertainty)
        
        return target_reg + extreme_penalty + smoothness_penalty
    
    def _calibration_regularization(
        self,
        uncertainty: torch.Tensor,
        confidence: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Regularize calibration of uncertainty and confidence."""
        # Ensure that high confidence corresponds to low uncertainty
        confidence_uncertainty_correlation = F.cosine_similarity(
            confidence.view(-1),
            (1 - uncertainty).view(-1),
            dim=0
        )
        
        # Encourage positive correlation
        calibration_reg = -confidence_uncertainty_correlation
        
        return calibration_reg
