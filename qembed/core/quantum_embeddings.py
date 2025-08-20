"""
Quantum Embeddings: Basic quantum superposition embeddings.

This module implements quantum-inspired embeddings that can exist in
superposition states until measured in a specific context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class QuantumEmbeddings(nn.Module):
    """
    Quantum-inspired embeddings that maintain superposition states.
    
    Each token can exist in multiple semantic states simultaneously
    until measured in a specific context, allowing for better
    representation of polysemy and contextual ambiguity.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_states: int = 4,
        superposition_strength: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize quantum embeddings.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors
            num_states: Number of possible quantum states per token
            superposition_strength: Strength of superposition mixing
            device: Device to place tensors on
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_states = num_states
        self.superposition_strength = superposition_strength
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create base embeddings for each quantum state
        self.state_embeddings = nn.Parameter(
            torch.randn(vocab_size, num_states, embedding_dim) * 0.1
        )
        
        # Superposition mixing matrix
        self.superposition_matrix = nn.Parameter(
            torch.eye(num_states) + torch.randn(num_states, num_states) * superposition_strength
        )
        
        # Normalize superposition matrix
        self.superposition_matrix.data = F.normalize(
            self.superposition_matrix.data, p=2, dim=1
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        collapse: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through quantum embeddings.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            context: Optional context tensor for measurement
            collapse: Whether to collapse superposition to single state
            
        Returns:
            Quantum embeddings [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # Get base state embeddings
        # [batch_size, seq_len, num_states, embedding_dim]
        state_embeds = self.state_embeddings[input_ids]
        
        if collapse and context is not None:
            # Measure/collapse superposition based on context
            return self._collapse_superposition(state_embeds, context)
        else:
            # Return superposition state
            return self._create_superposition(state_embeds)
    
    def _create_superposition(self, state_embeds: torch.Tensor) -> torch.Tensor:
        """Create superposition of quantum states."""
        # Apply superposition mixing
        # [batch_size, seq_len, num_states, embedding_dim]
        mixed_states = torch.einsum('bsne,ef->bsnf', state_embeds, self.superposition_matrix)
        
        # Combine states with equal weights (could be learned)
        weights = torch.ones(mixed_states.shape[:-1], device=mixed_states.device) / self.num_states
        weights = weights.unsqueeze(-1)
        
        # Weighted combination of states
        superposition = torch.sum(mixed_states * weights, dim=2)
        
        return superposition
    
    def _collapse_superposition(
        self, 
        state_embeds: torch.Tensor, 
        context: torch.Tensor
    ) -> torch.Tensor:
        """Collapse superposition based on context."""
        batch_size, seq_len, num_states, embedding_dim = state_embeds.shape
        
        # Compute context similarity with each state
        # [batch_size, seq_len, num_states]
        context_expanded = context.unsqueeze(2).expand(-1, -1, num_states, -1)
        similarities = F.cosine_similarity(
            state_embeds, context_expanded, dim=-1
        )
        
        # Convert similarities to probabilities
        probabilities = F.softmax(similarities / 0.1, dim=-1)
        
        # Sample single state based on probabilities
        if self.training:
            # During training, use weighted combination
            collapsed = torch.sum(state_embeds * probabilities.unsqueeze(-1), dim=2)
        else:
            # During inference, sample single state
            state_indices = torch.multinomial(probabilities.view(-1, num_states), 1)
            state_indices = state_indices.view(batch_size, seq_len)
            
            # Gather selected states
            batch_indices = torch.arange(batch_size, device=state_embeds.device).unsqueeze(1)
            seq_indices = torch.arange(seq_len, device=state_embeds.device).unsqueeze(0)
            collapsed = state_embeds[batch_indices, seq_indices, state_indices]
        
        return collapsed
    
    def get_uncertainty(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty for each token based on state variance."""
        state_embeds = self.state_embeddings[input_ids]
        
        # Compute variance across states
        mean_embed = torch.mean(state_embeds, dim=2)
        variance = torch.var(state_embeds, dim=2)
        
        # Uncertainty as normalized variance
        uncertainty = torch.norm(variance, dim=-1)
        
        return uncertainty
