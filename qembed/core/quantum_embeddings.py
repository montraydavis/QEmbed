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

# ADD these imports at the top
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers import BertConfig


class QuantumEmbeddings(nn.Module):
    """
    Quantum-inspired embeddings that maintain superposition states.
    
    Each token can exist in multiple semantic states simultaneously
    until measured in a specific context, allowing for better
    representation of polysemy and contextual ambiguity.
    """
    
    def __init__(
        self,
        config: Optional[BertConfig] = None,  # ADD: BERT config support
        vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        num_states: int = 4,
        superposition_strength: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize quantum embeddings.
        
        Args:
            config: BERT configuration object (preferred)
            vocab_size: Size of the vocabulary (fallback)
            embedding_dim: Dimension of the embedding vectors (fallback)
            num_states: Number of possible quantum states per token
            superposition_strength: Strength of superposition mixing
            device: Device to place tensors on
        """
        super().__init__()
        
        # Handle BERT config
        if config is not None:
            self.vocab_size = config.vocab_size
            self.embedding_dim = config.hidden_size
            self.max_position_embeddings = getattr(config, 'max_position_embeddings', 512)
            self.type_vocab_size = getattr(config, 'type_vocab_size', 2)
        else:
            if vocab_size is None or embedding_dim is None:
                raise ValueError("Must provide either config or both vocab_size and embedding_dim")
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.max_position_embeddings = 512  # Default
            self.type_vocab_size = 2  # Default
        
        self.num_states = num_states
        self.superposition_strength = superposition_strength
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ADD: BERT-compatible embedding components
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.embedding_dim)
        self.token_type_embeddings = nn.Embedding(self.type_vocab_size, self.embedding_dim)
        
        # ADD: Layer normalization and dropout (BERT standard)
        self.LayerNorm = nn.LayerNorm(self.embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        
        # KEEP existing quantum components
        self.state_embeddings = nn.Parameter(
            torch.randn(self.vocab_size, num_states, self.embedding_dim) * 0.1
        )
        
        # Superposition mixing matrix [num_states, num_states]
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
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        collapse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        BERT-compatible forward pass through quantum embeddings.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            context: Optional context tensor for measurement
            collapse: Whether to collapse superposition to single state
            
        Returns:
            Tuple of (quantum_embeddings, uncertainty_scores)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Get quantum superposition embeddings
        state_embeds = self.state_embeddings[input_ids]  # [batch, seq_len, num_states, embed_dim]
        
        if collapse and context is not None:
            # Measure/collapse superposition based on context
            word_embeddings = self._collapse_superposition(state_embeds, context)
            uncertainty = torch.zeros(batch_size, seq_len, device=input_ids.device)
        else:
            # Return superposition state
            word_embeddings = self._create_superposition(state_embeds)
            uncertainty = self.get_uncertainty_from_states(state_embeds)
        
        # Add position and token type embeddings (BERT standard)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        
        # Apply layer normalization and dropout (BERT standard)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings, uncertainty
    
    def get_uncertainty_from_states(self, state_embeds: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty from quantum states."""
        # Compute variance across quantum states
        variance = torch.var(state_embeds, dim=2)  # [batch, seq_len, embed_dim]
        uncertainty = torch.norm(variance, dim=-1)  # [batch, seq_len]
        return uncertainty
    
    def _create_superposition(self, state_embeds: torch.Tensor) -> torch.Tensor:
        """Create superposition of quantum states."""
        # Simple weighted combination of states (avoiding complex einsum for now)
        # [batch_size, seq_len, num_states, embedding_dim]
        
        # Apply superposition matrix to mix states
        # Reshape for matrix multiplication: [batch*seq, num_states, embed_dim]
        batch_size, seq_len, num_states, embed_dim = state_embeds.shape
        reshaped_states = state_embeds.view(-1, num_states, embed_dim)
        
        # Apply superposition matrix: [batch*seq, num_states, embed_dim]
        mixed_states = torch.bmm(
            self.superposition_matrix.unsqueeze(0).expand(reshaped_states.shape[0], -1, -1),
            reshaped_states
        )
        
        # Reshape back: [batch_size, seq_len, num_states, embedding_dim]
        mixed_states = mixed_states.view(batch_size, seq_len, num_states, embed_dim)
        
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
    
    def forward_legacy(
        self,
        input_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        collapse: bool = False
    ) -> torch.Tensor:
        """
        Legacy forward pass through quantum embeddings (maintains backward compatibility).
        
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
