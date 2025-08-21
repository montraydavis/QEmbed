"""
Context-driven collapse mechanisms for quantum embeddings.

This module implements various strategies for collapsing quantum
superposition states based on contextual information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any


class ContextCollapseLayer(nn.Module):
    """
    Context-driven collapse layer for quantum embeddings.
    
    This layer learns to collapse quantum superposition states
    based on surrounding context, effectively implementing
    quantum measurement in a learned manner.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        context_window: int = 5,
        collapse_strategy: str = "attention",
        temperature: float = 1.0
    ):
        """
        Initialize context collapse layer.
        
        Args:
            embedding_dim: Dimension of input embeddings
            context_window: Size of context window for collapse
            collapse_strategy: Strategy for collapsing ('attention', 'conv', 'rnn')
            temperature: Temperature for softmax in attention
        """
        super().__init__()
        
        # Validate collapse strategy
        valid_strategies = ["attention", "conv", "rnn"]
        if collapse_strategy not in valid_strategies:
            raise ValueError(f"Invalid collapse_strategy: {collapse_strategy}. Must be one of {valid_strategies}")
        
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.collapse_strategy = collapse_strategy
        self.temperature = temperature
        
        if collapse_strategy == "attention":
            self.context_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.context_projection = nn.Linear(embedding_dim, embedding_dim)
            
        elif collapse_strategy == "conv":
            self.context_conv = nn.Conv1d(
                embedding_dim,
                embedding_dim,
                kernel_size=context_window,
                padding=context_window // 2
            )
            self.context_norm = nn.LayerNorm(embedding_dim)
            
        elif collapse_strategy == "rnn":
            self.context_rnn = nn.LSTM(
                embedding_dim,
                embedding_dim // 2,
                bidirectional=True,
                batch_first=True
            )
            self.context_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Final collapse projection
        self.collapse_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        self.collapse_norm = nn.LayerNorm(embedding_dim)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through context collapse layer.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Collapsed embeddings [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Generate context representation
        context = self._generate_context(embeddings, attention_mask)
        
        # Combine original embeddings with context
        combined = torch.cat([embeddings, context], dim=-1)
        
        # Project to final collapsed representation
        collapsed = self.collapse_projection(combined)
        collapsed = self.collapse_norm(collapsed)
        
        return collapsed
    
    def _generate_context(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate contextual representation based on strategy."""
        
        if self.collapse_strategy == "attention":
            return self._attention_context(embeddings, attention_mask)
        elif self.collapse_strategy == "conv":
            return self._conv_context(embeddings)
        elif self.collapse_strategy == "rnn":
            return self._rnn_context(embeddings)
        else:
            raise ValueError(f"Unknown collapse strategy: {self.collapse_strategy}")
    
    def _attention_context(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate context using self-attention."""
        # Apply self-attention
        context, _ = self.context_attention(
            embeddings, embeddings, embeddings,
            attn_mask=attention_mask
        )
        
        # Project context
        context = self.context_projection(context)
        
        return context
    
    def _conv_context(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Generate context using 1D convolution."""
        # Transpose for conv1d: [batch, channels, seq_len]
        embeddings_t = embeddings.transpose(1, 2)
        
        # Apply convolution
        context = self.context_conv(embeddings_t)
        
        # Transpose back: [batch, seq_len, channels]
        context = context.transpose(1, 2)
        
        # Apply normalization
        context = self.context_norm(context)
        
        return context
    
    def _rnn_context(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Generate context using bidirectional LSTM."""
        # Apply RNN
        context, _ = self.context_rnn(embeddings)
        
        # Project context
        context = self.context_projection(context)
        
        return context


class AdaptiveCollapseLayer(nn.Module):
    """
    Adaptive collapse layer that learns when to collapse.
    
    This layer learns to predict the optimal collapse strategy
    for each token based on its context and uncertainty.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_strategies: int = 3,
        temperature: float = 1.0
    ):
        """
        Initialize adaptive collapse layer.
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_strategies: Number of available collapse strategies
            temperature: Temperature for strategy selection
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_strategies = num_strategies
        self.temperature = temperature
        
        # Strategy selection network
        self.strategy_selector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_strategies)
        )
        
        # Strategy-specific collapse layers
        self.strategy_layers = nn.ModuleList([
            ContextCollapseLayer(embedding_dim, collapse_strategy="attention"),
            ContextCollapseLayer(embedding_dim, collapse_strategy="conv"),
            ContextCollapseLayer(embedding_dim, collapse_strategy="rnn")
        ])
        
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with adaptive strategy selection.
        
        Returns:
            Tuple of (collapsed_embeddings, strategy_weights)
        """
        # Predict strategy weights
        strategy_logits = self.strategy_selector(embeddings)
        strategy_weights = F.softmax(strategy_logits / self.temperature, dim=-1)
        
        # Apply each strategy
        strategy_outputs = []
        for i, layer in enumerate(self.strategy_layers):
            output = layer(embeddings, attention_mask)
            strategy_outputs.append(output)
        
        # Combine outputs based on strategy weights
        strategy_outputs = torch.stack(strategy_outputs, dim=2)
        collapsed = torch.sum(
            strategy_outputs * strategy_weights.unsqueeze(-1),
            dim=2
        )
        
        return collapsed, strategy_weights
