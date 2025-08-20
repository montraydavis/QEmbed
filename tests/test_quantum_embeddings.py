"""
Tests for quantum embeddings module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from qembed.core.quantum_embeddings import QuantumEmbeddings
from qembed.core.collapse_layers import ContextCollapseLayer, AdaptiveCollapseLayer
from qembed.core.entanglement import EntanglementCorrelation, BellStateEntanglement
from qembed.core.measurement import QuantumMeasurement, AdaptiveMeasurement


class TestQuantumEmbeddings:
    """Test cases for QuantumEmbeddings class."""
    
    @pytest.fixture
    def embeddings(self):
        """Create a QuantumEmbeddings instance for testing."""
        return QuantumEmbeddings(
            vocab_size=1000,
            embedding_dim=128,
            num_states=4,
            dropout=0.1
        )
    
    @pytest.fixture
    def input_ids(self):
        """Create sample input IDs."""
        return torch.randint(0, 1000, (2, 10))
    
    def test_initialization(self, embeddings):
        """Test proper initialization of QuantumEmbeddings."""
        assert embeddings.vocab_size == 1000
        assert embeddings.embedding_dim == 128
        assert embeddings.num_states == 4
        assert embeddings.dropout == 0.1
        
        # Check embedding layers
        assert embeddings.token_embeddings.num_embeddings == 1000
        assert embeddings.token_embeddings.embedding_dim == 128
        assert embeddings.state_embeddings.num_embeddings == 1000
        assert embeddings.state_embeddings.embedding_dim == 128 * 4
        
        # Check collapse layer
        assert isinstance(embeddings.collapse_layer, ContextCollapseLayer)
    
    def test_forward_superposition(self, embeddings, input_ids):
        """Test forward pass without collapse (superposition state)."""
        output = embeddings(input_ids, collapse=False)
        
        # Output should have shape [batch, seq, embedding_dim]
        assert output.shape == (2, 10, 128)
        
        # Should not be the same as token embeddings (due to superposition)
        token_embeds = embeddings.token_embeddings(input_ids)
        assert not torch.allclose(output, token_embeds)
    
    def test_forward_collapse(self, embeddings, input_ids):
        """Test forward pass with collapse."""
        output = embeddings(input_ids, collapse=True)
        
        # Output should have shape [batch, seq, embedding_dim]
        assert output.shape == (2, 10, 128)
    
    def test_forward_with_context(self, embeddings, input_ids):
        """Test forward pass with context tensor."""
        context = torch.randn(2, 10, 128)
        output = embeddings(input_ids, context=context, collapse=True)
        
        assert output.shape == (2, 10, 128)
    
    def test_create_superposition(self, embeddings, input_ids):
        """Test superposition creation."""
        superposition = embeddings._create_superposition(input_ids)
        
        # Should have shape [batch, seq, num_states, embedding_dim]
        assert superposition.shape == (2, 10, 4, 128)
        
        # Each state should be different
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(
                    superposition[:, :, i, :], 
                    superposition[:, :, j, :]
                )
    
    def test_collapse_superposition(self, embeddings, input_ids):
        """Test superposition collapse."""
        superposition = embeddings._create_superposition(input_ids)
        context = torch.randn(2, 10, 128)
        
        collapsed = embeddings._collapse_superposition(superposition, context)
        
        assert collapsed.shape == (2, 10, 128)
    
    def test_get_uncertainty(self, embeddings, input_ids):
        """Test uncertainty calculation."""
        uncertainty = embeddings.get_uncertainty(input_ids)
        
        assert uncertainty.shape == (2, 10)
        assert torch.all(uncertainty >= 0)  # Uncertainty should be non-negative
        assert torch.all(uncertainty <= 1)  # Uncertainty should be <= 1
    
    def test_different_vocab_sizes(self):
        """Test initialization with different vocabulary sizes."""
        for vocab_size in [100, 1000, 10000]:
            embeddings = QuantumEmbeddings(
                vocab_size=vocab_size,
                embedding_dim=64,
                num_states=3
            )
            assert embeddings.token_embeddings.num_embeddings == vocab_size
            assert embeddings.state_embeddings.num_embeddings == vocab_size
    
    def test_different_embedding_dims(self):
        """Test initialization with different embedding dimensions."""
        for embedding_dim in [32, 64, 128, 256]:
            embeddings = QuantumEmbeddings(
                vocab_size=1000,
                embedding_dim=embedding_dim,
                num_states=4
            )
            assert embeddings.token_embeddings.embedding_dim == embedding_dim
            assert embeddings.state_embeddings.embedding_dim == embedding_dim * 4
    
    def test_different_num_states(self):
        """Test initialization with different numbers of states."""
        for num_states in [2, 3, 4, 8]:
            embeddings = QuantumEmbeddings(
                vocab_size=1000,
                embedding_dim=128,
                num_states=num_states
            )
            assert embeddings.num_states == num_states
            assert embeddings.state_embeddings.embedding_dim == 128 * num_states
    
    def test_gradients(self, embeddings, input_ids):
        """Test that gradients flow through the model."""
        embeddings.train()
        output = embeddings(input_ids, collapse=False)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert embeddings.token_embeddings.weight.grad is not None
        assert embeddings.state_embeddings.weight.grad is not None
        assert embeddings.collapse_layer.attention.weight.grad is not None


class TestContextCollapseLayer:
    """Test cases for ContextCollapseLayer class."""
    
    @pytest.fixture
    def collapse_layer(self):
        """Create a ContextCollapseLayer instance for testing."""
        return ContextCollapseLayer(
            embedding_dim=128,
            num_states=4,
            collapse_method='attention'
        )
    
    @pytest.fixture
    def superposition(self):
        """Create sample superposition tensor."""
        return torch.randn(2, 10, 4, 128)
    
    @pytest.fixture
    def context(self):
        """Create sample context tensor."""
        return torch.randn(2, 10, 128)
    
    def test_initialization(self, collapse_layer):
        """Test proper initialization."""
        assert collapse_layer.embedding_dim == 128
        assert collapse_layer.num_states == 4
        assert collapse_layer.collapse_method == 'attention'
        
        # Check that attention layer exists
        assert hasattr(collapse_layer, 'attention')
    
    def test_attention_collapse(self, collapse_layer, superposition, context):
        """Test collapse using attention method."""
        output = collapse_layer(superposition, context)
        
        assert output.shape == (2, 10, 128)
    
    def test_convolution_collapse(self):
        """Test collapse using convolution method."""
        collapse_layer = ContextCollapseLayer(
            embedding_dim=128,
            num_states=4,
            collapse_method='convolution'
        )
        
        superposition = torch.randn(2, 10, 4, 128)
        context = torch.randn(2, 10, 128)
        
        output = collapse_layer(superposition, context)
        assert output.shape == (2, 10, 128)
    
    def test_rnn_collapse(self):
        """Test collapse using RNN method."""
        collapse_layer = ContextCollapseLayer(
            embedding_dim=128,
            num_states=4,
            collapse_method='rnn'
        )
        
        superposition = torch.randn(2, 10, 4, 128)
        context = torch.randn(2, 10, 128)
        
        output = collapse_layer(superposition, context)
        assert output.shape == (2, 10, 128)
    
    def test_invalid_collapse_method(self):
        """Test that invalid collapse method raises error."""
        with pytest.raises(ValueError):
            ContextCollapseLayer(
                embedding_dim=128,
                num_states=4,
                collapse_method='invalid_method'
            )


class TestAdaptiveCollapseLayer:
    """Test cases for AdaptiveCollapseLayer class."""
    
    @pytest.fixture
    def adaptive_layer(self):
        """Create an AdaptiveCollapseLayer instance for testing."""
        return AdaptiveCollapseLayer(
            embedding_dim=128,
            num_states=4,
            collapse_methods=['attention', 'convolution', 'rnn']
        )
    
    def test_initialization(self, adaptive_layer):
        """Test proper initialization."""
        assert adaptive_layer.embedding_dim == 128
        assert adaptive_layer.num_states == 4
        assert len(adaptive_layer.collapse_methods) == 3
        assert 'attention' in adaptive_layer.collapse_methods
        assert 'convolution' in adaptive_layer.collapse_methods
        assert 'rnn' in adaptive_layer.collapse_methods
        
        # Check that method selector exists
        assert hasattr(adaptive_layer, 'method_selector')
    
    def test_forward(self, adaptive_layer):
        """Test forward pass."""
        superposition = torch.randn(2, 10, 4, 128)
        context = torch.randn(2, 10, 128)
        
        output = adaptive_layer(superposition, context)
        
        assert output.shape == (2, 10, 128)


class TestEntanglementCorrelation:
    """Test cases for EntanglementCorrelation class."""
    
    @pytest.fixture
    def entanglement(self):
        """Create an EntanglementCorrelation instance for testing."""
        return EntanglementCorrelation(embedding_dim=128)
    
    @pytest.fixture
    def embeddings(self):
        """Create sample embeddings."""
        return torch.randn(2, 10, 128)
    
    def test_initialization(self, entanglement):
        """Test proper initialization."""
        assert entanglement.embedding_dim == 128
        assert hasattr(entanglement, 'correlation_matrix')
    
    def test_forward(self, entanglement, embeddings):
        """Test forward pass."""
        output = entanglement(embeddings)
        
        assert output.shape == embeddings.shape
        assert not torch.allclose(output, embeddings)  # Should be modified


class TestBellStateEntanglement:
    """Test cases for BellStateEntanglement class."""
    
    @pytest.fixture
    def bell_entanglement(self):
        """Create a BellStateEntanglement instance for testing."""
        return BellStateEntanglement(embedding_dim=128)
    
    def test_initialization(self, bell_entanglement):
        """Test proper initialization."""
        assert bell_entanglement.embedding_dim == 128
    
    def test_forward(self, bell_entanglement):
        """Test forward pass."""
        embeddings = torch.randn(2, 10, 128)
        output = bell_entanglement(embeddings)
        
        assert output.shape == embeddings.shape


class TestQuantumMeasurement:
    """Test cases for QuantumMeasurement class."""
    
    @pytest.fixture
    def measurement(self):
        """Create a QuantumMeasurement instance for testing."""
        return QuantumMeasurement(embedding_dim=128, num_bases=3)
    
    @pytest.fixture
    def state(self):
        """Create sample quantum state."""
        return torch.randn(2, 10, 128)
    
    def test_initialization(self, measurement):
        """Test proper initialization."""
        assert measurement.embedding_dim == 128
        assert measurement.num_bases == 3
        assert hasattr(measurement, 'measurement_bases')
    
    def test_forward(self, measurement, state):
        """Test forward pass."""
        output = measurement(state)
        
        assert output.shape == state.shape
        assert hasattr(output, 'measurement_results')


class TestAdaptiveMeasurement:
    """Test cases for AdaptiveMeasurement class."""
    
    @pytest.fixture
    def adaptive_measurement(self):
        """Create an AdaptiveMeasurement instance for testing."""
        return AdaptiveMeasurement(embedding_dim=128)
    
    def test_initialization(self, adaptive_measurement):
        """Test proper initialization."""
        assert adaptive_measurement.embedding_dim == 128
        assert hasattr(adaptive_measurement, 'basis_selector')
    
    def test_forward(self, adaptive_measurement):
        """Test forward pass."""
        state = torch.randn(2, 10, 128)
        context = torch.randn(2, 10, 128)
        
        output = adaptive_measurement(state, context)
        
        assert output.shape == state.shape


if __name__ == "__main__":
    pytest.main([__file__])
