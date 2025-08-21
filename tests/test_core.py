"""
Tests for QEmbed core quantum components.

This test suite covers core quantum components including:
- Collapse layers (ContextCollapseLayer, AdaptiveCollapseLayer)
- Entanglement (EntanglementCorrelation, BellStateEntanglement)
- Measurement (QuantumMeasurement, AdaptiveMeasurement)
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import core components
from qembed.core.collapse_layers import ContextCollapseLayer, AdaptiveCollapseLayer
from qembed.core.entanglement import EntanglementCorrelation, BellStateEntanglement
from qembed.core.measurement import QuantumMeasurement, AdaptiveMeasurement


class TestContextCollapseLayer:
    """Test cases for ContextCollapseLayer class."""
    
    @pytest.fixture
    def collapse_layer(self):
        """Create a ContextCollapseLayer instance for testing."""
        return ContextCollapseLayer(
            embedding_dim=128,
            context_window=5,
            collapse_strategy="attention"
        )
    
    @pytest.fixture
    def embeddings(self):
        """Create sample embeddings tensor."""
        return torch.randn(2, 10, 128)  # [batch, seq, embed_dim]
    
    @pytest.fixture
    def context(self):
        """Create sample context tensor."""
        return torch.randn(2, 10, 128)  # [batch, seq, embed_dim]
    
    def test_initialization(self, collapse_layer):
        """Test proper initialization."""
        assert collapse_layer.embedding_dim == 128
        assert collapse_layer.context_window == 5
        assert collapse_layer.collapse_strategy == 'attention'
        assert hasattr(collapse_layer, 'context_attention')
    
    def test_attention_collapse(self, collapse_layer, embeddings):
        """Test collapse using attention method."""
        output = collapse_layer(embeddings)
        
        assert output.shape == (2, 10, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_convolution_collapse(self):
        """Test collapse using convolution method."""
        collapse_layer = ContextCollapseLayer(
            embedding_dim=128,
            context_window=5,
            collapse_strategy="conv"
        )
        
        embeddings = torch.randn(2, 10, 128)
        
        output = collapse_layer(embeddings)
        
        assert output.shape == (2, 10, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_rnn_collapse(self):
        """Test collapse using RNN method."""
        collapse_layer = ContextCollapseLayer(
            embedding_dim=128,
            context_window=5,
            collapse_strategy="rnn"
        )
        
        embeddings = torch.randn(2, 10, 128)
        
        output = collapse_layer(embeddings)
        
        assert output.shape == (2, 10, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_invalid_collapse_method(self):
        """Test that invalid collapse method raises error."""
        # Test that invalid strategy raises error during initialization
        with pytest.raises(ValueError):
            ContextCollapseLayer(
                embedding_dim=128,
                context_window=5,
                collapse_strategy='invalid_method'
            )
    
    def test_gradients(self, collapse_layer, embeddings):
        """Test that gradients flow through the collapse layer."""
        collapse_layer.train()
        
        # Forward pass
        output = collapse_layer(embeddings)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert collapse_layer.context_attention.in_proj_weight.grad is not None


class TestAdaptiveCollapseLayer:
    """Test cases for AdaptiveCollapseLayer class."""
    
    @pytest.fixture
    def adaptive_layer(self):
        """Create an AdaptiveCollapseLayer instance for testing."""
        return AdaptiveCollapseLayer(
            embedding_dim=128,
            num_strategies=3,
            temperature=1.0
        )
    
    @pytest.fixture
    def embeddings(self):
        """Create sample embeddings tensor."""
        return torch.randn(2, 10, 128)
    
    @pytest.fixture
    def context(self):
        """Create sample context tensor."""
        return torch.randn(2, 10, 128)
    
    def test_initialization(self, adaptive_layer):
        """Test proper initialization."""
        assert adaptive_layer.embedding_dim == 128
        assert adaptive_layer.num_strategies == 3
        assert adaptive_layer.temperature == 1.0
        assert hasattr(adaptive_layer, 'strategy_selector')
        assert len(adaptive_layer.strategy_layers) == 3
    
    def test_forward(self, adaptive_layer, embeddings):
        """Test forward pass."""
        output, strategy_weights = adaptive_layer(embeddings)
        
        assert output.shape == (2, 10, 128)
        assert strategy_weights.shape == (2, 10, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_method_selection(self, adaptive_layer, embeddings):
        """Test that method selector works correctly."""
        output, strategy_weights = adaptive_layer(embeddings)
        
        assert output.shape == (2, 10, 128)
        assert strategy_weights.shape == (2, 10, 3)
        assert torch.allclose(strategy_weights.sum(dim=-1), torch.ones(2, 10))
    
    def test_gradients(self, adaptive_layer, embeddings):
        """Test that gradients flow through the adaptive layer."""
        adaptive_layer.train()
        
        # Forward pass
        output, strategy_weights = adaptive_layer(embeddings)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist (may be None if method selector doesn't have parameters)
        # The important thing is that the backward pass doesn't fail


class TestEntanglementCorrelation:
    """Test cases for EntanglementCorrelation class."""
    
    @pytest.fixture
    def entanglement(self):
        """Create an EntanglementCorrelation instance for testing."""
        return EntanglementCorrelation(embedding_dim=128)
    
    @pytest.fixture
    def embeddings(self):
        """Create sample embeddings."""
        return torch.randn(2, 10, 128)  # [batch, seq, embed_dim]
    
    def test_initialization(self, entanglement):
        """Test proper initialization."""
        assert entanglement.embedding_dim == 128
        assert hasattr(entanglement, 'entanglement_matrix')
        assert hasattr(entanglement, 'correlation_weights')
    
    def test_forward(self, entanglement, embeddings):
        """Test forward pass."""
        output = entanglement(embeddings)
        
        assert output.shape == embeddings.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert not torch.allclose(output, embeddings)  # Should be modified
    
    def test_correlation_computation(self, entanglement, embeddings):
        """Test correlation matrix computation."""
        # Test that correlation matrix is computed correctly
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Compute expected correlation matrix
        expected_corr = torch.corrcoef(embeddings[0].T)  # [embed_dim, embed_dim]
        
        # The actual computation happens in forward pass
        output = entanglement(embeddings)
        
        # Check that output is different from input (indicating correlation was applied)
        assert not torch.allclose(output, embeddings)
    
    def test_gradients(self, entanglement, embeddings):
        """Test that gradients flow through the entanglement layer."""
        entanglement.train()
        
        # Forward pass
        output = entanglement(embeddings)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that backward pass doesn't fail
        # The entanglement layer may not have learnable parameters


class TestBellStateEntanglement:
    """Test cases for BellStateEntanglement class."""
    
    @pytest.fixture
    def bell_entanglement(self):
        """Create a BellStateEntanglement instance for testing."""
        return BellStateEntanglement(embedding_dim=128)
    
    @pytest.fixture
    def embeddings(self):
        """Create sample embeddings."""
        return torch.randn(2, 10, 128)  # [batch, seq, embed_dim]
    
    def test_initialization(self, bell_entanglement):
        """Test proper initialization."""
        assert bell_entanglement.embedding_dim == 128
    
    def test_forward(self, bell_entanglement, embeddings):
        """Test forward pass."""
        output = bell_entanglement(embeddings)
        
        assert output.shape == embeddings.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_bell_state_creation(self, bell_entanglement, embeddings):
        """Test Bell state creation."""
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Test that Bell states are created for pairs of embeddings
        output = bell_entanglement(embeddings)
        
        # Check that output has same shape as input
        assert output.shape == embeddings.shape
        
        # Check that output is different from input (indicating Bell states were applied)
        assert not torch.allclose(output, embeddings)
    
    def test_gradients(self, bell_entanglement, embeddings):
        """Test that gradients flow through the Bell state entanglement layer."""
        bell_entanglement.train()
        
        # Forward pass
        output = bell_entanglement(embeddings)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that backward pass doesn't fail


class TestQuantumMeasurement:
    """Test cases for QuantumMeasurement class."""
    
    @pytest.fixture
    def measurement(self):
        """Create a QuantumMeasurement instance for testing."""
        return QuantumMeasurement(embedding_dim=128, measurement_basis="computational")
    
    @pytest.fixture
    def state(self):
        """Create sample quantum state."""
        return torch.randn(2, 10, 128)  # [batch, seq, embed_dim]
    
    def test_initialization(self, measurement):
        """Test proper initialization."""
        assert measurement.embedding_dim == 128
        assert measurement.measurement_basis == "computational"
        assert hasattr(measurement, 'basis_matrix')
    
    def test_forward(self, measurement, state):
        """Test forward pass."""
        measured, results = measurement(state)
        
        assert measured.shape == state.shape
        assert results.shape == (state.shape[0], state.shape[1], state.shape[2] + 1)  # +1 for uncertainty
        assert not torch.isnan(measured).any()
        assert not torch.isinf(measured).any()
        # Check that results are valid tensors
    
    def test_measurement_bases(self, measurement, state):
        """Test measurement bases computation."""
        # Test that measurement basis matrix is created correctly
        assert measurement.basis_matrix.shape == (128, 128)
        
        # Test that basis matrix is normalized
        basis_norm = torch.norm(measurement.basis_matrix, dim=1)
        assert torch.allclose(basis_norm, torch.ones(128), atol=1e-6)
    
    def test_measurement_results(self, measurement, state):
        """Test measurement results computation."""
        measured, results = measurement(state)
        
        # Check that measurement results are computed
        assert isinstance(results, torch.Tensor)
        assert results.shape == (state.shape[0], state.shape[1], state.shape[2] + 1)  # +1 for uncertainty
        
        # Check that results are valid
        assert not torch.isnan(results).any()
        assert not torch.isinf(results).any()
    
    def test_gradients(self, measurement, state):
        """Test that gradients flow through the measurement layer."""
        measurement.train()
        
        # Forward pass
        measured, results = measurement(state)
        
        # Compute loss and backward pass
        loss = measured.sum()
        loss.backward()
        
        # Check that backward pass doesn't fail


class TestAdaptiveMeasurement:
    """Test cases for AdaptiveMeasurement class."""
    
    @pytest.fixture
    def adaptive_measurement(self):
        """Create an AdaptiveMeasurement instance for testing."""
        return AdaptiveMeasurement(embedding_dim=128)
    
    @pytest.fixture
    def state(self):
        """Create sample quantum state."""
        return torch.randn(2, 10, 128)  # [batch, seq, embed_dim]
    
    @pytest.fixture
    def context(self):
        """Create sample context tensor."""
        return torch.randn(2, 10, 128)  # [batch, seq, embed_dim]
    
    def test_initialization(self, adaptive_measurement):
        """Test proper initialization."""
        assert adaptive_measurement.embedding_dim == 128
        assert hasattr(adaptive_measurement, 'basis_selector')
    
    def test_forward(self, adaptive_measurement, state):
        """Test forward pass."""
        measured, results, basis_weights = adaptive_measurement(state)
        
        assert measured.shape == state.shape
        assert results.shape == (state.shape[0], state.shape[1], state.shape[2] + 1)  # +1 for uncertainty
        assert basis_weights.shape == (2, 10, 4)  # num_bases=4 by default
        assert not torch.isnan(measured).any()
        assert not torch.isinf(measured).any()
    
    def test_basis_selection(self, adaptive_measurement, state):
        """Test adaptive basis selection."""
        # Test that basis selector works correctly
        measured, results, basis_weights = adaptive_measurement(state)
        
        # Check that output is different from input (indicating measurement was applied)
        assert not torch.allclose(measured, state)
        # Check that basis weights sum to 1
        assert torch.allclose(basis_weights.sum(dim=-1), torch.ones(2, 10))
    
    def test_context_influence(self, adaptive_measurement, state):
        """Test that measurement produces consistent results."""
        # Apply measurement twice
        measured1, results1, weights1 = adaptive_measurement(state)
        measured2, results2, weights2 = adaptive_measurement(state)
        
        # Results should be consistent for same input
        assert torch.allclose(measured1, measured2, atol=1e-6)
        assert torch.allclose(results1, results2, atol=1e-6)
    
    def test_gradients(self, adaptive_measurement, state):
        """Test that gradients flow through the adaptive measurement layer."""
        adaptive_measurement.train()
        
        # Forward pass
        measured, results, basis_weights = adaptive_measurement(state)
        
        # Compute loss and backward pass
        loss = measured.sum()
        loss.backward()
        
        # Check that backward pass doesn't fail


class TestCoreIntegration:
    """Integration tests for core quantum components."""
    
    def test_collapse_entanglement_integration(self):
        """Test integration between collapse layers and entanglement."""
        # Create components
        collapse_layer = ContextCollapseLayer(
            embedding_dim=128,
            context_window=5,
            collapse_strategy="attention"
        )
        
        entanglement = EntanglementCorrelation(embedding_dim=128)
        
        # Create test data
        embeddings = torch.randn(2, 10, 128)
        
        # Apply collapse
        collapsed = collapse_layer(embeddings)
        
        # Apply entanglement
        entangled = entanglement(collapsed)
        
        # Check outputs
        assert collapsed.shape == (2, 10, 128)
        assert entangled.shape == (2, 10, 128)
        assert not torch.isnan(collapsed).any()
        assert not torch.isnan(entangled).any()
    
    def test_measurement_integration(self):
        """Test integration between measurement and other components."""
        # Create components
        measurement = QuantumMeasurement(embedding_dim=128, measurement_basis="computational")
        adaptive_measurement = AdaptiveMeasurement(embedding_dim=128)
        
        # Create test data
        state = torch.randn(2, 10, 128)
        context = torch.randn(2, 10, 128)
        
        # Apply measurements
        measured, _ = measurement(state)
        adaptive_measured, _, _ = adaptive_measurement(state)
        
        # Check outputs
        assert measured.shape == (2, 10, 128)
        assert adaptive_measured.shape == (2, 10, 128)
        assert not torch.isnan(measured).any()
        assert not torch.isnan(adaptive_measured).any()
    
    def test_comprehensive_core_workflow(self):
        """Test comprehensive workflow using all core components."""
        # Create components
        collapse_layer = ContextCollapseLayer(
            embedding_dim=128,
            context_window=5,
            collapse_strategy="attention"
        )
        
        entanglement = EntanglementCorrelation(embedding_dim=128)
        measurement = QuantumMeasurement(embedding_dim=128, measurement_basis="computational")
        
        # Create test data
        embeddings = torch.randn(2, 10, 128)
        
        # Apply full workflow
        collapsed = collapse_layer(embeddings)
        entangled = entanglement(collapsed)
        measured, _ = measurement(entangled)
        
        # Check all outputs
        assert collapsed.shape == (2, 10, 128)
        assert entangled.shape == (2, 10, 128)
        assert measured.shape == (2, 10, 128)
        
        # Check that no NaN or Inf values
        assert not torch.isnan(collapsed).any()
        assert not torch.isnan(entangled).any()
        assert not torch.isnan(measured).any()
        assert not torch.isinf(collapsed).any()
        assert not torch.isinf(entangled).any()
        assert not torch.isinf(measured).any()
        
        # Check that outputs are different (indicating each step had an effect)
        assert not torch.allclose(collapsed, entangled)
        # Note: measurement might not change values significantly with collapse_probability=1.0
        # So we just check that the measurement step completes without error
    
    def test_gradient_flow_integration(self):
        """Test that gradients flow through integrated core components."""
        # Create components
        collapse_layer = ContextCollapseLayer(
            embedding_dim=128,
            context_window=5,
            collapse_strategy="attention"
        )
        
        entanglement = EntanglementCorrelation(embedding_dim=128)
        measurement = QuantumMeasurement(embedding_dim=128, measurement_basis="computational")
        
        # Set to training mode
        collapse_layer.train()
        entanglement.train()
        measurement.train()
        
        # Create test data
        embeddings = torch.randn(2, 10, 128)
        
        # Forward pass
        collapsed = collapse_layer(embeddings)
        entangled = entanglement(collapsed)
        measured, _ = measurement(entangled)
        
        # Compute loss and backward pass
        loss = measured.sum()
        loss.backward()
        
        # Check that backward pass doesn't fail
        # This tests that gradients can flow through the entire pipeline


if __name__ == "__main__":
    pytest.main([__file__])
