"""
Tests for quantum embeddings module.
"""

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig
from qembed.core.quantum_embeddings import QuantumEmbeddings
from qembed.core.collapse_layers import ContextCollapseLayer, AdaptiveCollapseLayer
from qembed.core.measurement import QuantumMeasurement, AdaptiveMeasurement
from qembed.core.entanglement import EntanglementCorrelation


class TestQuantumEmbeddings:
    """Test quantum embeddings functionality."""
    
    @pytest.fixture
    def config(self):
        """Create a BERT config for testing."""
        return BertConfig(
            vocab_size=1000,
            hidden_size=128,
            max_position_embeddings=512,
            type_vocab_size=2
        )
    
    @pytest.fixture
    def embeddings(self, config):
        """Create quantum embeddings instance."""
        return QuantumEmbeddings(config=config)
    
    @pytest.fixture
    def input_ids(self):
        """Create input token IDs."""
        return torch.randint(0, 1000, (2, 10))
    
    @pytest.fixture
    def attention_mask(self):
        """Create attention mask."""
        return torch.ones(2, 10, dtype=torch.float)
    
    def test_initialization(self, config):
        """Test quantum embeddings initialization."""
        embeddings = QuantumEmbeddings(config=config)
        
        assert embeddings.vocab_size == 1000
        assert embeddings.embedding_dim == 128
        assert embeddings.num_states == 4  # Default
        assert embeddings.superposition_strength == 0.5  # Default
        assert hasattr(embeddings, 'state_embeddings')
        assert hasattr(embeddings, 'superposition_matrix')
        assert hasattr(embeddings, 'position_embeddings')
        assert hasattr(embeddings, 'token_type_embeddings')
        assert hasattr(embeddings, 'LayerNorm')
        assert hasattr(embeddings, 'dropout')
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        embeddings = QuantumEmbeddings(
            vocab_size=500,
            embedding_dim=256,
            num_states=8,
            superposition_strength=0.7
        )
        
        assert embeddings.vocab_size == 500
        assert embeddings.embedding_dim == 256
        assert embeddings.num_states == 8
        assert embeddings.superposition_strength == 0.7
    
    def test_forward_superposition(self, embeddings, input_ids):
        """Test forward pass with superposition."""
        embeddings_output, uncertainty = embeddings(input_ids, collapse=False)
        
        assert embeddings_output.shape == (2, 10, 128)
        assert uncertainty.shape == (2, 10)
        assert torch.all(uncertainty >= 0)  # Uncertainty should be non-negative
    
    def test_forward_collapse(self, embeddings, input_ids):
        """Test forward pass with collapse."""
        context = torch.randn(2, 10, 128)
        embeddings_output, uncertainty = embeddings(
            input_ids, 
            context=context, 
            collapse=True
        )
        
        assert embeddings_output.shape == (2, 10, 128)
        assert uncertainty.shape == (2, 10)
        assert torch.all(uncertainty == 0)  # Collapsed states have no uncertainty
    
    def test_forward_with_context(self, embeddings, input_ids):
        """Test forward pass with context."""
        context = torch.randn(2, 10, 128)
        embeddings_output, uncertainty = embeddings(
            input_ids, 
            context=context
        )
        
        assert embeddings_output.shape == (2, 10, 128)
        assert uncertainty.shape == (2, 10)
    
    def test_create_superposition(self, embeddings, input_ids):
        """Test superposition creation."""
        state_embeds = embeddings.state_embeddings[input_ids]
        superposition = embeddings._create_superposition(state_embeds)
        
        assert superposition.shape == (2, 10, 128)
        assert not torch.isnan(superposition).any()
    
    def test_collapse_superposition(self, embeddings, input_ids):
        """Test superposition collapse."""
        state_embeds = embeddings.state_embeddings[input_ids]
        context = torch.randn(2, 10, 128)
        collapsed = embeddings._collapse_superposition(state_embeds, context)
        
        assert collapsed.shape == (2, 10, 128)
        assert not torch.isnan(collapsed).any()
    
    def test_get_uncertainty(self, embeddings, input_ids):
        """Test uncertainty computation."""
        state_embeds = embeddings.state_embeddings[input_ids]
        uncertainty = embeddings.get_uncertainty_from_states(state_embeds)
        
        assert uncertainty.shape == (2, 10)
        assert torch.all(uncertainty >= 0)
    
    def test_gradients(self, embeddings, input_ids):
        """Test that gradients can be computed."""
        embeddings_output, uncertainty = embeddings(input_ids)
        loss = embeddings_output.sum() + uncertainty.sum()
        loss.backward()
        
        # Check that gradients exist
        assert embeddings.state_embeddings.grad is not None
        assert embeddings.superposition_matrix.grad is not None
    
    def test_different_vocab_sizes(self):
        """Test with different vocabulary sizes."""
        config = BertConfig(vocab_size=2000, hidden_size=128)
        embeddings = QuantumEmbeddings(config=config)
        
        assert embeddings.vocab_size == 2000
        assert embeddings.state_embeddings.shape[0] == 2000
    
    def test_different_embedding_dims(self):
        """Test with different embedding dimensions."""
        config = BertConfig(vocab_size=1000, hidden_size=256)
        embeddings = QuantumEmbeddings(config=config)
        
        assert embeddings.embedding_dim == 256
        assert embeddings.state_embeddings.shape[2] == 256
    
    def test_different_num_states(self):
        """Test with different numbers of quantum states."""
        config = BertConfig(vocab_size=1000, hidden_size=128)
        embeddings = QuantumEmbeddings(config=config, num_states=6)
        
        assert embeddings.num_states == 6
        assert embeddings.state_embeddings.shape[1] == 6


class TestContextCollapseLayer:
    """Test context collapse layer functionality."""
    
    @pytest.fixture
    def collapse_layer(self):
        """Create context collapse layer."""
        return ContextCollapseLayer(
            embedding_dim=128,
            context_window=5,
            collapse_strategy='attention'
        )
    
    @pytest.fixture
    def embeddings(self):
        """Create embeddings tensor."""
        return torch.randn(2, 10, 128)
    
    def test_initialization(self, collapse_layer):
        """Test initialization."""
        assert collapse_layer.embedding_dim == 128
        assert collapse_layer.context_window == 5
        assert collapse_layer.collapse_strategy == 'attention'
        assert hasattr(collapse_layer, 'context_attention')
    
    def test_attention_collapse(self, collapse_layer, embeddings):
        """Test attention-based collapse."""
        output = collapse_layer(embeddings)
        
        assert output.shape == (2, 10, 128)
        assert not torch.isnan(output).any()
    
    def test_convolution_collapse(self):
        """Test convolution-based collapse."""
        collapse_layer = ContextCollapseLayer(
            embedding_dim=128,
            context_window=5,
            collapse_strategy='conv'
        )
        
        embeddings = torch.randn(2, 10, 128)
        output = collapse_layer(embeddings)
        
        assert output.shape == (2, 10, 128)
        assert not torch.isnan(output).any()
    
    def test_rnn_collapse(self):
        """Test RNN-based collapse."""
        collapse_layer = ContextCollapseLayer(
            embedding_dim=128,
            context_window=5,
            collapse_strategy='rnn'
        )
        
        embeddings = torch.randn(2, 10, 128)
        output = collapse_layer(embeddings)
        
        assert output.shape == (2, 10, 128)
        assert not torch.isnan(output).any()
    
    def test_invalid_collapse_method(self):
        """Test invalid collapse method raises error."""
        with pytest.raises(ValueError):
            ContextCollapseLayer(
                embedding_dim=128,
                context_window=5,
                collapse_strategy='invalid'
            )
    
    def test_gradients(self, collapse_layer, embeddings):
        """Test that gradients can be computed."""
        output = collapse_layer(embeddings)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert collapse_layer.context_attention.in_proj_weight.grad is not None


class TestAdaptiveCollapseLayer:
    """Test adaptive collapse layer functionality."""
    
    @pytest.fixture
    def adaptive_layer(self):
        """Create adaptive collapse layer."""
        return AdaptiveCollapseLayer(
            embedding_dim=128,
            num_strategies=3,
            temperature=1.0
        )
    
    @pytest.fixture
    def embeddings(self):
        """Create embeddings tensor."""
        return torch.randn(2, 10, 128)
    
    def test_initialization(self, adaptive_layer):
        """Test initialization."""
        assert adaptive_layer.embedding_dim == 128
        assert adaptive_layer.num_strategies == 3
        assert adaptive_layer.temperature == 1.0
        assert len(adaptive_layer.strategy_layers) == 3
    
    def test_forward(self, adaptive_layer, embeddings):
        """Test forward pass."""
        output, strategy_weights = adaptive_layer(embeddings)
        
        assert output.shape == (2, 10, 128)
        assert strategy_weights.shape == (2, 10, 3)
        assert not torch.isnan(output).any()
        assert not torch.isnan(strategy_weights).any()
    
    def test_method_selection(self, adaptive_layer, embeddings):
        """Test strategy selection mechanism."""
        output, strategy_weights = adaptive_layer(embeddings)
        
        # Strategy weights should sum to 1 across strategies
        assert torch.allclose(strategy_weights.sum(dim=-1), torch.ones(2, 10), atol=1e-6)
    
    def test_gradients(self, adaptive_layer, embeddings):
        """Test that gradients can be computed."""
        output, strategy_weights = adaptive_layer(embeddings)
        loss = output.sum() + strategy_weights.sum()
        loss.backward()
        
        # Check that gradients exist
        assert adaptive_layer.strategy_layers[0].context_attention.in_proj_weight.grad is not None


class TestQuantumMeasurement:
    """Test quantum measurement functionality."""
    
    @pytest.fixture
    def measurement(self):
        """Create quantum measurement instance."""
        return QuantumMeasurement(
            embedding_dim=128,
            measurement_basis='computational'
        )
    
    @pytest.fixture
    def state(self):
        """Create quantum state tensor."""
        return torch.randn(2, 10, 128)
    
    def test_initialization(self, measurement):
        """Test initialization."""
        assert measurement.embedding_dim == 128
        assert measurement.measurement_basis == 'computational'
        assert hasattr(measurement, 'basis_matrix')
    
    def test_forward(self, measurement, state):
        """Test forward pass."""
        measured, results = measurement(state)
        
        assert measured.shape == (2, 10, 128)
        assert results.shape == (2, 10, 129)  # embed_dim + 1 for uncertainty
        assert not torch.isnan(measured).any()
        assert not torch.isnan(results).any()
    
    def test_measurement_bases(self, measurement):
        """Test measurement basis setup."""
        assert measurement.basis_matrix.shape == (128, 128)
        # Basis should be normalized
        assert torch.allclose(
            torch.norm(measurement.basis_matrix, dim=0),
            torch.ones(128),
            atol=1e-6
        )
    
    def test_measurement_results(self, measurement, state):
        """Test measurement results computation."""
        measured, results = measurement(state)
        
        # Results should contain measurement probabilities and uncertainty
        assert results.shape == (2, 10, 129)
        assert not torch.isnan(results).any()
    
    def test_gradients(self, measurement, state):
        """Test that gradients can be computed."""
        measured, results = measurement(state)
        loss = measured.sum() + results.sum()
        loss.backward()
        
        # Check that gradients exist
        assert measurement.basis_matrix.grad is not None


class TestAdaptiveMeasurement:
    """Test adaptive measurement functionality."""
    
    @pytest.fixture
    def adaptive_measurement(self):
        """Create adaptive measurement instance."""
        return AdaptiveMeasurement(
            embedding_dim=128,
            num_bases=4,
            temperature=1.0
        )
    
    @pytest.fixture
    def state(self):
        """Create quantum state tensor."""
        return torch.randn(2, 10, 128)
    
    def test_initialization(self, adaptive_measurement):
        """Test initialization."""
        assert adaptive_measurement.embedding_dim == 128
        assert adaptive_measurement.num_bases == 4
        assert adaptive_measurement.temperature == 1.0
        assert len(adaptive_measurement.measurement_bases) == 4
    
    def test_forward(self, adaptive_measurement, state):
        """Test forward pass."""
        measured, results, basis_weights = adaptive_measurement(state)
        
        assert measured.shape == (2, 10, 128)
        assert results.shape == (2, 10, 129)  # embed_dim + 1 for uncertainty
        assert basis_weights.shape == (2, 10, 4)
        assert not torch.isnan(measured).any()
        assert not torch.isnan(results).any()
        assert not torch.isnan(basis_weights).any()
    
    def test_basis_selection(self, adaptive_measurement, state):
        """Test basis selection mechanism."""
        measured, results, basis_weights = adaptive_measurement(state)
        
        # Basis weights should sum to 1 across bases
        assert torch.allclose(basis_weights.sum(dim=-1), torch.ones(2, 10), atol=1e-6)
    
    def test_context_influence(self, adaptive_measurement, state):
        """Test context influence on basis selection."""
        measured1, results1, weights1 = adaptive_measurement(state)
        measured2, results2, weights2 = adaptive_measurement(state)
        
        # Results should be deterministic for same input
        assert torch.allclose(measured1, measured2, atol=1e-6)
        assert torch.allclose(weights1, weights2, atol=1e-6)
    
    def test_gradients(self, adaptive_measurement, state):
        """Test that gradients can be computed."""
        measured, results, basis_weights = adaptive_measurement(state)
        loss = measured.sum() + results.sum() + basis_weights.sum()
        loss.backward()
        
        # Check that gradients exist
        assert adaptive_measurement.measurement_bases[0].basis_matrix.grad is not None


class TestEntanglementCorrelation:
    """Test entanglement correlation functionality."""
    
    @pytest.fixture
    def entanglement(self):
        """Create entanglement correlation instance."""
        return EntanglementCorrelation(
            embedding_dim=128,
            num_entangled_pairs=5,
            entanglement_strength=0.8,
            correlation_type='linear'
        )
    
    @pytest.fixture
    def embeddings(self):
        """Create embeddings tensor."""
        return torch.randn(2, 10, 128)
    
    def test_initialization(self, entanglement):
        """Test initialization."""
        assert entanglement.embedding_dim == 128
        assert entanglement.num_entangled_pairs == 5
        assert entanglement.entanglement_strength == 0.8
        assert entanglement.correlation_type == 'linear'
        assert hasattr(entanglement, 'entanglement_matrix')
        assert hasattr(entanglement, 'correlation_weights')
    
    def test_forward(self, entanglement, embeddings):
        """Test forward pass."""
        entangled = entanglement(embeddings)
        
        assert entangled.shape == (2, 10, 128)
        assert not torch.isnan(entangled).any()
    
    def test_entanglement_matrix(self, entanglement):
        """Test entanglement matrix properties."""
        matrix = entanglement.entanglement_matrix
        assert matrix.shape == (5, 128, 128)
        assert not torch.isnan(matrix).any()
    
    def test_correlation_weights(self, entanglement):
        """Test correlation weights."""
        weights = entanglement.correlation_weights
        assert weights.shape == (5,)
        assert torch.all(weights >= 0)  # Weights should be non-negative
    
    def test_gradients(self, entanglement, embeddings):
        """Test that gradients can be computed."""
        entangled = entanglement(embeddings)
        loss = entangled.sum()
        loss.backward()
        
        # Check that gradients exist
        assert entanglement.entanglement_matrix.grad is not None


class TestCoreIntegration:
    """Test integration between core components."""
    
    @pytest.fixture
    def embeddings(self):
        """Create quantum embeddings."""
        config = BertConfig(vocab_size=1000, hidden_size=128)
        return QuantumEmbeddings(config=config)
    
    @pytest.fixture
    def collapse_layer(self):
        """Create context collapse layer."""
        return ContextCollapseLayer(
            embedding_dim=128,
            context_window=3,
            collapse_strategy='attention'
        )
    
    @pytest.fixture
    def entanglement(self):
        """Create entanglement correlation."""
        return EntanglementCorrelation(
            embedding_dim=128,
            num_entangled_pairs=3,
            entanglement_strength=0.5,
            correlation_type='linear'
        )
    
    @pytest.fixture
    def measurement(self):
        """Create quantum measurement."""
        return QuantumMeasurement(
            embedding_dim=128,
            measurement_basis='computational'
        )
    
    def test_comprehensive_core_workflow(self, embeddings, collapse_layer, entanglement, measurement):
        """Test complete workflow through all core components."""
        input_ids = torch.randint(0, 1000, (2, 8))
        
        # 1. Generate quantum embeddings
        quantum_embeds, uncertainty = embeddings(input_ids)
        assert quantum_embeds.shape == (2, 8, 128)
        assert uncertainty.shape == (2, 8)
        
        # 2. Apply context collapse
        collapsed_embeds = collapse_layer(quantum_embeds)
        assert collapsed_embeds.shape == (2, 8, 128)
        
        # 3. Apply entanglement
        entangled_embeds = entanglement(collapsed_embeds)
        assert entangled_embeds.shape == (2, 8, 128)
        
        # 4. Measure quantum state
        measured_embeds, measurement_results = measurement(entangled_embeds)
        assert measured_embeds.shape == (2, 8, 128)
        assert measurement_results.shape == (2, 8, 129)
        
        # 5. Verify end-to-end gradients
        final_output = measured_embeds.sum() + measurement_results.sum()
        final_output.backward()
        
        # Check gradients exist
        assert embeddings.state_embeddings.grad is not None
        assert collapse_layer.context_attention.in_proj_weight.grad is not None
        assert entanglement.entanglement_matrix.grad is not None
        assert measurement.basis_matrix.grad is not None
