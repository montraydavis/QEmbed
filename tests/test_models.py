"""
Tests for model architectures.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from qembed.models.quantum_bert import (
    QuantumBERTEmbeddings, 
    QuantumBERTSelfAttention, 
    QuantumBERTLayer, 
    QuantumBERT
)
from qembed.models.quantum_transformer import (
    QuantumTransformerEmbeddings,
    QuantumMultiHeadAttention,
    QuantumTransformerLayer,
    QuantumTransformer
)
from qembed.models.hybrid_models import (
    HybridEmbeddingLayer,
    HybridAttention,
    HybridModel,
    HybridTransformerLayer
)


class TestQuantumBERTEmbeddings:
    """Test cases for QuantumBERTEmbeddings class."""
    
    @pytest.fixture
    def embeddings(self):
        """Create a QuantumBERTEmbeddings instance for testing."""
        return QuantumBERTEmbeddings(
            vocab_size=30000,
            hidden_size=768,
            max_position_embeddings=512,
            type_vocab_size=2,
            num_quantum_states=4,
            dropout=0.1
        )
    
    @pytest.fixture
    def input_ids(self):
        """Create sample input IDs."""
        return torch.randint(0, 30000, (2, 10))
    
    @pytest.fixture
    def token_type_ids(self):
        """Create sample token type IDs."""
        return torch.zeros(2, 10, dtype=torch.long)
    
    def test_initialization(self, embeddings):
        """Test proper initialization."""
        assert embeddings.vocab_size == 30000
        assert embeddings.hidden_size == 768
        assert embeddings.max_position_embeddings == 512
        assert embeddings.type_vocab_size == 2
        assert embeddings.num_quantum_states == 4
        assert embeddings.dropout == 0.1
        
        # Check that all embedding layers exist
        assert hasattr(embeddings, 'quantum_embeddings')
        assert hasattr(embeddings, 'position_embeddings')
        assert hasattr(embeddings, 'token_type_embeddings')
        assert hasattr(embeddings, 'layer_norm')
        assert hasattr(embeddings, 'dropout')
    
    def test_forward(self, embeddings, input_ids, token_type_ids):
        """Test forward pass."""
        output = embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            collapse=False
        )
        
        assert output.shape == (2, 10, 768)
    
    def test_forward_with_position_ids(self, embeddings, input_ids, token_type_ids):
        """Test forward pass with custom position IDs."""
        position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        output = embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            collapse=False
        )
        
        assert output.shape == (2, 10, 768)
    
    def test_forward_collapse(self, embeddings, input_ids, token_type_ids):
        """Test forward pass with collapse."""
        output = embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            collapse=True
        )
        
        assert output.shape == (2, 10, 768)


class TestQuantumBERTSelfAttention:
    """Test cases for QuantumBERTSelfAttention class."""
    
    @pytest.fixture
    def attention(self):
        """Create a QuantumBERTSelfAttention instance for testing."""
        return QuantumBERTSelfAttention(
            hidden_size=768,
            num_attention_heads=12,
            num_quantum_states=4,
            dropout=0.1
        )
    
    @pytest.fixture
    def hidden_states(self):
        """Create sample hidden states."""
        return torch.randn(2, 10, 768)
    
    @pytest.fixture
    def attention_mask(self):
        """Create sample attention mask."""
        return torch.ones(2, 10, dtype=torch.bool)
    
    def test_initialization(self, attention):
        """Test proper initialization."""
        assert attention.hidden_size == 768
        assert attention.num_attention_heads == 12
        assert attention.num_quantum_states == 4
        assert attention.dropout == 0.1
        assert attention.attention_head_size == 768 // 12
        
        # Check that quantum components exist
        assert hasattr(attention, 'quantum_measurement')
        assert hasattr(attention, 'entanglement')
    
    def test_forward(self, attention, hidden_states, attention_mask):
        """Test forward pass."""
        output = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            collapse=False
        )
        
        assert output.shape == (2, 10, 768)
    
    def test_forward_collapse(self, attention, hidden_states, attention_mask):
        """Test forward pass with collapse."""
        output = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            collapse=True
        )
        
        assert output.shape == (2, 10, 768)
    
    def test_forward_with_context(self, attention, hidden_states, attention_mask):
        """Test forward pass with context."""
        context = torch.randn(2, 10, 768)
        output = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            context=context,
            collapse=False
        )
        
        assert output.shape == (2, 10, 768)


class TestQuantumBERTLayer:
    """Test cases for QuantumBERTLayer class."""
    
    @pytest.fixture
    def layer(self):
        """Create a QuantumBERTLayer instance for testing."""
        return QuantumBERTLayer(
            hidden_size=768,
            num_attention_heads=12,
            num_quantum_states=4,
            intermediate_size=3072,
            dropout=0.1
        )
    
    @pytest.fixture
    def hidden_states(self):
        """Create sample hidden states."""
        return torch.randn(2, 10, 768)
    
    @pytest.fixture
    def attention_mask(self):
        """Create sample attention mask."""
        return torch.ones(2, 10, dtype=torch.bool)
    
    def test_initialization(self, layer):
        """Test proper initialization."""
        assert layer.hidden_size == 768
        assert layer.num_attention_heads == 12
        assert layer.num_quantum_states == 4
        assert layer.intermediate_size == 3072
        assert layer.dropout == 0.1
        
        # Check that components exist
        assert hasattr(layer, 'attention')
        assert hasattr(layer, 'intermediate')
        assert hasattr(layer, 'output')
    
    def test_forward(self, layer, hidden_states, attention_mask):
        """Test forward pass."""
        output = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            collapse=False
        )
        
        assert output.shape == (2, 10, 768)
    
    def test_forward_collapse(self, layer, hidden_states, attention_mask):
        """Test forward pass with collapse."""
        output = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            collapse=True
        )
        
        assert output.shape == (2, 10, 768)


class TestQuantumBERT:
    """Test cases for QuantumBERT class."""
    
    @pytest.fixture
    def model(self):
        """Create a QuantumBERT instance for testing."""
        return QuantumBERT(
            vocab_size=30000,
            hidden_size=768,
            num_hidden_layers=2,  # Small for testing
            num_attention_heads=12,
            num_quantum_states=4,
            intermediate_size=3072,
            dropout=0.1
        )
    
    @pytest.fixture
    def input_ids(self):
        """Create sample input IDs."""
        return torch.randint(0, 30000, (2, 10))
    
    @pytest.fixture
    def attention_mask(self):
        """Create sample attention mask."""
        return torch.ones(2, 10, dtype=torch.bool)
    
    def test_initialization(self, model):
        """Test proper initialization."""
        assert model.vocab_size == 30000
        assert model.hidden_size == 768
        assert model.num_hidden_layers == 2
        assert model.num_attention_heads == 12
        assert model.num_quantum_states == 4
        
        # Check that components exist
        assert hasattr(model, 'embeddings')
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'pooler')
    
    def test_forward(self, model, input_ids, attention_mask):
        """Test forward pass."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            collapse=False
        )
        
        assert outputs.last_hidden_state.shape == (2, 10, 768)
        assert outputs.pooler_output.shape == (2, 768)
    
    def test_forward_collapse(self, model, input_ids, attention_mask):
        """Test forward pass with collapse."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            collapse=True
        )
        
        assert outputs.last_hidden_state.shape == (2, 10, 768)
        assert outputs.pooler_output.shape == (2, 768)
    
    def test_get_uncertainty(self, model, input_ids):
        """Test uncertainty calculation."""
        uncertainty = model.get_uncertainty(input_ids)
        
        assert uncertainty.shape == (2, 10)
        assert torch.all(uncertainty >= 0)
        assert torch.all(uncertainty <= 1)


class TestQuantumTransformerEmbeddings:
    """Test cases for QuantumTransformerEmbeddings class."""
    
    @pytest.fixture
    def embeddings(self):
        """Create a QuantumTransformerEmbeddings instance for testing."""
        return QuantumTransformerEmbeddings(
            vocab_size=30000,
            hidden_size=512,
            max_position_embeddings=1024,
            num_quantum_states=4,
            dropout=0.1
        )
    
    @pytest.fixture
    def input_ids(self):
        """Create sample input IDs."""
        return torch.randint(0, 30000, (2, 20))
    
    def test_initialization(self, embeddings):
        """Test proper initialization."""
        assert embeddings.vocab_size == 30000
        assert embeddings.hidden_size == 512
        assert embeddings.max_position_embeddings == 1024
        assert embeddings.num_quantum_states == 4
        assert embeddings.dropout == 0.1
        
        # Check that components exist
        assert hasattr(embeddings, 'quantum_embeddings')
        assert hasattr(embeddings, 'position_embeddings')
        assert hasattr(embeddings, 'layer_norm')
        assert hasattr(embeddings, 'dropout')
    
    def test_forward(self, embeddings, input_ids):
        """Test forward pass."""
        output = embeddings(input_ids, collapse=False)
        
        assert output.shape == (2, 20, 512)
    
    def test_forward_collapse(self, embeddings, input_ids):
        """Test forward pass with collapse."""
        output = embeddings(input_ids, collapse=True)
        
        assert output.shape == (2, 20, 512)


class TestQuantumMultiHeadAttention:
    """Test cases for QuantumMultiHeadAttention class."""
    
    @pytest.fixture
    def attention(self):
        """Create a QuantumMultiHeadAttention instance for testing."""
        return QuantumMultiHeadAttention(
            hidden_size=512,
            num_attention_heads=8,
            num_quantum_states=4,
            dropout=0.1
        )
    
    @pytest.fixture
    def query(self):
        """Create sample query tensor."""
        return torch.randn(2, 20, 512)
    
    @pytest.fixture
    def key(self):
        """Create sample key tensor."""
        return torch.randn(2, 20, 512)
    
    @pytest.fixture
    def value(self):
        """Create sample value tensor."""
        return torch.randn(2, 20, 512)
    
    @pytest.fixture
    def mask(self):
        """Create sample mask tensor."""
        return torch.ones(2, 20, dtype=torch.bool)
    
    def test_initialization(self, attention):
        """Test proper initialization."""
        assert attention.hidden_size == 512
        assert attention.num_attention_heads == 8
        assert attention.num_quantum_states == 4
        assert attention.dropout == 0.1
        assert attention.attention_head_size == 512 // 8
        
        # Check that quantum components exist
        assert hasattr(attention, 'quantum_measurement')
        assert hasattr(attention, 'entanglement')
    
    def test_forward(self, attention, query, key, value, mask):
        """Test forward pass."""
        output = attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            collapse=False
        )
        
        assert output.shape == (2, 20, 512)
    
    def test_forward_collapse(self, attention, query, key, value, mask):
        """Test forward pass with collapse."""
        output = attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            collapse=True
        )
        
        assert output.shape == (2, 20, 512)


class TestQuantumTransformerLayer:
    """Test cases for QuantumTransformerLayer class."""
    
    @pytest.fixture
    def layer(self):
        """Create a QuantumTransformerLayer instance for testing."""
        return QuantumTransformerLayer(
            hidden_size=512,
            num_attention_heads=8,
            num_quantum_states=4,
            intermediate_size=2048,
            dropout=0.1
        )
    
    @pytest.fixture
    def hidden_states(self):
        """Create sample hidden states."""
        return torch.randn(2, 20, 512)
    
    @pytest.fixture
    def mask(self):
        """Create sample mask tensor."""
        return torch.ones(2, 20, dtype=torch.bool)
    
    def test_initialization(self, layer):
        """Test proper initialization."""
        assert layer.hidden_size == 512
        assert layer.num_attention_heads == 8
        assert layer.num_quantum_states == 4
        assert layer.intermediate_size == 2048
        assert layer.dropout == 0.1
        
        # Check that components exist
        assert hasattr(layer, 'attention')
        assert hasattr(layer, 'intermediate')
        assert hasattr(layer, 'output')
    
    def test_forward(self, layer, hidden_states, mask):
        """Test forward pass."""
        output = layer(
            hidden_states=hidden_states,
            mask=mask,
            collapse=False
        )
        
        assert output.shape == (2, 20, 512)
    
    def test_forward_collapse(self, layer, hidden_states, mask):
        """Test forward pass with collapse."""
        output = layer(
            hidden_states=hidden_states,
            mask=mask,
            collapse=True
        )
        
        assert output.shape == (2, 20, 512)


class TestQuantumTransformer:
    """Test cases for QuantumTransformer class."""
    
    @pytest.fixture
    def model(self):
        """Create a QuantumTransformer instance for testing."""
        return QuantumTransformer(
            vocab_size=30000,
            hidden_size=512,
            num_hidden_layers=2,  # Small for testing
            num_attention_heads=8,
            num_quantum_states=4,
            intermediate_size=2048,
            dropout=0.1
        )
    
    @pytest.fixture
    def input_ids(self):
        """Create sample input IDs."""
        return torch.randint(0, 30000, (2, 20))
    
    @pytest.fixture
    def mask(self):
        """Create sample mask tensor."""
        return torch.ones(2, 20, dtype=torch.bool)
    
    def test_initialization(self, model):
        """Test proper initialization."""
        assert model.vocab_size == 30000
        assert model.hidden_size == 512
        assert model.num_hidden_layers == 2
        assert model.num_attention_heads == 8
        assert model.num_quantum_states == 4
        
        # Check that components exist
        assert hasattr(model, 'embeddings')
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'output_layer')
    
    def test_forward(self, model, input_ids, mask):
        """Test forward pass."""
        output = model(
            input_ids=input_ids,
            mask=mask,
            collapse=False
        )
        
        assert output.shape == (2, 20, 30000)
    
    def test_forward_collapse(self, model, input_ids, mask):
        """Test forward pass with collapse."""
        output = model(
            input_ids=input_ids,
            mask=mask,
            collapse=True
        )
        
        assert output.shape == (2, 20, 30000)
    
    def test_generate(self, model, input_ids):
        """Test text generation."""
        generated = model.generate(
            input_ids=input_ids,
            max_length=25,
            num_beams=2
        )
        
        assert generated.shape[0] == 2
        assert generated.shape[1] <= 25


class TestHybridEmbeddingLayer:
    """Test cases for HybridEmbeddingLayer class."""
    
    @pytest.fixture
    def layer(self):
        """Create a HybridEmbeddingLayer instance for testing."""
        return HybridEmbeddingLayer(
            vocab_size=30000,
            embedding_dim=128,
            num_quantum_states=4,
            dropout=0.1
        )
    
    @pytest.fixture
    def input_ids(self):
        """Create sample input IDs."""
        return torch.randint(0, 30000, (2, 10))
    
    def test_initialization(self, layer):
        """Test proper initialization."""
        assert layer.vocab_size == 30000
        assert layer.embedding_dim == 128
        assert layer.num_quantum_states == 4
        assert layer.dropout == 0.1
        
        # Check that both classical and quantum embeddings exist
        assert hasattr(layer, 'classical_embeddings')
        assert hasattr(layer, 'quantum_embeddings')
        assert hasattr(layer, 'ambiguity_detector')
    
    def test_forward(self, layer, input_ids):
        """Test forward pass."""
        output = layer(input_ids)
        
        assert output.shape == (2, 10, 128)
    
    def test_forward_with_ambiguity(self, layer, input_ids):
        """Test forward pass with ambiguity scores."""
        output, ambiguity = layer(input_ids, return_ambiguity=True)
        
        assert output.shape == (2, 10, 128)
        assert ambiguity.shape == (2, 10)
        assert torch.all(ambiguity >= 0)
        assert torch.all(ambiguity <= 1)


class TestHybridAttention:
    """Test cases for HybridAttention class."""
    
    @pytest.fixture
    def attention(self):
        """Create a HybridAttention instance for testing."""
        return HybridAttention(
            hidden_size=512,
            num_attention_heads=8,
            num_quantum_states=4,
            dropout=0.1
        )
    
    @pytest.fixture
    def hidden_states(self):
        """Create sample hidden states."""
        return torch.randn(2, 20, 512)
    
    @pytest.fixture
    def mask(self):
        """Create sample mask tensor."""
        return torch.ones(2, 20, dtype=torch.bool)
    
    def test_initialization(self, attention):
        """Test proper initialization."""
        assert attention.hidden_size == 512
        assert attention.num_attention_heads == 8
        assert attention.num_quantum_states == 4
        assert attention.dropout == 0.1
        
        # Check that both classical and quantum attention exist
        assert hasattr(attention, 'classical_attention')
        assert hasattr(attention, 'quantum_attention')
        assert hasattr(attention, 'attention_selector')
    
    def test_forward(self, attention, hidden_states, mask):
        """Test forward pass."""
        output = attention(
            hidden_states=hidden_states,
            mask=mask
        )
        
        assert output.shape == (2, 20, 512)


class TestHybridModel:
    """Test cases for HybridModel class."""
    
    @pytest.fixture
    def model(self):
        """Create a HybridModel instance for testing."""
        return HybridModel(
            vocab_size=30000,
            hidden_size=512,
            num_hidden_layers=2,  # Small for testing
            num_attention_heads=8,
            num_quantum_states=4,
            intermediate_size=2048,
            dropout=0.1
        )
    
    @pytest.fixture
    def input_ids(self):
        """Create sample input IDs."""
        return torch.randint(0, 30000, (2, 20))
    
    @pytest.fixture
    def mask(self):
        """Create sample mask tensor."""
        return torch.ones(2, 20, dtype=torch.bool)
    
    def test_initialization(self, model):
        """Test proper initialization."""
        assert model.vocab_size == 30000
        assert model.hidden_size == 512
        assert model.num_hidden_layers == 2
        assert model.num_attention_heads == 8
        assert model.num_quantum_states == 4
        
        # Check that components exist
        assert hasattr(model, 'embeddings')
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'output_layer')
    
    def test_forward(self, model, input_ids, mask):
        """Test forward pass."""
        output = model(
            input_ids=input_ids,
            mask=mask
        )
        
        assert output.shape == (2, 20, 30000)
    
    def test_get_uncertainty(self, model, input_ids):
        """Test uncertainty calculation."""
        uncertainty = model.get_uncertainty(input_ids)
        
        assert uncertainty.shape == (2, 20)
        assert torch.all(uncertainty >= 0)
        assert torch.all(uncertainty <= 1)


class TestHybridTransformerLayer:
    """Test cases for HybridTransformerLayer class."""
    
    @pytest.fixture
    def layer(self):
        """Create a HybridTransformerLayer instance for testing."""
        return HybridTransformerLayer(
            hidden_size=512,
            num_attention_heads=8,
            num_quantum_states=4,
            intermediate_size=2048,
            dropout=0.1
        )
    
    @pytest.fixture
    def hidden_states(self):
        """Create sample hidden states."""
        return torch.randn(2, 20, 512)
    
    @pytest.fixture
    def mask(self):
        """Create sample mask tensor."""
        return torch.ones(2, 20, dtype=torch.bool)
    
    def test_initialization(self, layer):
        """Test proper initialization."""
        assert layer.hidden_size == 512
        assert layer.num_attention_heads == 8
        assert layer.num_quantum_states == 4
        assert layer.intermediate_size == 2048
        assert layer.dropout == 0.1
        
        # Check that components exist
        assert hasattr(layer, 'attention')
        assert hasattr(layer, 'intermediate')
        assert hasattr(layer, 'output')
    
    def test_forward(self, layer, hidden_states, mask):
        """Test forward pass."""
        output = layer(
            hidden_states=hidden_states,
            mask=mask
        )
        
        assert output.shape == (2, 20, 512)


if __name__ == "__main__":
    pytest.main([__file__])
