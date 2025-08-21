# QEmbed Core Modules

## Overview

The core modules provide the fundamental quantum-inspired components for creating and manipulating embeddings with quantum properties. These modules implement the basic building blocks that enable quantum superposition, entanglement, and measurement operations.

## Module Structure

### 1. [Quantum Embeddings](./quantum_embeddings.md)

**File:** `qembed/core/quantum_embeddings.py`

The main embedding class that maintains superposition states, allowing tokens to exist in multiple semantic states simultaneously until measured in a specific context.

**Key Features:**

- Superposition of multiple semantic states
- Context-aware state collapse
- Uncertainty quantification
- BERT-compatible interface

**Main Class:** `QuantumEmbeddings`

### 2. [Context Collapse Layers](./collapse_layers.md)

**File:** `qembed/core/collapse_layers.py`

Implements context-driven collapse mechanisms for quantum embeddings, learning to collapse quantum superposition states based on surrounding context.

**Key Features:**

- Multiple collapse strategies (attention, convolution, RNN)
- Learnable context modeling
- Position-aware collapse
- Configurable context windows

**Main Class:** `ContextCollapseLayer`

### 3. [Entanglement Correlation](./entanglement.md)

**File:** `qembed/core/entanglement.py`

Implements quantum entanglement-inspired mechanisms for modeling correlations between different tokens and positions in sequences.

**Key Features:**

- Multiple entanglement types (Bell states, GHZ states, custom)
- Position-dependent entanglement
- Learnable correlation patterns
- Configurable entanglement strength

**Main Class:** `EntanglementCorrelation`

### 4. [Quantum Measurement](./measurement.md)

**File:** `qembed/core/measurement.py`

Implements various quantum measurement operators for collapsing superposition states and extracting classical information from quantum embeddings.

**Key Features:**

- Multiple measurement bases (computational, Bell, custom)
- Configurable measurement noise
- Collapse probability control
- Extensible measurement strategies

**Main Class:** `QuantumMeasurement`

## Quick Start

### Basic Usage

```python
from qembed.core import QuantumEmbeddings, ContextCollapseLayer, EntanglementCorrelation, QuantumMeasurement

# Initialize quantum embeddings
quantum_emb = QuantumEmbeddings(vocab_size=10000, embedding_dim=256)

# Add context collapse
collapse_layer = ContextCollapseLayer(embedding_dim=256)

# Add entanglement
entanglement = EntanglementCorrelation(embedding_dim=256)

# Add measurement
measurement = QuantumMeasurement(embedding_dim=256)

# Create input
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# Forward pass through all components
embeddings, uncertainty = quantum_emb(input_ids)
collapsed = collapse_layer(embeddings)
entangled = entanglement(collapsed)
measured, results = measurement(entangled)
```

### Component Integration

The core modules are designed to work together seamlessly:

1. **QuantumEmbeddings** creates superposition states
2. **ContextCollapseLayer** learns context-aware collapse
3. **EntanglementCorrelation** models token correlations
4. **QuantumMeasurement** extracts classical information

## Architecture Principles

### 1. Modularity

Each core component is self-contained and can be used independently or in combination.

### 2. Extensibility

The base classes are designed to be easily extended for custom implementations.

### 3. BERT Compatibility

Core components maintain compatibility with BERT-style architectures for easy integration.

### 4. Quantum-Inspired

All components implement quantum computing concepts while maintaining classical computational efficiency.

## Performance Characteristics

- **Memory Usage**: Linear scaling with sequence length and embedding dimension
- **Computational Complexity**: Efficient implementations using PyTorch optimizations
- **Gradient Flow**: All operations maintain end-to-end differentiability
- **Device Support**: Full CPU and GPU support with automatic device placement

## Training Considerations

- **Superposition Strength**: Controls the mixing between quantum states
- **Entanglement Strength**: Determines the intensity of token correlations
- **Collapse Strategy**: Choose based on task requirements and computational constraints
- **Measurement Basis**: Select appropriate basis for information extraction

## Advanced Usage

### Custom Collapse Strategies

```python
class CustomCollapseLayer(ContextCollapseLayer):
    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__(embedding_dim, **kwargs)
        # Add custom components
        
    def _generate_context(self, embeddings, attention_mask=None):
        # Implement custom context generation
        pass
```

### Custom Entanglement Patterns

```python
class CustomEntanglement(EntanglementCorrelation):
    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__(embedding_dim, **kwargs)
        # Add custom entanglement mechanisms
        
    def _create_entangled_pairs(self, embeddings):
        # Implement custom pair creation
        pass
```

### Custom Measurement Operators

```python
class CustomMeasurement(QuantumMeasurement):
    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__(embedding_dim, **kwargs)
        # Add custom measurement logic
        
    def forward(self, embeddings, **kwargs):
        # Implement custom measurement
        pass
```

## Testing and Validation

Each core module includes comprehensive tests:

```bash
# Run core module tests
python -m pytest tests/test_core/ -v

# Run specific module tests
python -m pytest tests/test_core/test_quantum_embeddings.py -v
```

## Contributing

When contributing to core modules:

1. Maintain the modular architecture
2. Ensure BERT compatibility
3. Add comprehensive tests
4. Update documentation
5. Follow quantum computing principles

## References

- **Quantum Computing**: Nielsen & Chuang, "Quantum Computation and Quantum Information"
- **BERT Architecture**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers"
- **Quantum Machine Learning**: Schuld & Petruccione, "Supervised Learning with Quantum Computers"
