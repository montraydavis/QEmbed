# QEmbed: Quantum-Enhanced Embeddings for Natural Language Processing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://qembed.readthedocs.io/)

QEmbed is a cutting-edge Python library that brings quantum-inspired concepts to natural language processing. By incorporating quantum mechanics principles like superposition, entanglement, and measurement into neural network architectures, QEmbed enables models to better handle polysemy, contextual ambiguity, and uncertainty quantification.

## 🌟 Key Features

- **Quantum Superposition Embeddings**: Each token can exist in multiple semantic states simultaneously until measured in context
- **Context-Driven Collapse**: Intelligent collapse of superposition states based on surrounding context
- **Quantum Entanglement**: Modeling correlations between different positions in sequences
- **Uncertainty Quantification**: Built-in uncertainty estimates for model predictions
- **Quantum-Enhanced Models**: BERT, Transformer, and hybrid architectures with quantum components
- **Polysemy Handling**: Specialized datasets and training for ambiguous word resolution

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install qembed

# Install with development dependencies
pip install qembed[dev]

# Install with examples
pip install qembed[examples]
```

### Basic Usage

```python
import torch
from qembed.core.quantum_embeddings import QuantumEmbeddings
from qembed.utils.visualization import QuantumVisualization

# Create quantum embeddings
quantum_embeddings = QuantumEmbeddings(
    vocab_size=1000,
    embedding_dim=128,
    num_states=4
)

# Input tokens
input_ids = torch.randint(0, 1000, (2, 10))

# Get embeddings in superposition state
superposition_embeddings = quantum_embeddings(input_ids, collapse=False)

# Get uncertainty estimates
uncertainty = quantum_embeddings.get_uncertainty(input_ids)

# Visualize quantum states
viz = QuantumVisualization()
fig = viz.plot_superposition_states(superposition_embeddings)
```

### Quantum-Enhanced BERT

```python
from qembed.models.quantum_bert import QuantumBERT

# Create quantum BERT model
model = QuantumBERT(
    vocab_size=30000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_quantum_states=4
)

# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    context=context_tensor,
    collapse=False  # Keep in superposition
)

# Get uncertainty
uncertainty = model.get_uncertainty(input_ids)
```

## 🔬 Core Concepts

### Quantum Superposition

In QEmbed, each token can exist in multiple semantic states simultaneously, similar to quantum superposition. This allows the model to maintain ambiguity until sufficient context is provided:

```python
# Each token has 4 possible quantum states
states = quantum_embeddings.state_embeddings[input_ids]  # [batch, seq, 4, dim]

# Superposition combines these states
superposition = quantum_embeddings(input_ids, collapse=False)
```

### Context-Driven Collapse

The model learns when and how to collapse superposition states based on context:

```python
# Collapse based on context
collapsed = quantum_embeddings(
    input_ids, 
    context=context_tensor, 
    collapse=True
)
```

### Quantum Entanglement

Different positions in sequences can become entangled, modeling complex correlations:

```python
from qembed.core.entanglement import EntanglementCorrelation

entanglement = EntanglementCorrelation(embedding_dim=128)
entangled_embeddings = entanglement(embeddings)
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md)
- [API Reference](docs/api_reference.md)
- [Examples](examples/)

## 🧪 Examples

### Polysemy Resolution

```python
from qembed.datasets.polysemy_datasets import PolysemyDataset

# Create dataset for ambiguous words
dataset = PolysemyDataset(
    data=sample_data,
    tokenizer=tokenizer,
    max_length=128
)

# Train model to handle multiple meanings
# The word "bank" can mean financial institution or river side
```

### Uncertainty Quantification

```python
from qembed.training.quantum_trainer import QuantumTrainer

# Train with uncertainty awareness
trainer = QuantumTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    quantum_training_config={
        'uncertainty_regularization': 0.1,
        'superposition_schedule': 'linear'
    }
)

# Model provides uncertainty estimates
uncertainty = model.get_uncertainty(input_ids)
```

## 🏗️ Architecture

```markdown
qembed/
├── core/                    # Core quantum components
│   ├── quantum_embeddings.py   # Superposition embeddings
│   ├── collapse_layers.py      # Context-driven collapse
│   ├── entanglement.py         # Quantum correlations
│   └── measurement.py          # Measurement operators
├── models/                  # Model architectures
│   ├── quantum_bert.py         # Quantum-enhanced BERT
│   ├── quantum_transformer.py  # Quantum transformer
│   └── hybrid_models.py        # Classical-quantum hybrid
├── training/                # Training utilities
│   ├── quantum_trainer.py      # Quantum-aware training
│   ├── losses.py               # Quantum loss functions
│   └── optimizers.py           # Quantum optimizers
├── utils/                    # Utilities
│   ├── quantum_utils.py        # Quantum math utilities
│   ├── visualization.py        # Quantum state visualization
│   └── metrics.py              # Quantum-specific metrics
└── datasets/                 # Specialized datasets
    ├── polysemy_datasets.py    # Polysemy testing
    └── quantum_data_loaders.py # Uncertainty-aware loading
```

## 🔬 Research Applications

QEmbed is particularly useful for:

- **Word Sense Disambiguation**: Resolving ambiguous words in context
- **Polysemy Modeling**: Capturing multiple meanings of words
- **Uncertainty Quantification**: Providing confidence estimates
- **Context-Aware NLP**: Better understanding of contextual dependencies
- **Quantum Machine Learning**: Exploring quantum-inspired algorithms

## 📊 Performance

QEmbed models show improved performance on:

- **Polysemy Tasks**: Better handling of ambiguous words
- **Context Understanding**: Improved contextual awareness
- **Uncertainty Calibration**: More reliable confidence estimates
- **Robustness**: Better generalization to unseen contexts

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/qembed/qembed.git
cd qembed

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black qembed/
isort qembed/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by quantum computing principles
- Built on PyTorch ecosystem
- Research contributions from the NLP and quantum computing communities

---

**QEmbed**: Where quantum meets language, and ambiguity becomes clarity. 🌊✨
