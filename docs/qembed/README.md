# QEmbed Documentation

## Overview

QEmbed is a quantum-inspired embedding framework that brings quantum computing concepts to natural language processing. It implements quantum superposition, entanglement, and measurement operations to enhance traditional embedding models with quantum properties.

## Documentation Structure

### 🚀 [Getting Started](./overview.md)

- **Framework Overview**: Complete understanding of QEmbed architecture
- **Quick Start Guide**: Get up and running in minutes
- **Installation Guide**: Setup and configuration
- **Basic Usage Examples**: Simple examples to get started

### 🔧 [Core Modules](./core/)

**File:** `qembed/core/`

Fundamental quantum-inspired components for creating and manipulating embeddings with quantum properties.

- **[Overview](./core/README.md)** - Complete guide to core modules
- **[Quantum Embeddings](./core/quantum_embeddings.md)** - Main embedding class with superposition states
- **[Context Collapse Layers](./core/collapse_layers.md)** - Context-aware state collapse mechanisms
- **[Entanglement](./core/entanglement.md)** - Quantum entanglement between tokens
- **[Measurement](./core/measurement.md)** - Quantum measurement operators

### 🏗️ [Models](./models/)

**File:** `qembed/models/`

Complete model architectures that integrate quantum-inspired components with classical neural networks.

- **[Overview](./models/README.md)** - Complete guide to model architectures
- **[Quantum BERT](./models/quantum_bert.md)** - BERT-compatible quantum models
- **[Quantum Transformer](./models/quantum_transformer.md)** - Transformer with quantum components
- **[Hybrid Models](./models/hybrid_models.md)** - Classical-quantum hybrid architectures

### 🎯 [Training](./training/)

**File:** `qembed/training/`

Specialized training utilities designed for quantum-enhanced models.

- **[Overview](./training/README.md)** - Complete guide to training
- **[Quantum Trainer](./training/quantum_trainer.md)** - Quantum-aware training loops
- **[Loss Functions](./training/losses.md)** - Quantum-inspired loss functions
- **[Optimizers](./training/optimizers.md)** - Specialized optimizers

### 📊 [Datasets](./datasets/)

**File:** `qembed/datasets/`

Specialized dataset classes and data loaders for testing quantum-inspired models.

- **[Overview](./datasets/README.md)** - Complete guide to datasets
- **[Polysemy Datasets](./datasets/polysemy_datasets.md)** - Word sense disambiguation data
- **[Quantum Data Loaders](./datasets/quantum_data_loaders.md)** - Specialized data loaders

### 🛠️ [Utilities](./utils/)

**File:** `qembed/utils/`

Helper functions, visualization tools, and metrics specifically designed for quantum-inspired models.

- **[Overview](./utils/README.md)** - Complete guide to utilities
- **[Metrics](./utils/metrics.md)** - Evaluation metrics and quantum property measurements
- **[Quantum Utils](./utils/quantum_utils.md)** - Quantum computing utilities and helper functions
- **[Visualization](./utils/visualization.md)** - Plotting and visualization tools

### 📈 [Evaluation](./evaluation/)

**File:** `qembed/evaluation/`

Comprehensive evaluation frameworks for quantum-enhanced models.

- **[Overview](./evaluation/README.md)** - Complete guide to evaluation
- **[Module Overview](./evaluation/overview.md)** - Comprehensive component overview
- **Task-Specific Evaluators** - Classification, MLM, and embedding evaluation ✅
- **Quantum Analysis Tools** - Coherence, uncertainty, and superposition analysis ✅
- **Pipeline & Reporting** - Multi-model evaluation and comprehensive reporting ✅

## Quick Navigation

### For Beginners

1. **[Framework Overview](./overview.md)** - Start here to understand the big picture
2. **[Installation Guide](../installation.md)** - Set up QEmbed on your system
3. **[Quick Start Guide](../quickstart.md)** - Run your first example
4. **[Core Modules](./core/README.md)** - Learn about fundamental components

### For Developers

1. **[Models](./models/README.md)** - Choose and configure model architectures
2. **[Training](./training/README.md)** - Set up training pipelines
3. **[Datasets](./datasets/README.md)** - Work with specialized datasets
4. **[Utilities](./utils/README.md)** - Use helper functions and tools

### For Researchers

1. **[Evaluation](./evaluation/README.md)** - Comprehensive model evaluation
2. **[Module Overview](./evaluation/overview.md)** - Complete evaluation framework overview
3. **Quantum Analysis Tools** - Deep analysis of quantum properties
4. **[API Reference](../api_reference.md)** - Complete API documentation

## Key Features

- **🔮 Quantum Superposition**: Maintain multiple semantic states per token
- **🔗 Entanglement**: Model correlations between tokens using quantum entanglement
- **📏 Adaptive Measurement**: Learn optimal measurement strategies
- **❓ Uncertainty Quantification**: Built-in uncertainty estimation
- **🤖 BERT Compatibility**: Drop-in replacement for BERT embeddings
- **🎓 Comprehensive Training**: Specialized training infrastructure
- **🏆 Rich Model Zoo**: Multiple architecture options
- **📊 Full Evaluation**: Complete benchmarking and analysis tools

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/qembed.git
cd qembed

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from qembed.models import QuantumBertModel
from transformers import BertConfig

# Create quantum BERT model
config = BertConfig(vocab_size=30522, hidden_size=768)
model = QuantumBertModel(config=config)

# Use like standard BERT
input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])
outputs = model(input_ids)
```

## Examples and Tutorials

- **[Basic Usage](../examples/basic_usage.py)** - Simple examples to get started
- **[Quantum BERT Example](../examples/quantum_bert_example.py)** - Complete quantum BERT workflow
- **[Benchmarks](../examples/benchmarks/)** - Performance benchmarks and comparisons

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation Issues**: Open an issue in this repository
- **Code Issues**: Open an issue in the main QEmbed repository
- **Questions**: Use GitHub Discussions or open an issue
- **Contributions**: Submit pull requests following our guidelines
