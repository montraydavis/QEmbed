# API Reference

This document provides comprehensive API documentation for all QEmbed components.

## 📚 Table of Contents

- [Core Components](#core-components)
- [Model Architectures](#model-architectures)
- [Training Utilities](#training-utilities)
- [Utility Functions](#utility-functions)
- [Dataset Classes](#dataset-classes)
- [Examples](#examples)

## 🔧 Core Components

### QuantumEmbeddings

The main quantum embedding layer that creates superposition states.

```python
class qembed.core.quantum_embeddings.QuantumEmbeddings(
    vocab_size: int,
    embedding_dim: int,
    num_states: int = 4,
    dropout: float = 0.1,
    collapse_method: str = 'attention'
)
```

**Parameters:**

- `vocab_size` (int): Size of the vocabulary
- `embedding_dim` (int): Dimension of embeddings
- `num_states` (int): Number of quantum states per token (default: 4)
- `dropout` (float): Dropout probability (default: 0.1)
- `collapse_method` (str): Method for collapsing superposition ('attention', 'convolution', 'rnn')

**Methods:**

#### `forward(input_ids, context=None, collapse=False)`

Forward pass through the quantum embeddings.

**Parameters:**

- `input_ids` (torch.Tensor): Input token IDs [batch_size, seq_len]
- `context` (torch.Tensor, optional): Context tensor for collapse [batch_size, seq_len, embedding_dim]
- `collapse` (bool): Whether to collapse superposition (default: False)

**Returns:**

- `torch.Tensor`: Embeddings [batch_size, seq_len, embedding_dim]

#### `get_uncertainty(input_ids)`

Calculate uncertainty for given input tokens.

**Parameters:**

- `input_ids` (torch.Tensor): Input token IDs

**Returns:**

- `torch.Tensor`: Uncertainty scores [batch_size, seq_len]

### ContextCollapseLayer

Context-driven collapse layer for superposition states.

```python
class qembed.core.collapse_layers.ContextCollapseLayer(
    embedding_dim: int,
    num_states: int,
    collapse_method: str = 'attention',
    dropout: float = 0.1
)
```

**Parameters:**

- `embedding_dim` (int): Dimension of embeddings
- `num_states` (int): Number of quantum states
- `collapse_method` (str): Collapse method ('attention', 'convolution', 'rnn')
- `dropout` (float): Dropout probability

**Methods:**

#### `forward(superposition_states, context)`

Collapse superposition states based on context.

**Parameters:**

- `superposition_states` (torch.Tensor): States in superposition [batch_size, seq_len, num_states, embedding_dim]
- `context` (torch.Tensor): Context tensor [batch_size, seq_len, embedding_dim]

**Returns:**

- `torch.Tensor`: Collapsed embeddings [batch_size, seq_len, embedding_dim]

### AdaptiveCollapseLayer

Adaptive collapse layer that learns optimal collapse strategy.

```python
class qembed.core.collapse_layers.AdaptiveCollapseLayer(
    embedding_dim: int,
    num_states: int,
    collapse_methods: List[str],
    dropout: float = 0.1
)
```

**Parameters:**

- `embedding_dim` (int): Dimension of embeddings
- `num_states` (int): Number of quantum states
- `collapse_methods` (List[str]): Available collapse methods
- `dropout` (float): Dropout probability

### EntanglementCorrelation

General entanglement correlation modeling.

```python
class qembed.core.entanglement.EntanglementCorrelation(
    embedding_dim: int,
    correlation_type: str = 'general',
    dropout: float = 0.1
)
```

**Parameters:**

- `embedding_dim` (int): Dimension of embeddings
- `correlation_type` (str): Type of correlation ('general', 'pairwise', 'multi_party')
- `dropout` (float): Dropout probability

### BellStateEntanglement

Pairwise entanglement using Bell states.

```python
class qembed.core.entanglement.BellStateEntanglement(
    embedding_dim: int,
    dropout: float = 0.1
)
```

### GHZStateEntanglement

Multi-party entanglement using GHZ states.

```python
class qembed.core.entanglement.GHZStateEntanglement(
    embedding_dim: int,
    num_parties: int = 3,
    dropout: float = 0.1
)
```

### QuantumMeasurement

Base quantum measurement operator.

```python
class qembed.core.measurement.QuantumMeasurement(
    embedding_dim: int,
    num_bases: int = 3,
    dropout: float = 0.1
)
```

**Parameters:**

- `embedding_dim` (int): Dimension of embeddings
- `num_bases` (int): Number of measurement bases
- `dropout` (float): Dropout probability

### AdaptiveMeasurement

Adaptive measurement with learnable bases.

```python
class qembed.core.measurement.AdaptiveMeasurement(
    embedding_dim: int,
    dropout: float = 0.1
)
```

### WeakMeasurement

Partial collapse measurement operator.

```python
class qembed.core.measurement.WeakMeasurement(
    embedding_dim: int,
    measurement_strength: float = 0.5,
    dropout: float = 0.1
)
```

### POVMMeasurement

Positive Operator-Valued Measure (POVM) measurement.

```python
class qembed.core.measurement.POVMMeasurement(
    embedding_dim: int,
    num_operators: int = 4,
    dropout: float = 0.1
)
```

## 🧠 Model Architectures

### QuantumBERT

Quantum-enhanced BERT model.

```python
class qembed.models.quantum_bert.QuantumBERT(
    vocab_size: int,
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    num_quantum_states: int = 4,
    intermediate_size: int = 3072,
    dropout: float = 0.1,
    max_position_embeddings: int = 512,
    type_vocab_size: int = 2
)
```

**Parameters:**

- `vocab_size` (int): Size of the vocabulary
- `hidden_size` (int): Hidden layer dimension (default: 768)
- `num_hidden_layers` (int): Number of transformer layers (default: 12)
- `num_attention_heads` (int): Number of attention heads (default: 12)
- `num_quantum_states` (int): Number of quantum states (default: 4)
- `intermediate_size` (int): Intermediate layer size (default: 3072)
- `dropout` (float): Dropout probability (default: 0.1)
- `max_position_embeddings` (int): Maximum position embeddings (default: 512)
- `type_vocab_size` (int): Token type vocabulary size (default: 2)

**Methods:**

#### `forward(input_ids, attention_mask=None, token_type_ids=None, position_ids=None, context=None, collapse=False)`

Forward pass through the quantum BERT model.

**Parameters:**

- `input_ids` (torch.Tensor): Input token IDs [batch_size, seq_len]
- `attention_mask` (torch.Tensor, optional): Attention mask [batch_size, seq_len]
- `token_type_ids` (torch.Tensor, optional): Token type IDs [batch_size, seq_len]
- `position_ids` (torch.Tensor, optional): Position IDs [batch_size, seq_len]
- `context` (torch.Tensor, optional): Context tensor for collapse
- `collapse` (bool): Whether to collapse superposition (default: False)

**Returns:**

- `QuantumBERTOutput`: Model outputs with last_hidden_state and pooler_output

#### `get_uncertainty(input_ids)`

Calculate uncertainty for given input tokens.

**Returns:**

- `torch.Tensor`: Uncertainty scores [batch_size, seq_len]

### QuantumTransformer

Quantum-enhanced Transformer model.

```python
class qembed.models.quantum_transformer.QuantumTransformer(
    vocab_size: int,
    hidden_size: int = 512,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 8,
    num_quantum_states: int = 4,
    intermediate_size: int = 2048,
    dropout: float = 0.1,
    max_position_embeddings: int = 1024
)
```

**Methods:**

#### `forward(input_ids, mask=None, context=None, collapse=False)`

Forward pass through the quantum transformer.

#### `generate(input_ids, max_length, num_beams=1, context=None)`

Generate text using the quantum transformer.

### HybridModel

Hybrid classical-quantum model.

```python
class qembed.models.hybrid_models.HybridModel(
    vocab_size: int,
    hidden_size: int = 512,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 8,
    num_quantum_states: int = 4,
    intermediate_size: int = 2048,
    dropout: float = 0.1,
    max_position_embeddings: int = 1024
)
```

## 🎯 Training Utilities

### QuantumTrainer

Quantum-aware training loop.

```python
class qembed.training.quantum_trainer.QuantumTrainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    device: str = 'cpu',
    quantum_training_config: dict = None
)
```

**Parameters:**

- `model` (torch.nn.Module): Model to train
- `optimizer` (torch.optim.Optimizer): Optimizer
- `loss_fn` (callable): Loss function
- `device` (str): Device to use ('cpu' or 'cuda')
- `quantum_training_config` (dict): Quantum training configuration

**Methods:**

#### `train_epoch(dataloader, context=None)`

Train for one epoch.

**Returns:**

- `dict`: Training metrics

#### `validate(dataloader, context=None)`

Validate the model.

**Returns:**

- `dict`: Validation metrics

#### `train(train_dataloader, val_dataloader, num_epochs, context=None)`

Full training loop.

**Returns:**

- `dict`: Training history

#### `save_checkpoint(path, epoch, metrics)`

Save model checkpoint.

#### `load_checkpoint(path, model, optimizer, device)`

Load model checkpoint.

### QuantumLoss

Base quantum loss function.

```python
class qembed.training.losses.QuantumLoss(
    base_loss: str = 'cross_entropy',
    quantum_regularization: float = 0.1,
    uncertainty_regularization: float = 0.05
)
```

**Parameters:**

- `base_loss` (str): Base loss function ('cross_entropy', 'mse', 'mae')
- `quantum_regularization` (float): Quantum regularization weight
- `uncertainty_regularization` (float): Uncertainty regularization weight

### SuperpositionLoss

Loss function for training superposition states.

```python
class qembed.training.losses.SuperpositionLoss(
    base_loss: str = 'cross_entropy',
    superposition_regularization: float = 0.1,
    collapse_regularization: float = 0.05
)
```

### EntanglementLoss

Loss function for training entanglement correlations.

```python
class qembed.training.losses.EntanglementLoss(
    base_loss: str = 'cross_entropy',
    entanglement_regularization: float = 0.1,
    correlation_regularization: float = 0.05
)
```

### UncertaintyLoss

Loss function for uncertainty quantification.

```python
class qembed.training.losses.UncertaintyLoss(
    base_loss: str = 'cross_entropy',
    uncertainty_regularization: float = 0.1,
    calibration_regularization: float = 0.05
)
```

### QuantumOptimizer

Base quantum optimizer.

```python
class qembed.training.optimizers.QuantumOptimizer(
    params: Iterable,
    lr: float = 0.001,
    quantum_lr_factor: float = 1.5,
    uncertainty_threshold: float = 0.5,
    **kwargs
)
```

**Parameters:**

- `params` (Iterable): Model parameters
- `lr` (float): Learning rate
- `quantum_lr_factor` (float): Quantum learning rate factor
- `uncertainty_threshold` (float): Uncertainty threshold for LR adjustment

### SuperpositionOptimizer

Optimizer for superposition states.

```python
class qembed.training.optimizers.SuperpositionOptimizer(
    params: Iterable,
    lr: float = 0.001,
    superposition_schedule: str = 'linear',
    collapse_threshold: float = 0.3,
    **kwargs
)
```

### EntanglementOptimizer

Optimizer for entanglement correlations.

```python
class qembed.training.optimizers.EntanglementOptimizer(
    params: Iterable,
    lr: float = 0.001,
    entanglement_strength: float = 0.8,
    correlation_threshold: float = 0.6,
    **kwargs
)
```

## 🛠️ Utility Functions

### QuantumUtils

Static utility functions for quantum operations.

```python
class qembed.utils.quantum_utils.QuantumUtils
```

**Static Methods:**

#### `create_superposition(states, weights=None)`

Create superposition of quantum states.

#### `measure_superposition(superposition, basis)`

Measure superposition in given basis.

#### `create_bell_state(dim)`

Create Bell state entanglement.

#### `create_ghz_state(dim, num_parties)`

Create GHZ state entanglement.

#### `compute_fidelity(state1, state2)`

Compute fidelity between quantum states.

#### `compute_entanglement_entropy(state)`

Compute entanglement entropy.

#### `apply_quantum_gate(state, gate)`

Apply quantum gate to state.

#### `create_quantum_circuit(num_qubits)`

Create quantum circuit.

#### `quantum_fourier_transform(state)`

Apply quantum Fourier transform.

### QuantumVisualization

Visualization tools for quantum states.

```python
class qembed.utils.visualization.QuantumVisualization(
    figsize: tuple = (10, 8),
    style: str = 'default'
)
```

**Methods:**

#### `plot_superposition_states(superposition_states)`

Plot superposition states.

#### `plot_entanglement_correlations(embeddings)`

Plot entanglement correlations.

#### `plot_uncertainty_analysis(uncertainty)`

Plot uncertainty analysis.

#### `plot_quantum_circuit(circuit)`

Plot quantum circuit.

#### `plot_3d_quantum_states(states)`

Plot 3D quantum states.

### QuantumMetrics

Evaluation metrics for quantum models.

```python
class qembed.utils.metrics.QuantumMetrics()
```

**Methods:**

#### `compute_superposition_quality(embeddings)`

Compute superposition quality metric.

#### `compute_entanglement_strength(embeddings)`

Compute entanglement strength metric.

#### `compute_uncertainty_calibration(predictions, targets, uncertainty)`

Compute uncertainty calibration.

#### `compute_quantum_metrics(model, dataloader)`

Compute comprehensive quantum metrics.

## 📊 Dataset Classes

### PolysemyDataset

Dataset for polysemy testing.

```python
class qembed.datasets.polysemy_datasets.PolysemyDataset(
    data: List[Dict],
    tokenizer: callable,
    max_length: int = 128,
    num_quantum_states: int = 4
)
```

**Parameters:**

- `data` (List[Dict]): List of data samples
- `tokenizer` (callable): Tokenizer function
- `max_length` (int): Maximum sequence length
- `num_quantum_states` (int): Number of quantum states

### WordSenseDisambiguationDataset

Dataset for word sense disambiguation.

```python
class qembed.datasets.polysemy_datasets.WordSenseDisambiguationDataset(
    data: List[Dict],
    tokenizer: callable,
    max_length: int = 128,
    num_senses: int = 5
)
```

### QuantumDataLoader

Quantum-aware data loader.

```python
class qembed.datasets.quantum_data_loaders.QuantumDataLoader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    quantum_batching: bool = True
)
```

### UncertaintyDataLoader

Uncertainty-aware data loader.

```python
class qembed.datasets.quantum_data_loaders.UncertaintyDataLoader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    sampling_strategy: str = 'uncertainty'
)
```

## 📝 Examples

### Basic Usage

```python
# Basic quantum embeddings
from qembed.core.quantum_embeddings import QuantumEmbeddings

embeddings = QuantumEmbeddings(vocab_size=1000, embedding_dim=128, num_states=4)
input_ids = torch.randint(0, 1000, (2, 10))

# Superposition state
superposition = embeddings(input_ids, collapse=False)

# Collapsed state
collapsed = embeddings(input_ids, collapse=True)

# Uncertainty
uncertainty = embeddings.get_uncertainty(input_ids)
```

### Quantum BERT

```python
# Quantum BERT model
from qembed.models.quantum_bert import QuantumBERT

model = QuantumBERT(
    vocab_size=30000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_quantum_states=4
)

input_ids = torch.randint(0, 30000, (2, 10))
attention_mask = torch.ones(2, 10, dtype=torch.bool)

# Forward pass
outputs = model(input_ids, attention_mask=attention_mask, collapse=False)

# Get uncertainty
uncertainty = model.get_uncertainty(input_ids)
```

### Training

```python
# Quantum training
from qembed.training.quantum_trainer import QuantumTrainer
from qembed.training.losses import QuantumLoss
from qembed.training.optimizers import QuantumOptimizer

loss_fn = QuantumLoss(
    base_loss='cross_entropy',
    quantum_regularization=0.1,
    uncertainty_regularization=0.05
)

optimizer = QuantumOptimizer(
    model.parameters(),
    lr=0.001,
    quantum_lr_factor=1.5
)

trainer = QuantumTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device='cpu'
)

# Train
history = trainer.train(train_dataloader, val_dataloader, num_epochs=10)
```

### Visualization

```python
# Quantum visualization
from qembed.utils.visualization import QuantumVisualization

viz = QuantumVisualization()

# Plot superposition states
fig = viz.plot_superposition_states(superposition_embeddings)

# Plot uncertainty
fig = viz.plot_uncertainty_analysis(uncertainty)

# Plot entanglement
fig = viz.plot_entanglement_correlations(embeddings)
```

### Metrics

```python
# Quantum metrics
from qembed.utils.metrics import QuantumMetrics

metrics = QuantumMetrics()

# Compute metrics
superposition_quality = metrics.compute_superposition_quality(embeddings)
entanglement_strength = metrics.compute_entanglement_strength(embeddings)
uncertainty_calibration = metrics.compute_uncertainty_calibration(
    predictions, targets, uncertainty
)
```

## 🔗 Related Documentation

- **[Installation Guide](installation.md)** - Setup instructions
- **[Quick Start Tutorial](quickstart.md)** - Getting started guide
- **[Examples](../examples/)** - Working code examples
- **[GitHub Repository](https://github.com/qembed/qembed)** - Source code

---

This API reference covers all the main components of QEmbed. For more detailed examples and use cases, check out the examples directory and tutorials.
