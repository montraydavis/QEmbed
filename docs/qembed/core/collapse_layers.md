# Context Collapse Layers

## Overview

The `ContextCollapseLayer` implements context-driven collapse mechanisms for quantum embeddings. This layer learns to collapse quantum superposition states based on surrounding context, effectively implementing quantum measurement in a learned manner.

## Class Definition

### `ContextCollapseLayer`

```python
class ContextCollapseLayer(nn.Module)
```

Context-driven collapse layer for quantum embeddings. This layer learns to collapse quantum superposition states based on surrounding context, effectively implementing quantum measurement in a learned manner.

#### Constructor

```python
def __init__(
    self,
    embedding_dim: int,
    context_window: int = 5,
    collapse_strategy: str = "attention",
    temperature: float = 1.0
)
```

**Parameters:**

- `embedding_dim` (int): Dimension of input embeddings
- `context_window` (int): Size of context window for collapse (default: 5)
- `collapse_strategy` (str): Strategy for collapsing ('attention', 'conv', 'rnn') (default: "attention")
- `temperature` (float): Temperature for softmax in attention (default: 1.0)

#### Methods

##### `forward()`

```python
def forward(
    self,
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor
```

Forward pass through context collapse layer.

**Parameters:**

- `embeddings` (torch.Tensor): Input embeddings [batch_size, seq_len, embedding_dim]
- `attention_mask` (Optional[torch.Tensor]): Optional attention mask

**Returns:**

- `torch.Tensor`: Collapsed embeddings [batch_size, seq_len, embedding_dim]

##### `_generate_context()`

```python
def _generate_context(
    self,
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor
```

Generate context representation using the selected strategy.

**Parameters:**

- `embeddings` (torch.Tensor): Input embeddings
- `attention_mask` (Optional[torch.Tensor]): Optional attention mask

**Returns:**

- `torch.Tensor`: Context representation [batch_size, seq_len, embedding_dim]

### `AdaptiveCollapseLayer`

```python
class AdaptiveCollapseLayer(nn.Module)
```

Adaptive collapse layer that learns when to collapse. This layer learns to predict the optimal collapse strategy for each token based on its context and uncertainty.

#### Constructor

```python
def __init__(
    self,
    embedding_dim: int,
    num_strategies: int = 3,
    temperature: float = 1.0
)
```

**Parameters:**

- `embedding_dim` (int): Dimension of input embeddings
- `num_strategies` (int): Number of available collapse strategies (default: 3)
- `temperature` (float): Temperature for strategy selection (default: 1.0)

#### Methods

##### `forward()`

```python
def forward(
    self,
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

Forward pass with adaptive strategy selection.

**Parameters:**

- `embeddings` (torch.Tensor): Input embeddings [batch_size, seq_len, embedding_dim]
- `attention_mask` (Optional[torch.Tensor]): Optional attention mask

**Returns:**

- `Tuple[torch.Tensor, torch.Tensor]`: (collapsed_embeddings, strategy_weights)

## Collapse Strategies

### 1. Attention Strategy (`collapse_strategy="attention"`)

Uses multi-head attention to learn context-aware collapse patterns.

**Components:**

- `context_attention`: Multi-head attention layer
- `context_projection`: Linear projection layer

**Advantages:**

- Learns optimal attention patterns
- Handles variable sequence lengths
- Position-aware context modeling

### 2. Convolutional Strategy (`collapse_strategy="conv"`)

Uses 1D convolutions to capture local context patterns.

**Components:**

- `context_conv`: 1D convolutional layer
- `context_norm`: Layer normalization

**Advantages:**

- Efficient local context capture
- Fixed receptive field
- Good for short-range dependencies

### 3. RNN Strategy (`collapse_strategy="rnn"`)

Uses bidirectional LSTM to capture sequential context.

**Components:**

- `context_rnn`: Bidirectional LSTM layer
- `context_projection`: Linear projection layer

**Advantages:**

- Captures long-range dependencies
- Sequential context modeling
- Good for language modeling tasks

## Usage Examples

### Basic Usage

```python
from qembed.core import ContextCollapseLayer
import torch

# Initialize collapse layer
collapse_layer = ContextCollapseLayer(
    embedding_dim=256,
    context_window=5,
    collapse_strategy="attention"
)

# Create sample embeddings
embeddings = torch.randn(2, 10, 256)  # [batch_size, seq_len, embedding_dim]

# Apply context collapse
collapsed = collapse_layer(embeddings)
print(f"Collapsed shape: {collapsed.shape}")
```

### Different Strategies

```python
# Attention-based collapse
attention_collapse = ContextCollapseLayer(
    embedding_dim=256,
    collapse_strategy="attention"
)

# Convolutional collapse
conv_collapse = ContextCollapseLayer(
    embedding_dim=256,
    collapse_strategy="conv",
    context_window=7
)

# RNN-based collapse
rnn_collapse = ContextCollapseLayer(
    embedding_dim=256,
    collapse_strategy="rnn"
)

# Apply different strategies
collapsed_attn = attention_collapse(embeddings)
collapsed_conv = conv_collapse(embeddings)
collapsed_rnn = rnn_collapse(embeddings)
```

### With Attention Mask

```python
# Create attention mask
attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

# Apply with mask
collapsed = collapse_layer(embeddings, attention_mask=attention_mask)
```

## Architecture Details

The layer consists of several key components:

1. **Context Generation**: Strategy-specific context representation
2. **Combination**: Concatenation of original embeddings with context
3. **Final Projection**: Linear projection to output dimension
4. **Normalization**: Layer normalization for stability

### Mathematical Formulation

For input embeddings \(E \in \mathbb{R}^{B \times L \times D}\):

1. **Context Generation**: \(C = f_{\text{strategy}}(E)\)
2. **Combination**: \(H = [E; C]\)
3. **Final Projection**: \(O = \text{LayerNorm}(H \cdot W + b)\)

Where:

- \(B\): batch size
- \(L\): sequence length  
- \(D\): embedding dimension
- \(f_{\text{strategy}}\): context generation function
- \(W, b\): learnable parameters

## Training Considerations

- **Temperature Scaling**: Lower temperature makes attention more focused
- **Context Window**: Larger windows capture more context but increase computation
- **Strategy Selection**: Choose based on task requirements and computational constraints
- **Gradient Flow**: All strategies maintain end-to-end differentiability

## Integration with Quantum Embeddings

The `ContextCollapseLayer` is designed to work seamlessly with `QuantumEmbeddings`:

```python
from qembed.core import QuantumEmbeddings, ContextCollapseLayer

# Initialize components
quantum_emb = QuantumEmbeddings(vocab_size=10000, embedding_dim=256)
collapse_layer = ContextCollapseLayer(embedding_dim=256)

# Forward pass
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
embeddings, uncertainty = quantum_emb(input_ids)

# Apply context collapse
collapsed = collapse_layer(embeddings)
```

This allows for learned context-aware collapse of quantum superposition states, enabling the model to adaptively choose when and how to measure quantum states based on surrounding context.
