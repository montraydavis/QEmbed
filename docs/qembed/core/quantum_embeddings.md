# Quantum Embeddings

## Overview

The `QuantumEmbeddings` class implements quantum-inspired embeddings that maintain superposition states, allowing tokens to exist in multiple semantic states simultaneously until measured in a specific context.

## Class Definition

### `QuantumEmbeddings`

```python
class QuantumEmbeddings(nn.Module)
```

Quantum-inspired embeddings that maintain superposition states. Each token can exist in multiple semantic states simultaneously until measured in a specific context, allowing for better representation of polysemy and contextual ambiguity.

#### Constructor

```python
def __init__(
    self,
    config: Optional[BertConfig] = None,
    vocab_size: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    num_states: int = 4,
    superposition_strength: float = 0.5,
    device: Optional[str] = None
)
```

**Parameters:**

- `config` (Optional[BertConfig]): BERT configuration object (preferred)
- `vocab_size` (Optional[int]): Size of the vocabulary (fallback)
- `embedding_dim` (Optional[int]): Dimension of the embedding vectors (fallback)
- `num_states` (int): Number of possible quantum states per token (default: 4)
- `superposition_strength` (float): Strength of superposition mixing (default: 0.5)
- `device` (Optional[str]): Device to place tensors on

**Note:** Either provide a BERT config object or both `vocab_size` and `embedding_dim`.

#### Methods

##### `forward()`

```python
def forward(
    self,
    input_ids: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    collapse: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]
```

Forward pass through the quantum embeddings.

**Parameters:**

- `input_ids` (torch.Tensor): Input token IDs [batch_size, seq_len]
- `token_type_ids` (Optional[torch.Tensor]): Token type IDs for BERT compatibility
- `position_ids` (Optional[torch.Tensor]): Position IDs for BERT compatibility
- `attention_mask` (Optional[torch.Tensor]): Attention mask for BERT compatibility
- `context` (Optional[torch.Tensor]): Context information for state collapse
- `collapse` (bool): Whether to collapse superposition states

**Returns:**

- `Tuple[torch.Tensor, torch.Tensor]`: (embeddings, uncertainty_scores)
  - `embeddings`: Final embeddings [batch_size, seq_len, embedding_dim]
  - `uncertainty_scores`: Uncertainty scores for each token [batch_size, seq_len]

##### `get_uncertainty_from_states()`

```python
def get_uncertainty_from_states(self, state_embeds: torch.Tensor) -> torch.Tensor
```

Compute uncertainty from quantum states.

**Parameters:**

- `state_embeds` (torch.Tensor): Superposition states [batch_size, seq_len, num_states, embedding_dim]

**Returns:**

- `torch.Tensor`: Uncertainty scores [batch_size, seq_len]

##### `get_uncertainty()`

```python
def get_uncertainty(self, input_ids: torch.Tensor) -> torch.Tensor
```

Compute uncertainty for each token based on state variance.

**Parameters:**

- `input_ids` (torch.Tensor): Input token IDs [batch_size, seq_len]

**Returns:**

- `torch.Tensor`: Uncertainty scores [batch_size, seq_len]

##### `forward_legacy()`

```python
def forward_legacy(
    self,
    input_ids: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    collapse: bool = False
) -> torch.Tensor
```

Legacy forward pass through quantum embeddings (maintains backward compatibility).

**Parameters:**

- `input_ids` (torch.Tensor): Input token IDs [batch_size, seq_len]
- `context` (Optional[torch.Tensor]): Optional context tensor for measurement
- `collapse` (bool): Whether to collapse superposition to single state

**Returns:**

- `torch.Tensor`: Quantum embeddings [batch_size, seq_len, embedding_dim]

#### Private Methods

##### `_create_superposition()`

```python
def _create_superposition(self, state_embeds: torch.Tensor) -> torch.Tensor
```

Create superposition of quantum states.

**Parameters:**

- `state_embeds` (torch.Tensor): State embeddings [batch_size, seq_len, num_states, embedding_dim]

**Returns:**

- `torch.Tensor`: Superposition embeddings [batch_size, seq_len, embedding_dim]

##### `_collapse_superposition()`

```python
def _collapse_superposition(
    self, 
    state_embeds: torch.Tensor, 
    context: torch.Tensor
) -> torch.Tensor
```

Collapse superposition based on context.

**Parameters:**

- `state_embeds` (torch.Tensor): Superposition states to collapse
- `context` (torch.Tensor): Context information for collapse

**Returns:**

- `torch.Tensor`: Collapsed embeddings [batch_size, seq_len, embedding_dim]

## Usage Examples

### Basic Usage

```python
from qembed.core import QuantumEmbeddings
import torch

# Initialize with BERT config
from transformers import BertConfig
config = BertConfig(vocab_size=30522, hidden_size=768)
embedding = QuantumEmbeddings(config=config)

# Or initialize with explicit parameters
embedding = QuantumEmbeddings(vocab_size=10000, embedding_dim=256)

# Create sample input
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# Forward pass
embeddings, uncertainty = embedding(input_ids)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Uncertainty scores: {uncertainty}")
```

### With Context Collapse

```python
# Create context information
context = torch.randn(1, 5, 256)  # [batch_size, seq_len, embedding_dim]

# Forward pass with context collapse
embeddings, uncertainty = embedding(
    input_ids=input_ids,
    context=context,
    collapse=True
)
```

### BERT-Compatible Interface

```python
# Use BERT-compatible parameters
embeddings, uncertainty = embedding(
    input_ids=input_ids,
    token_type_ids=torch.zeros_like(input_ids),
    position_ids=torch.arange(input_ids.size(1)).unsqueeze(0),
    attention_mask=torch.ones_like(input_ids)
)
```

## Key Features

- **Superposition States**: Maintains multiple semantic states per token
- **BERT Compatibility**: Drop-in replacement for BERT embeddings
- **Context-Aware Collapse**: Learns optimal collapse strategies
- **Uncertainty Quantification**: Built-in uncertainty estimation
- **Position and Token Type Embeddings**: Full BERT compatibility
- **Learnable Superposition Matrix**: Adapts superposition mixing

## Architecture Details

The model consists of several key components:

1. **State Embeddings**: `[vocab_size, num_states, embedding_dim]` parameter tensor
2. **Superposition Matrix**: `[num_states, num_states]` learnable mixing matrix
3. **Position Embeddings**: Standard BERT position embeddings
4. **Token Type Embeddings**: Standard BERT token type embeddings
5. **Layer Normalization**: BERT-standard normalization
6. **Dropout**: Regularization during training

## Training Considerations

- The superposition matrix is initialized with identity + small random noise
- Superposition strength controls the mixing between states
- Uncertainty scores can be used as additional supervision signals
- Context collapse can be learned end-to-end with the model
