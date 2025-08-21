# Quantum Measurement

## Overview

The `QuantumMeasurement` module implements various quantum measurement operators that can be used to collapse superposition states and extract classical information from quantum embeddings. This provides the foundation for implementing various quantum measurement strategies on embeddings.

## Class Hierarchy

### Base Class: `QuantumMeasurement`

```python
class QuantumMeasurement(nn.Module)
```

Base quantum measurement operator. This class provides the foundation for implementing various quantum measurement strategies on embeddings.

#### Constructor

```python
def __init__(
    self,
    embedding_dim: int,
    measurement_basis: str = "computational",
    noise_level: float = 0.0
)
```

**Parameters:**

- `embedding_dim` (int): Dimension of input embeddings
- `measurement_basis` (str): Basis for measurement ('computational', 'bell', 'custom') (default: "computational")
- `noise_level` (float): Level of measurement noise to add (default: 0.0)

#### Methods

##### `forward()`

```python
def forward(
    self,
    embeddings: torch.Tensor,
    collapse_probability: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]
```

Perform quantum measurement on embeddings.

**Parameters:**

- `embeddings` (torch.Tensor): Input quantum embeddings [batch_size, seq_len, embedding_dim]
- `collapse_probability` (float): Probability of collapsing superposition (default: 1.0)

**Returns:**

- `Tuple[torch.Tensor, torch.Tensor]`: (measured_embeddings, measurement_results)

##### `_create_bell_basis()`

```python
def _create_bell_basis(self) -> torch.Tensor
```

Create Bell state measurement basis.

**Returns:**

- `torch.Tensor`: Bell basis matrix [embedding_dim, embedding_dim]

##### `_add_measurement_noise()`

```python
def _add_measurement_noise(
    self,
    embeddings: torch.Tensor,
    noise_level: float
) -> torch.Tensor
```

Add measurement noise to embeddings.

**Parameters:**

- `embeddings` (torch.Tensor): Input embeddings
- `noise_level` (float): Noise level

**Returns:**

- `torch.Tensor`: Noisy embeddings

## Measurement Bases

### 1. Computational Basis (`measurement_basis="computational"`)

Standard computational basis measurement.

**Characteristics:**

- Identity matrix basis
- Standard quantum measurement
- Minimal information loss

### 2. Bell Basis (`measurement_basis="bell"`)

Bell state-inspired measurement basis.

**Characteristics:**

- Entanglement-aware measurement
- Off-diagonal elements for correlation
- Enhanced quantum information extraction

### 3. Custom Basis (`measurement_basis="custom"`)

Learnable measurement basis.

**Characteristics:**

- Adaptive measurement strategy
- Task-specific optimization
- Learnable quantum operations

## Advanced Measurement Strategies

### `AdaptiveMeasurement`

```python
class AdaptiveMeasurement(QuantumMeasurement)
```

Adaptive quantum measurement operator. This operator learns to choose the optimal measurement basis for each embedding based on context and uncertainty.

#### Constructor

```python
def __init__(
    self,
    embedding_dim: int,
    num_bases: int = 4,
    temperature: float = 1.0
)
```

**Parameters:**

- `embedding_dim` (int): Dimension of input embeddings
- `num_bases` (int): Number of measurement bases to choose from (default: 4)
- `temperature` (float): Temperature for basis selection (default: 1.0)

#### Methods

##### `forward()`

```python
def forward(
    self,
    embeddings: torch.Tensor,
    collapse_probability: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Perform adaptive quantum measurement.

**Parameters:**

- `embeddings` (torch.Tensor): Input quantum embeddings [batch_size, seq_len, embedding_dim]
- `collapse_probability` (float): Probability of collapsing superposition (default: 1.0)

**Returns:**

- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: (measured_embeddings, measurement_results, basis_weights)

### `WeakMeasurement`

```python
class WeakMeasurement(QuantumMeasurement)
```

Weak quantum measurement operator. This operator performs weak measurements that don't completely collapse the quantum state, preserving some quantum properties.

#### Constructor

```python
def __init__(
    self,
    embedding_dim: int,
    measurement_strength: float = 0.5,
    decoherence_rate: float = 0.1
)
```

**Parameters:**

- `embedding_dim` (int): Dimension of input embeddings
- `measurement_strength` (float): Strength of measurement (0 = no collapse, 1 = full collapse) (default: 0.5)
- `decoherence_rate` (float): Rate of decoherence during measurement (default: 0.1)

#### Methods

##### `forward()`

```python
def forward(
    self,
    embeddings: torch.Tensor,
    measurement_strength: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

Perform weak quantum measurement.

**Parameters:**

- `embeddings` (torch.Tensor): Input quantum embeddings [batch_size, seq_len, embedding_dim]
- `measurement_strength` (Optional[float]): Override default measurement strength

**Returns:**

- `Tuple[torch.Tensor, torch.Tensor]`: (weakly_measured_embeddings, measurement_strength_used)

### `POVMMeasurement`

```python
class POVMMeasurement(QuantumMeasurement)
```

Positive Operator-Valued Measure (POVM) measurement. Implements POVM measurements which are more general than projective measurements and can handle mixed states.

#### Constructor

```python
def __init__(
    self,
    embedding_dim: int,
    num_povm_elements: int = 4
)
```

**Parameters:**

- `embedding_dim` (int): Dimension of input embeddings
- `num_povm_elements` (int): Number of POVM elements (default: 4)

#### Methods

##### `forward()`

```python
def forward(
    self,
    embeddings: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]
```

Perform POVM measurement.

**Parameters:**

- `embeddings` (torch.Tensor): Input quantum embeddings [batch_size, seq_len, embedding_dim]

**Returns:**

- `Tuple[torch.Tensor, torch.Tensor]`: (measured_embeddings, povm_probabilities)

## Usage Examples with Advanced Classes

### Adaptive Measurement

```python
from qembed.core import AdaptiveMeasurement

# Initialize adaptive measurement
adaptive_measurement = AdaptiveMeasurement(
    embedding_dim=256,
    num_bases=4,
    temperature=0.5
)

# Perform adaptive measurement
measured, results, basis_weights = adaptive_measurement(embeddings)
print(f"Basis weights: {basis_weights}")
```

### Weak Measurement

```python
from qembed.core import WeakMeasurement

# Initialize weak measurement
weak_measurement = WeakMeasurement(
    embedding_dim=256,
    measurement_strength=0.3,
    decoherence_rate=0.05
)

# Perform weak measurement
weak_measured, strength_used = weak_measurement(embeddings)
print(f"Measurement strength used: {strength_used}")
```

### POVM Measurement

```python
from qembed.core import POVMMeasurement

# Initialize POVM measurement
povm_measurement = POVMMeasurement(
    embedding_dim=256,
    num_povm_elements=6
)

# Perform POVM measurement
measured, povm_probs = povm_measurement(embeddings)
print(f"POVM probabilities: {povm_probs}")
```

## Integration with Quantum Embeddings

The `QuantumMeasurement` class is designed to work seamlessly with `QuantumEmbeddings`:

```python
from qembed.core import QuantumEmbeddings, QuantumMeasurement

# Initialize components
quantum_emb = QuantumEmbeddings(vocab_size=10000, embedding_dim=256)
measurement = QuantumMeasurement(embedding_dim=256)

# Forward pass
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
embeddings, uncertainty = quantum_emb(input_ids)

# Perform measurement
measured, results = measurement(embeddings)
```

## Mathematical Formulation

### Measurement Process

For input embeddings \(|\psi\rangle \in \mathbb{C}^{B \times L \times D}\):

1. **Basis Projection**: \(|\phi\rangle = M|\psi\rangle\)
2. **Noise Addition**: \(|\phi'\rangle = |\phi\rangle + \mathcal{N}(0, \sigma^2)\)
3. **Collapse Decision**: \(|\psi_{\text{final}}\rangle = \begin{cases} |\phi'\rangle & \text{if collapse} \\ |\psi\rangle & \text{otherwise} \end{cases}\)

Where:

- \(M\): measurement basis matrix
- \(\sigma^2\): noise variance
- \(\text{collapse}\): Bernoulli random variable

### Bell Basis Construction

For Bell state measurement:
\[M_{ij} = \begin{cases}
\frac{1}{\sqrt{2}} & \text{if } i = j \text{ or } i = j \pm 1 \\
0 & \text{otherwise}
\end{cases}\]

## Training Considerations

- **Noise Level**: Higher noise improves robustness but reduces measurement precision
- **Basis Selection**: Choose based on task requirements and quantum properties
- **Collapse Probability**: Controls the trade-off between quantum and classical information
- **Basis Learning**: Custom bases can be learned end-to-end

## Performance Characteristics

- **Computational Complexity**: \(O(B \times L \times D^2)\) for basis projection
- **Memory Usage**: \(O(D^2)\) for basis matrix storage
- **Gradient Flow**: All operations maintain differentiability
- **Device Compatibility**: Works on CPU and GPU

## Error Handling

The measurement operators include robust error handling:

```python
try:
    measured, results = measurement(embeddings)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Handle memory issues
        measured, results = measurement(embeddings.to('cpu'))
    else:
        raise e
```

## Future Extensions

The measurement framework is designed to be extensible:

- **Multi-qubit measurements**: For higher-dimensional quantum systems
- **Continuous measurements**: For time-evolving quantum states
- **Quantum error correction**: For robust measurement in noisy environments
- **Hybrid classical-quantum measurements**: For mixed quantum-classical systems
