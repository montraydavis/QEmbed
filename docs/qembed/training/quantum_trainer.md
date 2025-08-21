# Quantum Trainer

## Overview

The `QuantumTrainer` class implements a quantum-aware training loop specifically designed for models with quantum-inspired components. It handles training with quantum-specific considerations like superposition collapse, entanglement training, and uncertainty quantification.

## Class Definition

### `QuantumTrainer`

```python
class QuantumTrainer
```

Quantum-aware trainer for quantum-enhanced models. Handles training with quantum-specific considerations like superposition collapse, entanglement training, and uncertainty quantification.

#### Constructor

```python
def __init__(
    self,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: Optional[str] = None,
    quantum_training_config: Optional[Dict[str, Any]] = None
)
```

**Parameters:**

- `model` (nn.Module): Model to train
- `optimizer` (torch.optim.Optimizer): Optimizer for training
- `loss_fn` (nn.Module): Loss function
- `device` (Optional[str]): Device to train on (default: auto-detect)
- `quantum_training_config` (Optional[Dict[str, Any]]): Configuration for quantum training

**Quantum Training Configuration Options:**

- `superposition_schedule` (str): Schedule for superposition collapse ('linear', 'cyclic', 'constant') (default: 'linear')
- `entanglement_training` (bool): Whether to use entanglement-specific training (default: True)
- `uncertainty_regularization` (float): Weight for uncertainty regularization (default: 0.1)

#### Methods

##### `train_epoch()`

```python
def train_epoch(
    self,
    dataloader: DataLoader,
    epoch: int,
    collapse_probability: float = 1.0
) -> Dict[str, float]
```

Train for one epoch.

**Parameters:**

- `dataloader` (DataLoader): Training data loader
- `epoch` (int): Current epoch number
- `collapse_probability` (float): Probability of collapsing quantum states (default: 1.0)

**Returns:**

- `Dict[str, float]`: Dictionary of training metrics

##### `train()`

```python
def train(
    self,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    early_stopping_patience: int = 5,
    save_best_model: bool = True,
    model_save_path: str = "./best_model.pth"
) -> Dict[str, List[float]]
```

Complete training loop.

**Parameters:**

- `train_dataloader` (DataLoader): Training data loader
- `val_dataloader` (Optional[DataLoader]): Validation data loader (optional)
- `num_epochs` (int): Number of training epochs (default: 10)
- `early_stopping_patience` (int): Patience for early stopping (default: 5)
- `save_best_model` (bool): Whether to save the best model (default: True)
- `model_save_path` (str): Path to save the best model (default: "./best_model.pth")

**Returns:**

- `Dict[str, List[float]]`: Training history with metrics

##### `validate()`

```python
def validate(
    self,
    dataloader: DataLoader,
    collapse_probability: float = 1.0
) -> Dict[str, float]
```

Validate the model.

**Parameters:**

- `dataloader` (DataLoader): Validation data loader
- `collapse_probability` (float): Probability of collapsing quantum states (default: 1.0)

**Returns:**

- `Dict[str, float]`: Dictionary of validation metrics

##### `save_model()`

```python
def save_model(self, path: str, include_optimizer: bool = False)
```

Save the trained model.

**Parameters:**

- `path` (str): Path to save the model
- `include_optimizer` (bool): Whether to include optimizer state (default: False)

##### `load_model()`

```python
def load_model(self, path: str, include_optimizer: bool = False)
```

Load a saved model.

**Parameters:**

- `path` (str): Path to load the model from
- `include_optimizer` (bool): Whether to load optimizer state (default: False)

## Superposition Schedules

### 1. Linear Schedule (`superposition_schedule="linear"`)

Gradually increases collapse probability from 0 to 1 over training.

**Formula:**

```
collapse_probability = min(1.0, epoch / 10.0)
```

**Use Case:** Standard training where you want to gradually transition from quantum to classical behavior.

### 2. Cyclic Schedule (`superposition_schedule="cyclic"`)

Oscillates collapse probability between 0 and 1 during training.

**Formula:**

```
collapse_probability = 0.5 + 0.5 * sin(epoch * 0.1)
```

**Use Case:** Training scenarios where you want to maintain some quantum behavior throughout training.

### 3. Constant Schedule (`superposition_schedule="constant"`)

Maintains a fixed collapse probability throughout training.

**Use Case:** When you want consistent quantum behavior or manual control over collapse probability.

## Usage Examples

### Basic Training

```python
from qembed.training import QuantumTrainer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Initialize trainer
trainer = QuantumTrainer(
    model=quantum_model,
    optimizer=AdamW(quantum_model.parameters(), lr=2e-5),
    loss_fn=CrossEntropyLoss(),
    quantum_training_config={
        'superposition_schedule': 'linear',
        'entanglement_training': True,
        'uncertainty_regularization': 0.1
    }
)

# Train for multiple epochs
history = trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=10
)

# Access training history
print(f"Final training loss: {history['train_loss'][-1]}")
print(f"Final validation loss: {history['val_loss'][-1]}")
```

### Custom Training Loop

```python
# Custom training with manual epoch control
for epoch in range(num_epochs):
    # Train one epoch
    train_metrics = trainer.train_epoch(
        dataloader=train_loader,
        epoch=epoch,
        collapse_probability=0.1 + epoch * 0.05  # Custom schedule
    )
    
    # Validate
    val_metrics = trainer.validate(
        dataloader=val_loader,
        collapse_probability=1.0  # Always collapse during validation
    )
    
    # Log metrics
    print(f"Epoch {epoch}:")
    print(f"  Train Loss: {train_metrics['loss']:.4f}")
    print(f"  Val Loss: {val_metrics['loss']:.4f}")
    print(f"  Quantum Loss: {train_metrics['quantum_loss']:.4f}")
    print(f"  Uncertainty: {train_metrics['uncertainty']:.4f}")
```

### Advanced Configuration

```python
# Advanced quantum training configuration
advanced_config = {
    'superposition_schedule': 'cyclic',
    'entanglement_training': True,
    'uncertainty_regularization': 0.2,
    'entanglement_strength_schedule': 'adaptive',
    'collapse_strategy': 'context_aware',
    'quantum_gradient_clipping': True,
    'max_quantum_grad_norm': 1.0
}

trainer = QuantumTrainer(
    model=quantum_model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    quantum_training_config=advanced_config
)
```

## Training Workflows

### 1. Pre-training Workflow

```python
# Pre-train quantum model
pretrain_config = {
    'superposition_schedule': 'linear',
    'entanglement_training': True,
    'uncertainty_regularization': 0.05
}

pretrainer = QuantumTrainer(
    model=quantum_model,
    optimizer=pretrain_optimizer,
    loss_fn=mlm_loss,
    quantum_training_config=pretrain_config
)

# Pre-train
pretrainer.train(
    train_dataloader=pretrain_loader,
    num_epochs=100,
    save_best_model=True,
    model_save_path="./pretrained_quantum_model.pth"
)
```

### 2. Fine-tuning Workflow

```python
# Fine-tune for specific task
finetune_config = {
    'superposition_schedule': 'constant',
    'entanglement_training': False,  # Disable entanglement training
    'uncertainty_regularization': 0.1
}

finetuner = QuantumTrainer(
    model=quantum_model,
    optimizer=finetune_optimizer,
    loss_fn=task_loss,
    quantum_training_config=finetune_config
)

# Fine-tune
finetuner.train(
    train_dataloader=task_train_loader,
    val_dataloader=task_val_loader,
    num_epochs=5,
    early_stopping_patience=3
)
```

### 3. Continual Learning

```python
# Continual learning with quantum components
continual_config = {
    'superposition_schedule': 'cyclic',
    'entanglement_training': True,
    'uncertainty_regularization': 0.15,
    'quantum_memory_replay': True
}

continual_trainer = QuantumTrainer(
    model=quantum_model,
    optimizer=continual_optimizer,
    loss_fn=continual_loss,
    quantum_training_config=continual_config
)

# Train on new tasks while preserving quantum properties
for task_id, task_data in enumerate(task_sequence):
    continual_trainer.train(
        train_dataloader=task_data['train'],
        val_dataloader=task_data['val'],
        num_epochs=3
    )
```

## Monitoring and Metrics

### 1. Training Metrics

The trainer tracks various metrics during training:

```python
# Access training metrics
train_metrics = trainer.train_epoch(train_loader, epoch=0)

print(f"Loss: {train_metrics['loss']}")
print(f"Quantum Loss: {train_metrics['quantum_loss']}")
print(f"Uncertainty: {train_metrics['uncertainty']}")
print(f"Entanglement Strength: {train_metrics['entanglement_strength']}")
print(f"Superposition Coherence: {train_metrics['superposition_coherence']}")
```

### 2. Validation Metrics

```python
# Validate and get metrics
val_metrics = trainer.validate(val_loader)

print(f"Validation Loss: {val_metrics['loss']}")
print(f"Validation Accuracy: {val_metrics['accuracy']}")
print(f"Validation Uncertainty: {val_metrics['uncertainty']}")
```

### 3. Training History

```python
# Get complete training history
history = trainer.training_history

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history['uncertainty'])
plt.title('Uncertainty')

plt.subplot(1, 3, 3)
plt.plot(history['quantum_loss'])
plt.title('Quantum Loss')

plt.tight_layout()
plt.show()
```

## Advanced Features

### 1. Custom Loss Functions

```python
class QuantumAwareLoss(nn.Module):
    def __init__(self, base_loss, uncertainty_weight=0.1):
        super().__init__()
        self.base_loss = base_loss
        self.uncertainty_weight = uncertainty_weight
    
    def forward(self, outputs, labels, uncertainty=None):
        # Base task loss
        task_loss = self.base_loss(outputs.logits, labels)
        
        # Uncertainty regularization
        if uncertainty is not None:
            uncertainty_loss = uncertainty.mean()
            total_loss = task_loss + self.uncertainty_weight * uncertainty_loss
        else:
            total_loss = task_loss
        
        return total_loss

# Use custom loss
custom_loss = QuantumAwareLoss(CrossEntropyLoss(), uncertainty_weight=0.2)
trainer = QuantumTrainer(model, optimizer, custom_loss)
```

### 2. Gradient Clipping

```python
# Enable quantum-aware gradient clipping
quantum_config = {
    'quantum_gradient_clipping': True,
    'max_quantum_grad_norm': 1.0,
    'separate_clipping': True
}

trainer = QuantumTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    quantum_training_config=quantum_config
)
```

### 3. Model Checkpointing

```python
# Save checkpoints during training
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=10,
    save_best_model=True,
    model_save_path="./checkpoints/best_model.pth"
)

# Load best model
trainer.load_model("./checkpoints/best_model.pth")
```

## Best Practices

### 1. Superposition Schedule Selection

- **Linear**: Use for standard training with gradual quantum-to-classical transition
- **Cyclic**: Use for maintaining quantum behavior throughout training
- **Constant**: Use for consistent quantum behavior or manual control

### 2. Uncertainty Regularization

- **Start Small**: Begin with low uncertainty regularization (0.05-0.1)
- **Monitor**: Track uncertainty scores during training
- **Adjust**: Increase if model becomes overconfident, decrease if too uncertain

### 3. Entanglement Training

- **Enable Early**: Use entanglement training during pre-training
- **Disable Late**: Consider disabling during fine-tuning for specific tasks
- **Monitor Strength**: Track entanglement strength to ensure proper training

### 4. Early Stopping

- **Patience**: Use appropriate patience for your dataset size
- **Metrics**: Monitor both task and quantum metrics
- **Save Best**: Always save the best model for later use

## Troubleshooting

### Common Issues

1. **High Uncertainty**: Reduce uncertainty regularization weight
2. **Training Instability**: Use gradient clipping or reduce learning rate
3. **Poor Quantum Behavior**: Adjust superposition schedule or entanglement training
4. **Memory Issues**: Reduce batch size or disable some quantum features

### Debugging Tips

1. **Monitor Metrics**: Track all training metrics closely
2. **Check Configurations**: Verify quantum training configuration
3. **Gradient Flow**: Ensure gradients flow through quantum components
4. **Component Isolation**: Test quantum components independently

## Performance Optimization

### 1. Memory Efficiency

```python
# Optimize memory usage
quantum_config = {
    'gradient_checkpointing': True,
    'mixed_precision': True,
    'quantum_memory_efficient': True
}
```

### 2. Computational Efficiency

```python
# Optimize computation
quantum_config = {
    'quantum_parallel': True,
    'entanglement_optimization': 'sparse',
    'collapse_optimization': 'batched'
}
```

### 3. Distributed Training

```python
# Enable distributed training
quantum_config = {
    'distributed_training': True,
    'quantum_sync': True,
    'entanglement_sync': True
}
```
