# Quick Start Tutorial

Welcome to QEmbed! This tutorial will get you up and running with quantum-enhanced embeddings in just a few minutes.

## 🚀 Your First Quantum Embeddings

### 1. Basic Setup

First, let's import the necessary components:

```python
import torch
from qembed.core.quantum_embeddings import QuantumEmbeddings
from qembed.utils.visualization import QuantumVisualization
```

### 2. Create Quantum Embeddings

Create a simple quantum embedding layer:

```python
# Initialize quantum embeddings
quantum_embeddings = QuantumEmbeddings(
    vocab_size=1000,      # Size of your vocabulary
    embedding_dim=128,     # Dimension of embeddings
    num_states=4          # Number of quantum states per token
)

print(f"Created quantum embeddings with {quantum_embeddings.num_states} states")
```

### 3. Create Sample Input

```python
# Create sample token IDs
input_ids = torch.randint(0, 1000, (2, 10))  # 2 batches, 10 tokens each
print(f"Input shape: {input_ids.shape}")
print(f"Sample tokens: {input_ids[0]}")
```

### 4. Get Embeddings in Superposition

```python
# Get embeddings without collapsing (superposition state)
superposition_embeddings = quantum_embeddings(input_ids, collapse=False)
print(f"Superposition embeddings shape: {superposition_embeddings.shape}")

# Get uncertainty estimates
uncertainty = quantum_embeddings.get_uncertainty(input_ids)
print(f"Uncertainty shape: {uncertainty.shape}")
print(f"Average uncertainty: {uncertainty.mean().item():.3f}")
```

### 5. Collapse Superposition with Context

```python
# Create context tensor
context = torch.randn(2, 10, 128)

# Collapse superposition based on context
collapsed_embeddings = quantum_embeddings(
    input_ids, 
    context=context, 
    collapse=True
)
print(f"Collapsed embeddings shape: {collapsed_embeddings.shape}")

# Compare uncertainty before and after collapse
uncertainty_after = quantum_embeddings.get_uncertainty(input_ids)
print(f"Uncertainty after collapse: {uncertainty_after.mean().item():.3f}")
```

## 🔬 Visualizing Quantum States

### 6. Plot Superposition States

```python
# Create visualization object
viz = QuantumVisualization()

# Plot superposition states
fig = viz.plot_superposition_states(superposition_embeddings)
fig.suptitle("Quantum Superposition States")
fig.show()
```

### 7. Plot Uncertainty Analysis

```python
# Plot uncertainty analysis
fig = viz.plot_uncertainty_analysis(uncertainty)
fig.suptitle("Uncertainty Analysis")
fig.show()
```

## 🧠 Using Quantum-Enhanced Models

### 8. Create a Quantum BERT Model

```python
from qembed.models.quantum_bert import QuantumBERT

# Create a small quantum BERT model for testing
model = QuantumBERT(
    vocab_size=30000,
    hidden_size=256,        # Smaller for quick testing
    num_hidden_layers=2,    # Fewer layers for speed
    num_attention_heads=8,
    num_quantum_states=4
)

print(f"Created Quantum BERT with {model.num_hidden_layers} layers")
```

### 9. Forward Pass with Quantum Features

```python
# Create attention mask
attention_mask = torch.ones(2, 10, dtype=torch.bool)

# Forward pass without collapse (keep superposition)
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    collapse=False
)

print(f"Output shape: {outputs.last_hidden_state.shape}")
print(f"Pooler output shape: {outputs.pooler_output.shape}")

# Get uncertainty estimates
uncertainty = model.get_uncertainty(input_ids)
print(f"Model uncertainty: {uncertainty.mean().item():.3f}")
```

### 10. Forward Pass with Collapse

```python
# Forward pass with collapse
outputs_collapsed = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    collapse=True
)

print(f"Collapsed output shape: {outputs_collapsed.last_hidden_state.shape}")

# Compare uncertainty
uncertainty_collapsed = model.get_uncertainty(input_ids)
print(f"Collapsed uncertainty: {uncertainty_collapsed.mean().item():.3f}")
```

## 🎯 Training with Quantum Features

### 11. Set Up Training Components

```python
from qembed.training.quantum_trainer import QuantumTrainer
from qembed.training.losses import QuantumLoss
from qembed.training.optimizers import QuantumOptimizer

# Create loss function
loss_fn = QuantumLoss(
    base_loss='cross_entropy',
    quantum_regularization=0.1,
    uncertainty_regularization=0.05
)

# Create optimizer
optimizer = QuantumOptimizer(
    model.parameters(),
    lr=0.001,
    quantum_lr_factor=1.5
)

# Create trainer
trainer = QuantumTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device='cpu'  # Use 'cuda' if you have GPU
)
```

### 12. Create Sample Data

```python
# Create simple training data
batch_size, seq_len, num_classes = 4, 10, 100

# Sample inputs and targets
train_inputs = torch.randint(0, 30000, (batch_size, seq_len))
train_targets = torch.randint(0, num_classes, (batch_size, seq_len))

# Create simple dataloader
class SimpleDataLoader:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __iter__(self):
        yield (self.inputs, self.targets)

train_dataloader = SimpleDataLoader(train_inputs, train_targets)
val_dataloader = SimpleDataLoader(train_inputs, train_targets)  # Same for demo
```

### 13. Train the Model

```python
# Train for a few epochs
history = trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    num_epochs=3,
    context=None  # No context for this simple example
)

print("Training completed!")
print(f"Final train loss: {history['train_loss'][-1]:.4f}")
print(f"Final val loss: {history['val_loss'][-1]:.4f}")
```

## 🔍 Exploring Quantum Features

### 14. Analyze Superposition Quality

```python
from qembed.utils.metrics import QuantumMetrics

# Create metrics object
metrics = QuantumMetrics()

# Analyze superposition quality
superposition_quality = metrics.compute_superposition_quality(superposition_embeddings)
print(f"Superposition quality: {superposition_quality:.3f}")

# Analyze entanglement strength
entanglement_strength = metrics.compute_entanglement_strength(superposition_embeddings)
print(f"Entanglement strength: {entanglement_strength:.3f}")
```

### 15. Compare Classical vs Quantum

```python
# Get classical embeddings (first state only)
classical_embeddings = superposition_embeddings[:, :, 0, :]

# Compare with quantum embeddings
quantum_vs_classical = torch.norm(superposition_embeddings - classical_embeddings.unsqueeze(2))
print(f"Quantum vs Classical difference: {quantum_vs_classical.mean().item():.3f}")
```

## 🎨 Advanced Visualization

### 16. 3D Quantum State Visualization

```python
# Plot 3D quantum states
fig = viz.plot_3d_quantum_states(superposition_embeddings[0, :5, :, :])
fig.suptitle("3D Quantum States")
fig.show()
```

### 17. Entanglement Correlation Plot

```python
# Plot entanglement correlations
fig = viz.plot_entanglement_correlations(superposition_embeddings[0])
fig.suptitle("Entanglement Correlations")
fig.show()
```

## 🔧 Customizing Quantum Behavior

### 18. Adjust Quantum Parameters

```python
# Create embeddings with different quantum characteristics
quantum_embeddings_high = QuantumEmbeddings(
    vocab_size=1000,
    embedding_dim=128,
    num_states=8,          # More quantum states
    dropout=0.2            # Higher dropout for more uncertainty
)

# Compare uncertainty
uncertainty_high = quantum_embeddings_high.get_uncertainty(input_ids)
print(f"High quantum uncertainty: {uncertainty_high.mean().item():.3f}")
print(f"Standard uncertainty: {uncertainty.mean().item():.3f}")
```

### 19. Custom Collapse Strategy

```python
from qembed.core.collapse_layers import ContextCollapseLayer

# Create custom collapse layer
custom_collapse = ContextCollapseLayer(
    embedding_dim=128,
    num_states=4,
    collapse_method='convolution'  # Use convolution instead of attention
)

# Apply custom collapse
custom_collapsed = custom_collapse(superposition_embeddings, context)
print(f"Custom collapsed shape: {custom_collapsed.shape}")
```

## 📊 Performance Analysis

### 20. Benchmark Your Setup

```python
import time

# Benchmark inference speed
start_time = time.time()
for _ in range(100):
    _ = quantum_embeddings(input_ids, collapse=False)
inference_time = time.time() - start_time

print(f"100 forward passes took: {inference_time:.3f} seconds")
print(f"Average time per forward pass: {inference_time/100:.5f} seconds")

# Memory usage
import sys
model_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Model size: {model_size / 1024 / 1024:.2f} MB")
```

## 🎉 What You've Learned

In this tutorial, you've:

1. ✅ **Created quantum embeddings** with superposition states
2. ✅ **Explored superposition vs collapse** behavior
3. ✅ **Built quantum-enhanced models** (BERT)
4. ✅ **Trained with quantum features** and uncertainty
5. ✅ **Visualized quantum states** and correlations
6. ✅ **Analyzed quantum metrics** and performance
7. ✅ **Customized quantum behavior** for your needs

## 🚀 Next Steps

Now that you're comfortable with the basics:

1. **Explore the Examples** - Check out more complex examples in the `examples/` directory
2. **Read the API Reference** - Deep dive into all available functions and classes
3. **Try Your Own Data** - Apply QEmbed to your specific NLP tasks
4. **Experiment with Parameters** - Tune quantum characteristics for your use case
5. **Join the Community** - Share your findings and get help from others

## 🔗 Additional Resources

- **[API Reference](api_reference.md)** - Complete API documentation
- **[Examples](../examples/)** - More working code examples
- **[Installation Guide](installation.md)** - Detailed setup instructions
- **[GitHub Repository](https://github.com/qembed/qembed)** - Source code and issues

---

**Congratulations!** 🎉 You're now ready to explore the quantum world of natural language processing with QEmbed. The superposition states are your oyster!
