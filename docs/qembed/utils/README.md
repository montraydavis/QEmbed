# QEmbed Utilities

## Overview

The utilities module provides helper functions, visualization tools, and metrics specifically designed for quantum-inspired models. These utilities help with model evaluation, quantum property analysis, and result visualization.

## Module Structure

### 1. [Metrics](./metrics.md)

**File:** `qembed/utils/metrics.py`

Evaluation metrics and quantum property measurements.

**Key Features:**

- Quantum-specific evaluation metrics
- Uncertainty quantification
- Entanglement strength measurement
- Superposition coherence analysis

**Main Classes:**

- `QuantumMetrics`
- `UncertaintyMetrics`
- `EntanglementMetrics`

### 2. [Quantum Utils](./quantum_utils.md)

**File:** `qembed/utils/quantum_utils.py`

Quantum computing utilities and helper functions.

**Key Features:**

- Quantum state manipulation
- Entanglement calculations
- Measurement operations
- Quantum circuit utilities

**Main Classes:**

- `QuantumUtils`
- `QuantumState`
- `QuantumCircuit`

### 3. [Visualization](./visualization.md)

**File:** `qembed/utils/visualization.py`

Plotting and visualization tools for quantum properties.

**Key Features:**

- Quantum state visualization
- Entanglement plots
- Uncertainty visualization
- Training progress plots

**Main Classes:**

- `QuantumVisualization`
- `TrainingVisualizer`
- `QuantumStatePlotter`

## Quick Start

### Basic Utility Usage

```python
from qembed.utils import QuantumMetrics, QuantumUtils, QuantumVisualization

# Initialize utilities
metrics = QuantumMetrics()
quantum_utils = QuantumUtils()
visualizer = QuantumVisualization()

# Evaluate model performance
model_metrics = metrics.compute_metrics(predictions, labels)
print(f"Accuracy: {model_metrics['accuracy']}")
print(f"Uncertainty: {model_metrics['uncertainty']}")

# Analyze quantum properties
quantum_properties = quantum_utils.analyze_quantum_state(embeddings)
print(f"Superposition coherence: {quantum_properties['coherence']}")
print(f"Entanglement strength: {quantum_properties['entanglement']}")

# Visualize results
visualizer.plot_quantum_states(quantum_properties)
```

### Utility Configuration

```python
# Configure utility parameters
metrics_config = {
    'uncertainty_threshold': 0.5,
    'entanglement_type': 'bell_state',
    'coherence_measure': 'von_neumann'
}

quantum_config = {
    'state_representation': 'density_matrix',
    'entanglement_measure': 'concurrence',
    'measurement_basis': 'computational'
}

visualization_config = {
    'plot_style': 'seaborn',
    'color_scheme': 'quantum',
    'interactive': True
}

# Initialize with configuration
metrics = QuantumMetrics(**metrics_config)
quantum_utils = QuantumUtils(**quantum_config)
visualizer = QuantumVisualization(**visualization_config)
```

## Metrics and Evaluation

### 1. Model Performance Metrics

```python
# Standard NLP metrics
performance_metrics = metrics.compute_performance_metrics(
    predictions=predictions,
    labels=labels,
    task_type='classification'
)

print(f"Accuracy: {performance_metrics['accuracy']}")
print(f"F1 Score: {performance_metrics['f1']}")
print(f"Precision: {performance_metrics['precision']}")
print(f"Recall: {performance_metrics['recall']}")
```

### 2. Quantum Property Metrics

```python
# Quantum-specific metrics
quantum_metrics = metrics.compute_quantum_metrics(
    embeddings=embeddings,
    uncertainty_scores=uncertainty_scores
)

print(f"Superposition coherence: {quantum_metrics['coherence']}")
print(f"Entanglement strength: {quantum_metrics['entanglement']}")
print(f"Measurement uncertainty: {quantum_metrics['uncertainty']}")
print(f"Quantum purity: {quantum_metrics['purity']}")
```

### 3. Uncertainty Quantification

```python
# Uncertainty analysis
uncertainty_metrics = metrics.compute_uncertainty_metrics(
    predictions=predictions,
    uncertainty_scores=uncertainty_scores,
    labels=labels
)

print(f"Calibration error: {uncertainty_metrics['calibration_error']}")
print(f"Reliability: {uncertainty_metrics['reliability']}")
print(f"Confidence correlation: {uncertainty_metrics['confidence_correlation']}")
```

## Quantum Utilities

### 1. Quantum State Analysis

```python
# Analyze quantum states
state_analysis = quantum_utils.analyze_quantum_state(
    embeddings=embeddings,
    analysis_type='comprehensive'
)

print(f"State dimension: {state_analysis['dimension']}")
print(f"State purity: {state_analysis['purity']}")
print(f"State coherence: {state_analysis['coherence']}")
print(f"State entanglement: {state_analysis['entanglement']}")
```

### 2. Entanglement Calculations

```python
# Calculate entanglement measures
entanglement_measures = quantum_utils.compute_entanglement(
    state=quantum_state,
    measure_type='concurrence'
)

print(f"Concurrence: {entanglement_measures['concurrence']}")
print(f"Negativity: {entanglement_measures['negativity']}")
print(f"Entanglement of formation: {entanglement_measures['formation']}")
```

### 3. Measurement Operations

```python
# Perform quantum measurements
measurement_results = quantum_utils.measure_quantum_state(
    state=quantum_state,
    basis='computational',
    noise_level=0.1
)

print(f"Measurement outcome: {measurement_results['outcome']}")
print(f"Measurement probability: {measurement_results['probability']}")
print(f"Post-measurement state: {measurement_results['post_state']}")
```

## Visualization Tools

### 1. Quantum State Visualization

```python
# Visualize quantum states
visualizer.plot_quantum_state(
    state=quantum_state,
    plot_type='bloch_sphere',
    title='Quantum State on Bloch Sphere'
)

# Multi-state visualization
visualizer.plot_multiple_states(
    states=[state1, state2, state3],
    plot_type='density_matrix',
    labels=['State 1', 'State 2', 'State 3']
)
```

### 2. Training Progress Visualization

```python
# Plot training curves
visualizer.plot_training_progress(
    history=training_history,
    metrics=['loss', 'accuracy', 'uncertainty'],
    plot_type='line'
)

# Plot quantum properties during training
visualizer.plot_quantum_properties(
    history=training_history,
    properties=['coherence', 'entanglement', 'purity']
)
```

### 3. Entanglement Visualization

```python
# Visualize entanglement patterns
visualizer.plot_entanglement(
    entanglement_matrix=entanglement_matrix,
    plot_type='heatmap',
    title='Entanglement Matrix'
)

# Plot entanglement over time
visualizer.plot_entanglement_evolution(
    entanglement_history=entanglement_history,
    time_steps=time_steps
)
```

## Advanced Usage

### 1. Custom Metrics

```python
# Create custom quantum metrics
class CustomQuantumMetrics(QuantumMetrics):
    def __init__(self, custom_threshold=0.5):
        super().__init__()
        self.custom_threshold = custom_threshold
    
    def compute_custom_metric(self, embeddings, labels):
        # Implement custom metric
        custom_score = self._calculate_custom_score(embeddings, labels)
        return {'custom_metric': custom_score}

# Use custom metrics
custom_metrics = CustomQuantumMetrics(custom_threshold=0.7)
results = custom_metrics.compute_custom_metric(embeddings, labels)
```

### 2. Batch Processing

```python
# Process multiple models
model_results = []
for model_name, model in models.items():
    # Get predictions and embeddings
    predictions, embeddings, uncertainty = model(input_data)
    
    # Compute metrics
    metrics_result = metrics.compute_all_metrics(
        predictions=predictions,
        labels=labels,
        embeddings=embeddings,
        uncertainty_scores=uncertainty
    )
    
    model_results.append({
        'model': model_name,
        'metrics': metrics_result
    })

# Compare results
comparison = metrics.compare_models(model_results)
visualizer.plot_model_comparison(comparison)
```

### 3. Real-time Monitoring

```python
# Real-time metrics monitoring
class RealTimeMonitor:
    def __init__(self, metrics, visualizer):
        self.metrics = metrics
        self.visualizer = visualizer
        self.history = []
    
    def update(self, predictions, labels, embeddings):
        # Compute current metrics
        current_metrics = self.metrics.compute_metrics(
            predictions, labels, embeddings
        )
        
        # Update history
        self.history.append(current_metrics)
        
        # Real-time visualization
        self.visualizer.update_plots(self.history)

# Use real-time monitor
monitor = RealTimeMonitor(metrics, visualizer)
for batch in dataloader:
    outputs = model(batch)
    monitor.update(outputs.predictions, batch.labels, outputs.embeddings)
```

## Integration with Training

### 1. Training Loop Integration

```python
# Integrate with quantum trainer
from qembed.training import QuantumTrainer

class MetricsAwareTrainer(QuantumTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = QuantumMetrics()
        self.visualizer = QuantumVisualization()
    
    def training_step(self, batch, batch_idx):
        # Standard training step
        loss = super().training_step(batch, batch_idx)
        
        # Compute metrics
        metrics = self.metrics.compute_quantum_metrics(
            embeddings=self.current_embeddings,
            uncertainty_scores=self.current_uncertainty
        )
        
        # Log metrics
        self.log('quantum_coherence', metrics['coherence'])
        self.log('entanglement_strength', metrics['entanglement'])
        
        return loss

# Use metrics-aware trainer
trainer = MetricsAwareTrainer(model, optimizer, loss_fn)
trainer.train(train_loader, val_loader)
```

### 2. Evaluation Integration

```python
# Comprehensive evaluation
def evaluate_model_comprehensive(model, dataloader, metrics, visualizer):
    model.eval()
    all_predictions = []
    all_labels = []
    all_embeddings = []
    all_uncertainty = []
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch)
            all_predictions.append(outputs.predictions)
            all_labels.append(batch.labels)
            all_embeddings.append(outputs.embeddings)
            all_uncertainty.append(outputs.uncertainty)
    
    # Concatenate results
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    embeddings = torch.cat(all_embeddings)
    uncertainty = torch.cat(all_uncertainty)
    
    # Compute comprehensive metrics
    results = metrics.compute_all_metrics(
        predictions=predictions,
        labels=labels,
        embeddings=embeddings,
        uncertainty_scores=uncertainty
    )
    
    # Generate comprehensive visualization
    visualizer.create_comprehensive_report(results)
    
    return results

# Run comprehensive evaluation
results = evaluate_model_comprehensive(
    model=model,
    dataloader=test_loader,
    metrics=metrics,
    visualizer=visualizer
)
```

## Performance Optimization

### 1. Efficient Metric Computation

```python
# Optimize metric computation
optimized_config = {
    'batch_processing': True,
    'parallel_computation': True,
    'memory_efficient': True,
    'cache_results': True
}

metrics = QuantumMetrics(**optimized_config)
```

### 2. Visualization Optimization

```python
# Optimize visualization
visualization_config = {
    'interactive': False,  # Faster static plots
    'dpi': 150,           # Lower resolution for speed
    'cache_plots': True,  # Cache generated plots
    'batch_rendering': True  # Batch render operations
}

visualizer = QuantumVisualization(**visualization_config)
```

### 3. Memory Management

```python
# Memory-efficient processing
memory_config = {
    'gradient_checkpointing': True,
    'mixed_precision': True,
    'chunk_processing': True,
    'memory_pooling': True
}

quantum_utils = QuantumUtils(**memory_config)
```

## Best Practices

### 1. Metric Selection

- **Task Alignment**: Choose metrics relevant to your task
- **Quantum Properties**: Include quantum-specific metrics
- **Performance Balance**: Balance accuracy with interpretability
- **Validation**: Validate metric calculations

### 2. Visualization

- **Clarity**: Ensure plots are clear and informative
- **Consistency**: Use consistent plotting styles
- **Interactivity**: Use interactive plots when appropriate
- **Documentation**: Document plot meanings and interpretations

### 3. Performance

- **Efficiency**: Use efficient computation methods
- **Caching**: Cache frequently computed results
- **Parallelization**: Enable parallel processing when possible
- **Memory**: Monitor and optimize memory usage

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or enable memory optimization
2. **Slow Computation**: Enable parallel processing or caching
3. **Visualization Issues**: Check plotting library compatibility
4. **Metric Errors**: Validate input data and metric calculations

### Debugging Tips

1. **Check Inputs**: Verify input data format and dimensions
2. **Monitor Performance**: Track computation time and memory usage
3. **Validate Results**: Check metric values for reasonableness
4. **Test Components**: Test individual utility components

## Future Extensions

The utility framework is designed for extensibility:

- **New Metrics**: Easy addition of new evaluation metrics
- **Advanced Visualization**: Support for 3D and interactive plots
- **Real-time Analysis**: Support for streaming data analysis
- **Multi-modal Utilities**: Support for text, audio, and visual data
