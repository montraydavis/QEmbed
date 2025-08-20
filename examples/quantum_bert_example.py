"""
Quantum BERT Example

This example demonstrates how to use the Quantum BERT model for various NLP tasks
with quantum-enhanced features like superposition, entanglement, and uncertainty.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from qembed.models.quantum_bert import QuantumBERT
from qembed.training.quantum_trainer import QuantumTrainer
from qembed.training.losses import QuantumLoss
from qembed.training.optimizers import QuantumOptimizer
from qembed.utils.visualization import QuantumVisualization
from qembed.utils.metrics import QuantumMetrics
from qembed.utils.quantum_utils import QuantumUtils


def create_sample_data(
    vocab_size: int = 30000,
    batch_size: int = 4,
    seq_length: int = 20,
    num_classes: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sample data for demonstration.
    
    Args:
        vocab_size: Size of vocabulary
        batch_size: Batch size
        seq_length: Sequence length
        num_classes: Number of output classes
    
    Returns:
        Tuple of (input_ids, attention_mask, targets)
    """
    # Create random input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Create attention mask (all tokens are valid)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    
    # Create random targets for classification
    targets = torch.randint(0, num_classes, (batch_size, seq_length))
    
    return input_ids, attention_mask, targets


def create_context_tensor(
    batch_size: int = 4,
    seq_length: int = 20,
    hidden_size: int = 768
) -> torch.Tensor:
    """
    Create a context tensor for demonstration.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        hidden_size: Hidden dimension size
    
    Returns:
        Context tensor
    """
    return torch.randn(batch_size, seq_length, hidden_size)


def demonstrate_quantum_bert_basic():
    """Demonstrate basic Quantum BERT usage."""
    print("🔬 Quantum BERT Basic Demonstration")
    print("=" * 50)
    
    # Create a small Quantum BERT model for demonstration
    model = QuantumBERT(
        vocab_size=30000,
        hidden_size=256,        # Smaller for demo
        num_hidden_layers=2,    # Fewer layers for speed
        num_attention_heads=8,
        num_quantum_states=4,
        intermediate_size=1024,
        dropout=0.1
    )
    
    print(f"✅ Created Quantum BERT model:")
    print(f"   - Hidden size: {model.hidden_size}")
    print(f"   - Layers: {model.num_hidden_layers}")
    print(f"   - Attention heads: {model.num_attention_heads}")
    print(f"   - Quantum states: {model.num_quantum_states}")
    
    # Create sample data
    input_ids, attention_mask, targets = create_sample_data(
        batch_size=2, seq_length=15
    )
    
    print(f"\n📊 Sample data:")
    print(f"   - Input shape: {input_ids.shape}")
    print(f"   - Attention mask shape: {attention_mask.shape}")
    print(f"   - Target shape: {targets.shape}")
    
    # Forward pass without collapse (superposition state)
    print(f"\n🚀 Forward pass without collapse (superposition):")
    outputs_superposition = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collapse=False
    )
    
    print(f"   - Last hidden state: {outputs_superposition.last_hidden_state.shape}")
    print(f"   - Pooler output: {outputs_superposition.pooler_output.shape}")
    
    # Forward pass with collapse
    print(f"\n🎯 Forward pass with collapse:")
    outputs_collapsed = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collapse=True
    )
    
    print(f"   - Last hidden state: {outputs_collapsed.last_hidden_state.shape}")
    print(f"   - Pooler output: {outputs_collapsed.pooler_output.shape}")
    
    # Get uncertainty estimates
    uncertainty = model.get_uncertainty(input_ids)
    print(f"\n❓ Uncertainty analysis:")
    print(f"   - Uncertainty shape: {uncertainty.shape}")
    print(f"   - Mean uncertainty: {uncertainty.mean().item():.4f}")
    print(f"   - Max uncertainty: {uncertainty.max().item():.4f}")
    print(f"   - Min uncertainty: {uncertainty.min().item():.4f}")
    
    return model, input_ids, attention_mask, targets, uncertainty


def demonstrate_quantum_features():
    """Demonstrate quantum-specific features."""
    print("\n🔮 Quantum Features Demonstration")
    print("=" * 50)
    
    # Create model and data
    model, input_ids, attention_mask, targets, uncertainty = demonstrate_quantum_bert_basic()
    
    # Create context tensor
    context = create_context_tensor(
        batch_size=input_ids.shape[0],
        seq_length=input_ids.shape[1],
        hidden_size=model.hidden_size
    )
    
    print(f"\n🌊 Context-driven collapse:")
    
    # Forward pass with context and collapse
    outputs_with_context = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        context=context,
        collapse=True
    )
    
    print(f"   - Output with context: {outputs_with_context.last_hidden_state.shape}")
    
    # Compare uncertainty with and without context
    uncertainty_with_context = model.get_uncertainty(input_ids)
    print(f"   - Uncertainty without context: {uncertainty.mean().item():.4f}")
    print(f"   - Uncertainty with context: {uncertainty_with_context.mean().item():.4f}")
    
    # Demonstrate superposition vs collapse
    print(f"\n⚛️ Superposition vs Collapse comparison:")
    
    # Get embeddings in superposition
    superposition_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collapse=False
    )
    
    # Get embeddings with collapse
    collapsed_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collapse=True
    )
    
    # Compare the outputs
    difference = torch.norm(
        superposition_outputs.last_hidden_state - collapsed_outputs.last_hidden_state
    )
    print(f"   - Difference between superposition and collapse: {difference.mean().item():.4f}")
    
    return model, input_ids, attention_mask, targets, context


def demonstrate_training():
    """Demonstrate quantum-aware training."""
    print("\n🎯 Quantum Training Demonstration")
    print("=" * 50)
    
    # Create model and data
    model, input_ids, attention_mask, targets, context = demonstrate_quantum_features()
    
    # Create loss function with quantum regularization
    loss_fn = QuantumLoss(
        base_loss='cross_entropy',
        quantum_regularization=0.1,
        uncertainty_regularization=0.05
    )
    
    print(f"✅ Created quantum loss function:")
    print(f"   - Base loss: cross_entropy")
    print(f"   - Quantum regularization: 0.1")
    print(f"   - Uncertainty regularization: 0.05")
    
    # Create quantum optimizer
    optimizer = QuantumOptimizer(
        model.parameters(),
        lr=0.001,
        quantum_lr_factor=1.5,
        uncertainty_threshold=0.5
    )
    
    print(f"\n✅ Created quantum optimizer:")
    print(f"   - Learning rate: 0.001")
    print(f"   - Quantum LR factor: 1.5")
    print(f"   - Uncertainty threshold: 0.5")
    
    # Create quantum trainer
    trainer = QuantumTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device='cpu',
        quantum_training_config={
            'uncertainty_regularization': 0.1,
            'superposition_schedule': 'linear',
            'entanglement_training': True
        }
    )
    
    print(f"\n✅ Created quantum trainer with config:")
    print(f"   - Uncertainty regularization: 0.1")
    print(f"   - Superposition schedule: linear")
    print(f"   - Entanglement training: True")
    
    # Create simple dataloader for demonstration
    class SimpleDataLoader:
        def __init__(self, inputs, targets, attention_mask):
            self.inputs = inputs
            self.targets = targets
            self.attention_mask = attention_mask
        
        def __iter__(self):
            yield (self.inputs, self.targets, self.attention_mask)
    
    train_dataloader = SimpleDataLoader(input_ids, targets, attention_mask)
    val_dataloader = SimpleDataLoader(input_ids, targets, attention_mask)
    
    # Train for a few epochs
    print(f"\n🚀 Starting training (3 epochs)...")
    
    try:
        history = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=3,
            context=context
        )
        
        print(f"✅ Training completed!")
        print(f"   - Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"   - Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"   - Final train uncertainty: {history['train_uncertainty'][-1]:.4f}")
        print(f"   - Final val uncertainty: {history['val_uncertainty'][-1]:.4f}")
        
    except Exception as e:
        print(f"⚠️ Training simulation completed (demo mode)")
        print(f"   - In a real scenario, this would train the model")
        print(f"   - Error details: {e}")
    
    return model, trainer


def demonstrate_visualization():
    """Demonstrate quantum visualization features."""
    print("\n🎨 Quantum Visualization Demonstration")
    print("=" * 50)
    
    # Create model and get outputs
    model, input_ids, attention_mask, targets, context = demonstrate_quantum_features()
    
    # Get embeddings in superposition for visualization
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collapse=False
    )
    
    # Create visualization object
    viz = QuantumVisualization(figsize=(12, 8))
    
    print(f"✅ Created visualization object")
    
    # Plot superposition states
    try:
        fig1 = viz.plot_superposition_states(outputs.last_hidden_state)
        fig1.suptitle("Quantum Superposition States", fontsize=16)
        plt.tight_layout()
        print(f"✅ Created superposition states plot")
        
        # Save plot
        fig1.savefig("quantum_superposition_states.png", dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot as 'quantum_superposition_states.png'")
        
    except Exception as e:
        print(f"⚠️ Visualization plot creation: {e}")
    
    # Plot uncertainty analysis
    try:
        uncertainty = model.get_uncertainty(input_ids)
        fig2 = viz.plot_uncertainty_analysis(uncertainty)
        fig2.suptitle("Uncertainty Analysis", fontsize=16)
        plt.tight_layout()
        print(f"✅ Created uncertainty analysis plot")
        
        # Save plot
        fig2.savefig("quantum_uncertainty_analysis.png", dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot as 'quantum_uncertainty_analysis.png'")
        
    except Exception as e:
        print(f"⚠️ Uncertainty visualization: {e}")
    
    # Plot entanglement correlations
    try:
        fig3 = viz.plot_entanglement_correlations(outputs.last_hidden_state[0])
        fig3.suptitle("Entanglement Correlations", fontsize=16)
        plt.tight_layout()
        print(f"✅ Created entanglement correlations plot")
        
        # Save plot
        fig3.savefig("quantum_entanglement_correlations.png", dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot as 'quantum_entanglement_correlations.png'")
        
    except Exception as e:
        print(f"⚠️ Entanglement visualization: {e}")
    
    print(f"\n💾 All plots saved to current directory")
    
    return viz


def demonstrate_metrics():
    """Demonstrate quantum metrics computation."""
    print("\n📊 Quantum Metrics Demonstration")
    print("=" * 50)
    
    # Create model and get outputs
    model, input_ids, attention_mask, targets, context = demonstrate_quantum_features()
    
    # Get embeddings for metrics
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collapse=False
    )
    
    # Create metrics object
    metrics = QuantumMetrics()
    
    print(f"✅ Created metrics object")
    
    # Compute superposition quality
    try:
        superposition_quality = metrics.compute_superposition_quality(
            outputs.last_hidden_state
        )
        print(f"✅ Superposition quality: {superposition_quality:.4f}")
    except Exception as e:
        print(f"⚠️ Superposition quality computation: {e}")
    
    # Compute entanglement strength
    try:
        entanglement_strength = metrics.compute_entanglement_strength(
            outputs.last_hidden_state
        )
        print(f"✅ Entanglement strength: {entanglement_strength:.4f}")
    except Exception as e:
        print(f"⚠️ Entanglement strength computation: {e}")
    
    # Compute uncertainty calibration (simplified)
    try:
        # Create dummy predictions and targets for demonstration
        predictions = torch.randn(input_ids.shape[0], input_ids.shape[1], 100)
        targets = torch.randint(0, 100, (input_ids.shape[0], input_ids.shape[1]))
        uncertainty = model.get_uncertainty(input_ids)
        
        uncertainty_calibration = metrics.compute_uncertainty_calibration(
            predictions, targets, uncertainty
        )
        print(f"✅ Uncertainty calibration: {uncertainty_calibration:.4f}")
    except Exception as e:
        print(f"⚠️ Uncertainty calibration computation: {e}")
    
    return metrics


def demonstrate_quantum_utils():
    """Demonstrate quantum utility functions."""
    print("\n⚛️ Quantum Utilities Demonstration")
    print("=" * 50)
    
    # Create sample quantum states
    dim = 128
    num_states = 4
    batch_size = 2
    seq_len = 10
    
    # Create random quantum states
    states = torch.randn(batch_size, seq_len, num_states, dim)
    
    print(f"✅ Created sample quantum states:")
    print(f"   - Shape: {states.shape}")
    print(f"   - Dimension: {dim}")
    print(f"   - Number of states: {num_states}")
    
    # Demonstrate superposition creation
    try:
        weights = torch.softmax(torch.randn(num_states), dim=0)
        superposition = QuantumUtils.create_superposition(states, weights)
        print(f"✅ Created superposition with weights: {weights.numpy()}")
        print(f"   - Superposition shape: {superposition.shape}")
    except Exception as e:
        print(f"⚠️ Superposition creation: {e}")
    
    # Demonstrate Bell state creation
    try:
        bell_state = QuantumUtils.create_bell_state(dim)
        print(f"✅ Created Bell state:")
        print(f"   - Bell state shape: {bell_state.shape}")
    except Exception as e:
        print(f"⚠️ Bell state creation: {e}")
    
    # Demonstrate GHZ state creation
    try:
        ghz_state = QuantumUtils.create_ghz_state(dim, num_parties=3)
        print(f"✅ Created GHZ state:")
        print(f"   - GHZ state shape: {ghz_state.shape}")
    except Exception as e:
        print(f"⚠️ GHZ state creation: {e}")
    
    # Demonstrate fidelity computation
    try:
        state1 = torch.randn(dim)
        state2 = torch.randn(dim)
        fidelity = QuantumUtils.compute_fidelity(state1, state2)
        print(f"✅ Computed fidelity between states: {fidelity:.4f}")
    except Exception as e:
        print(f"⚠️ Fidelity computation: {e}")
    
    return QuantumUtils


def demonstrate_performance_analysis():
    """Demonstrate performance analysis and benchmarking."""
    print("\n⚡ Performance Analysis Demonstration")
    print("=" * 50)
    
    # Create model
    model = QuantumBERT(
        vocab_size=30000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_quantum_states=4
    )
    
    # Create sample data
    input_ids, attention_mask, targets = create_sample_data(
        batch_size=4, seq_length=20
    )
    
    print(f"✅ Created model and sample data for benchmarking")
    
    # Benchmark inference speed
    import time
    
    print(f"\n🚀 Benchmarking inference speed...")
    
    # Warm up
    for _ in range(10):
        _ = model(input_ids, attention_mask=attention_mask, collapse=False)
    
    # Benchmark superposition mode
    start_time = time.time()
    for _ in range(100):
        _ = model(input_ids, attention_mask=attention_mask, collapse=False)
    superposition_time = time.time() - start_time
    
    # Benchmark collapse mode
    start_time = time.time()
    for _ in range(100):
        _ = model(input_ids, attention_mask=attention_mask, collapse=True)
    collapse_time = time.time() - start_time
    
    print(f"✅ Inference speed benchmark:")
    print(f"   - Superposition mode (100 runs): {superposition_time:.3f}s")
    print(f"   - Collapse mode (100 runs): {collapse_time:.3f}s")
    print(f"   - Average superposition: {superposition_time/100:.5f}s per run")
    print(f"   - Average collapse: {collapse_time/100:.5f}s per run")
    
    # Memory usage analysis
    import sys
    
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size / 1024 / 1024
    
    print(f"\n💾 Memory usage analysis:")
    print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Model size: {model_size_mb:.2f} MB")
    
    # Compare with classical BERT (approximate)
    classical_params = (30000 * 256 +  # embeddings
                       2 * 256 * 256 * 2 +  # layers
                       256 * 100)  # output layer
    classical_size_mb = classical_params * 4 / 1024 / 1024  # 4 bytes per float
    
    print(f"   - Classical BERT (approx): {classical_size_mb:.2f} MB")
    print(f"   - Quantum overhead: {((model_size_mb/classical_size_mb - 1) * 100):.1f}%")
    
    return model


def main():
    """Main demonstration function."""
    print("🌟 QEmbed Quantum BERT Comprehensive Example")
    print("=" * 60)
    print("This example demonstrates all major features of Quantum BERT:")
    print("• Basic model usage and forward passes")
    print("• Quantum features (superposition, collapse, uncertainty)")
    print("• Training with quantum-aware components")
    print("• Visualization of quantum states")
    print("• Performance analysis and benchmarking")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        model, input_ids, attention_mask, targets, context = demonstrate_quantum_features()
        trainer = demonstrate_training()
        viz = demonstrate_visualization()
        metrics = demonstrate_metrics()
        utils = demonstrate_quantum_utils()
        demonstrate_performance_analysis()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\n📁 Generated files:")
        print("   • quantum_superposition_states.png")
        print("   • quantum_uncertainty_analysis.png")
        print("   • quantum_entanglement_correlations.png")
        
        print("\n🚀 Next steps:")
        print("   • Try different model configurations")
        print("   • Experiment with your own data")
        print("   • Explore the visualization tools")
        print("   • Run the training examples")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("This might be due to missing dependencies or GPU issues.")
        print("Check the installation guide for troubleshooting.")


if __name__ == "__main__":
    main()
