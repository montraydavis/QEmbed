#!/usr/bin/env python3
"""
Basic usage examples for QEmbed.

This script demonstrates how to use the basic quantum embedding
components and create simple quantum-enhanced models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import QEmbed components
from qembed.core.quantum_embeddings import QuantumEmbeddings
from qembed.core.collapse_layers import ContextCollapseLayer
from qembed.core.entanglement import EntanglementCorrelation
from qembed.core.measurement import QuantumMeasurement
from qembed.utils.quantum_utils import QuantumUtils
from qembed.utils.visualization import QuantumVisualization


def basic_quantum_embeddings_example():
    """Demonstrate basic quantum embeddings."""
    print("=== Basic Quantum Embeddings Example ===\n")
    
    # Create quantum embeddings
    vocab_size = 1000
    embedding_dim = 128
    num_states = 4
    
    quantum_embeddings = QuantumEmbeddings(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_states=num_states,
        superposition_strength=0.3
    )
    
    print(f"Created quantum embeddings:")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Number of quantum states: {num_states}")
    print(f"  - Superposition strength: {quantum_embeddings.superposition_strength}")
    
    # Create sample input
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward pass without collapse (superposition state)
    print("\n1. Forward pass in superposition state:")
    superposition_embeddings = quantum_embeddings(input_ids, collapse=False)
    print(f"   Output shape: {superposition_embeddings.shape}")
    
    # Get uncertainty estimates
    uncertainty = quantum_embeddings.get_uncertainty(input_ids)
    print(f"   Uncertainty shape: {uncertainty.shape}")
    print(f"   Average uncertainty: {uncertainty.mean().item():.4f}")
    
    # Forward pass with collapse (measured state)
    print("\n2. Forward pass with collapse:")
    # Create simple context
    context = torch.randn(batch_size, seq_len, embedding_dim)
    collapsed_embeddings = quantum_embeddings(input_ids, context=context, collapse=True)
    print(f"   Output shape: {collapsed_embeddings.shape}")
    
    # Compare superposition vs collapsed
    difference = torch.norm(superposition_embeddings - collapsed_embeddings, dim=-1)
    print(f"   Average difference between superposition and collapsed: {difference.mean().item():.4f}")
    
    return quantum_embeddings, input_ids


def context_collapse_example():
    """Demonstrate context-driven collapse."""
    print("\n=== Context Collapse Example ===\n")
    
    embedding_dim = 128
    context_collapse = ContextCollapseLayer(
        embedding_dim=embedding_dim,
        context_window=5,
        collapse_strategy="attention"
    )
    
    print(f"Created context collapse layer:")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Context window: {context_collapse.context_window}")
    print(f"  - Collapse strategy: {context_collapse.collapse_strategy}")
    
    # Create sample embeddings
    batch_size = 2
    seq_len = 8
    embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    
    print(f"\nInput embeddings shape: {embeddings.shape}")
    
    # Apply context collapse
    collapsed_embeddings = context_collapse(embeddings)
    print(f"Output embeddings shape: {collapsed_embeddings.shape}")
    
    # Compare before and after
    change = torch.norm(embeddings - collapsed_embeddings, dim=-1)
    print(f"Average change per position: {change.mean().item():.4f}")
    
    return context_collapse, embeddings


def entanglement_example():
    """Demonstrate quantum entanglement."""
    print("\n=== Quantum Entanglement Example ===\n")
    
    embedding_dim = 128
    entanglement = EntanglementCorrelation(
        embedding_dim=embedding_dim,
        num_entangled_pairs=4,
        entanglement_strength=0.5
    )
    
    print(f"Created entanglement correlation layer:")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Number of entangled pairs: {entanglement.num_entangled_pairs}")
    print(f"  - Entanglement strength: {entanglement.entanglement_strength}")
    
    # Create sample embeddings
    batch_size = 2
    seq_len = 6
    embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    
    print(f"\nInput embeddings shape: {embeddings.shape}")
    
    # Apply entanglement
    entangled_embeddings = entanglement(embeddings)
    print(f"Output embeddings shape: {entangled_embeddings.shape}")
    
    # Analyze entanglement effects
    change = torch.norm(embeddings - entangled_embeddings, dim=-1)
    print(f"Average entanglement effect per position: {change.mean().item():.4f}")
    
    return entanglement, embeddings


def measurement_example():
    """Demonstrate quantum measurement."""
    print("\n=== Quantum Measurement Example ===\n")
    
    embedding_dim = 128
    measurement = QuantumMeasurement(
        embedding_dim=embedding_dim,
        measurement_basis="bell",
        noise_level=0.1
    )
    
    print(f"Created quantum measurement operator:")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Measurement basis: {measurement.measurement_basis}")
    print(f"  - Noise level: {measurement.noise_level}")
    
    # Create sample embeddings
    batch_size = 2
    seq_len = 5
    embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    
    print(f"\nInput embeddings shape: {embeddings.shape}")
    
    # Perform measurement
    measured_embeddings, measurement_results = measurement(embeddings, collapse_probability=0.8)
    print(f"Measured embeddings shape: {measured_embeddings.shape}")
    print(f"Measurement results shape: {measurement_results.shape}")
    
    # Analyze measurement effects
    change = torch.norm(embeddings - measured_embeddings, dim=-1)
    print(f"Average measurement effect per position: {change.mean().item():.4f}")
    
    return measurement, embeddings


def quantum_utils_example():
    """Demonstrate quantum utilities."""
    print("\n=== Quantum Utilities Example ===\n")
    
    # Create sample quantum states
    batch_size = 3
    seq_len = 4
    num_states = 3
    embedding_dim = 64
    
    states = torch.randn(batch_size, seq_len, num_states, embedding_dim)
    
    print(f"Created sample quantum states:")
    print(f"  - Shape: {states.shape}")
    
    # Create superposition
    superposition = QuantumUtils.create_superposition(states)
    print(f"\nSuperposition shape: {superposition.shape}")
    
    # Compute fidelity between original states and superposition
    fidelity = QuantumUtils.compute_fidelity(states[:, :, 0], superposition)
    print(f"Fidelity between first state and superposition: {fidelity:.4f}")
    
    # Create Bell state
    state1 = torch.randn(embedding_dim)
    state2 = torch.randn(embedding_dim)
    bell_state = QuantumUtils.create_bell_state(state1, state2, bell_type="phi_plus")
    print(f"\nBell state shape: {bell_state.shape}")
    
    return states, superposition, bell_state


def visualization_example():
    """Demonstrate quantum visualization."""
    print("\n=== Quantum Visualization Example ===\n")
    
    # Create visualization object
    viz = QuantumVisualization(style="quantum")
    
    # Create sample data for visualization
    batch_size = 10
    seq_len = 8
    num_states = 4
    embedding_dim = 64
    
    # Sample superposition states
    superposition_states = torch.randn(batch_size, seq_len, num_states, embedding_dim)
    
    print(f"Created sample superposition states for visualization:")
    print(f"  - Shape: {superposition_states.shape}")
    
    # Create superposition visualization
    fig = viz.plot_superposition_states(
        superposition_states,
        title="Sample Superposition States"
    )
    
    print("Generated superposition states visualization")
    print("(Figure object created - would display in interactive environment)")
    
    # Sample entanglement matrix
    entanglement_matrix = torch.randn(seq_len, seq_len)
    # Make it symmetric
    entanglement_matrix = (entanglement_matrix + entanglement_matrix.t()) / 2
    
    print(f"\nCreated sample entanglement matrix:")
    print(f"  - Shape: {entanglement_matrix.shape}")
    
    # Create entanglement visualization
    fig2 = viz.plot_entanglement_correlations(
        entanglement_matrix,
        title="Sample Entanglement Correlations"
    )
    
    print("Generated entanglement correlations visualization")
    print("(Figure object created - would display in interactive environment)")
    
    return viz, superposition_states, entanglement_matrix


def main():
    """Run all examples."""
    print("QEmbed Basic Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        quantum_embeddings, input_ids = basic_quantum_embeddings_example()
        
        context_collapse, embeddings = context_collapse_example()
        
        entanglement, ent_embeddings = entanglement_example()
        
        measurement, meas_embeddings = measurement_example()
        
        states, superposition, bell_state = quantum_utils_example()
        
        viz, viz_states, viz_ent = visualization_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nSummary of created objects:")
        print(f"  - QuantumEmbeddings: {quantum_embeddings}")
        print(f"  - ContextCollapseLayer: {context_collapse}")
        print(f"  - EntanglementCorrelation: {entanglement}")
        print(f"  - QuantumMeasurement: {measurement}")
        print(f"  - QuantumVisualization: {viz}")
        
        print("\nYou can now use these objects for further experimentation!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
