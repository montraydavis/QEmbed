"""
Quantum state visualization tools.

This module provides visualization utilities for quantum states,
superposition, entanglement, and other quantum phenomena.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


class QuantumVisualization:
    """
    Visualization tools for quantum-enhanced embeddings.
    
    Provides various plotting and visualization functions
    for analyzing quantum states and their properties.
    """
    
    def __init__(self, style: str = "default"):
        """
        Initialize quantum visualization.
        
        Args:
            style: Plotting style ('default', 'quantum', 'minimal')
        """
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Setup plotting style."""
        if self.style == "quantum":
            plt.style.use('dark_background')
            plt.rcParams['figure.facecolor'] = 'black'
            plt.rcParams['axes.facecolor'] = 'black'
            plt.rcParams['text.color'] = 'white'
            plt.rcParams['axes.labelcolor'] = 'white'
            plt.rcParams['xtick.color'] = 'white'
            plt.rcParams['ytick.color'] = 'white'
        elif self.style == "minimal":
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (8, 6)
            plt.rcParams['font.size'] = 10
    
    def plot_superposition_states(
        self,
        superposition_states: torch.Tensor,
        labels: Optional[List[str]] = None,
        title: str = "Superposition States",
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot superposition states visualization.
        
        Args:
            superposition_states: Tensor of superposition states
            labels: Labels for each state
            title: Plot title
            figsize: Figure size
        """
        # Convert to numpy for plotting
        if isinstance(superposition_states, torch.Tensor):
            states_np = superposition_states.detach().cpu().numpy()
        else:
            states_np = superposition_states
        
        # Get dimensions
        if states_np.ndim == 3:
            # [batch, seq_len, dim]
            batch_size, seq_len, dim = states_np.shape
            states_flat = states_np.reshape(-1, dim)
        elif states_np.ndim == 4:
            # [batch, seq_len, num_states, dim]
            batch_size, seq_len, num_states, dim = states_np.shape
            states_flat = states_np.reshape(-1, num_states, dim)
        else:
            states_flat = states_np
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: State distribution
        if states_flat.ndim == 2:
            # Single state per position
            axes[0, 0].hist(states_flat.flatten(), bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title("State Distribution")
            axes[0, 0].set_xlabel("State Value")
            axes[0, 0].set_ylabel("Frequency")
        else:
            # Multiple states per position
            for i in range(min(3, states_flat.shape[1])):  # Plot first 3 states
                axes[0, 0].hist(states_flat[:, i, :].flatten(), bins=30, alpha=0.6, 
                               label=f"State {i+1}")
            axes[0, 0].set_title("State Distribution")
            axes[0, 0].set_xlabel("State Value")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].legend()
        
        # Plot 2: State variance across positions
        if states_flat.ndim == 3:
            state_variance = np.var(states_flat, axis=1)
            axes[0, 1].plot(state_variance)
            axes[0, 1].set_title("State Variance Across Positions")
            axes[0, 1].set_xlabel("Position")
            axes[0, 1].set_ylabel("Variance")
        
        # Plot 3: State correlation matrix
        if states_flat.ndim == 2:
            correlation_matrix = np.corrcoef(states_flat.T)
            im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_title("State Correlation Matrix")
            axes[1, 0].set_xlabel("Dimension")
            axes[1, 0].set_ylabel("Dimension")
            plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: State magnitude distribution
        if states_flat.ndim == 2:
            magnitudes = np.linalg.norm(states_flat, axis=1)
            axes[1, 1].hist(magnitudes, bins=30, alpha=0.7, color='green')
            axes[1, 1].set_title("State Magnitude Distribution")
            axes[1, 1].set_xlabel("Magnitude")
            axes[1, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        return fig
    
    def plot_entanglement_correlations(
        self,
        entanglement_matrix: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        title: str = "Entanglement Correlations",
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot entanglement correlations visualization.
        
        Args:
            entanglement_matrix: Entanglement correlation matrix
            attention_weights: Optional attention weights for comparison
            title: Plot title
            figsize: Figure size
        """
        # Convert to numpy
        if isinstance(entanglement_matrix, torch.Tensor):
            ent_matrix = entanglement_matrix.detach().cpu().numpy()
        else:
            ent_matrix = entanglement_matrix
        
        if attention_weights is not None and isinstance(attention_weights, torch.Tensor):
            attn_weights = attention_weights.detach().cpu().numpy()
        else:
            attn_weights = attention_weights
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Entanglement matrix heatmap
        im1 = axes[0, 0].imshow(ent_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title("Entanglement Matrix")
        axes[0, 0].set_xlabel("Position")
        axes[0, 0].set_ylabel("Position")
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Entanglement strength distribution
        axes[0, 1].hist(ent_matrix.flatten(), bins=50, alpha=0.7, color='purple')
        axes[0, 1].set_title("Entanglement Strength Distribution")
        axes[0, 1].set_xlabel("Entanglement Strength")
        axes[0, 1].set_ylabel("Frequency")
        
        # Plot 3: Entanglement vs position
        if ent_matrix.ndim == 2:
            avg_entanglement = np.mean(ent_matrix, axis=1)
            axes[0, 2].plot(avg_entanglement)
            axes[0, 2].set_title("Average Entanglement vs Position")
            axes[0, 2].set_xlabel("Position")
            axes[0, 2].set_ylabel("Average Entanglement")
        
        # Plot 4: Attention weights comparison (if available)
        if attn_weights is not None:
            if attn_weights.ndim == 4:  # [batch, heads, seq_len, seq_len]
                avg_attention = np.mean(attn_weights, axis=(0, 1))
                im2 = axes[1, 0].imshow(avg_attention, cmap='plasma', aspect='auto')
                axes[1, 0].set_title("Average Attention Weights")
                axes[1, 0].set_xlabel("Position")
                axes[1, 0].set_ylabel("Position")
                plt.colorbar(im2, ax=axes[1, 0])
            
            # Plot 5: Entanglement vs Attention correlation
            if ent_matrix.ndim == 2 and attn_weights.ndim == 4:
                avg_attn = np.mean(attn_weights, axis=(0, 1))
                correlation = np.corrcoef(ent_matrix.flatten(), avg_attn.flatten())[0, 1]
                axes[1, 1].text(0.5, 0.5, f'Correlation: {correlation:.3f}', 
                               ha='center', va='center', transform=axes[1, 1].transAxes,
                               fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[1, 1].set_title("Entanglement-Attention Correlation")
                axes[1, 1].axis('off')
        
        # Plot 6: Entanglement network (if 2D)
        if ent_matrix.ndim == 2:
            # Create network visualization
            n = ent_matrix.shape[0]
            x = np.arange(n)
            y = np.arange(n)
            X, Y = np.meshgrid(x, y)
            
            # Only show strong correlations
            threshold = np.percentile(ent_matrix, 80)
            mask = ent_matrix > threshold
            
            axes[1, 2].scatter(X[mask], Y[mask], c=ent_matrix[mask], 
                              cmap='viridis', alpha=0.7, s=50)
            axes[1, 2].set_title("Entanglement Network (Strong Correlations)")
            axes[1, 2].set_xlabel("Position")
            axes[1, 2].set_ylabel("Position")
        
        plt.tight_layout()
        return fig
    
    def plot_uncertainty_analysis(
        self,
        uncertainty: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        title: str = "Uncertainty Analysis",
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot uncertainty analysis visualization.
        
        Args:
            uncertainty: Uncertainty estimates
            predictions: Model predictions (optional)
            targets: Ground truth targets (optional)
            title: Plot title
            figsize: Figure size
        """
        # Convert to numpy
        if isinstance(uncertainty, torch.Tensor):
            unc_np = uncertainty.detach().cpu().numpy()
        else:
            unc_np = uncertainty
        
        if predictions is not None and isinstance(predictions, torch.Tensor):
            pred_np = predictions.detach().cpu().numpy()
        else:
            pred_np = predictions
        
        if targets is not None and isinstance(targets, torch.Tensor):
            target_np = targets.detach().cpu().numpy()
        else:
            target_np = targets
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Uncertainty distribution
        axes[0, 0].hist(unc_np.flatten(), bins=50, alpha=0.7, color='red')
        axes[0, 0].set_title("Uncertainty Distribution")
        axes[0, 0].set_xlabel("Uncertainty")
        axes[0, 0].set_ylabel("Frequency")
        
        # Plot 2: Uncertainty vs position
        if unc_np.ndim == 2:
            avg_uncertainty = np.mean(unc_np, axis=0)
            axes[0, 1].plot(avg_uncertainty)
            axes[0, 1].set_title("Average Uncertainty vs Position")
            axes[0, 1].set_xlabel("Position")
            axes[0, 1].set_ylabel("Uncertainty")
        
        # Plot 3: Uncertainty heatmap
        if unc_np.ndim == 2:
            im = axes[0, 2].imshow(unc_np, cmap='Reds', aspect='auto')
            axes[0, 2].set_title("Uncertainty Heatmap")
            axes[0, 2].set_xlabel("Position")
            axes[0, 2].set_ylabel("Sample")
            plt.colorbar(im, ax=axes[0, 2])
        
        # Plot 4: Uncertainty vs prediction confidence
        if pred_np is not None and unc_np.ndim == 2:
            if pred_np.ndim == 3:  # [batch, seq_len, vocab_size]
                confidence = np.max(pred_np, axis=-1)
            else:
                confidence = pred_np
            
            # Flatten for correlation
            unc_flat = unc_np.flatten()
            conf_flat = confidence.flatten()
            
            # Remove invalid values
            valid_mask = ~(np.isnan(unc_flat) | np.isnan(conf_flat))
            if np.sum(valid_mask) > 0:
                correlation = np.corrcoef(unc_flat[valid_mask], conf_flat[valid_mask])[0, 1]
                
                axes[1, 0].scatter(conf_flat[valid_mask], unc_flat[valid_mask], alpha=0.6)
                axes[1, 0].set_title(f"Uncertainty vs Confidence\nCorrelation: {correlation:.3f}")
                axes[1, 0].set_xlabel("Confidence")
                axes[1, 0].set_ylabel("Uncertainty")
        
        # Plot 5: Uncertainty calibration
        if pred_np is not None and target_np is not None and unc_np.ndim == 2:
            # Group by uncertainty levels
            unc_bins = np.linspace(0, 1, 11)
            bin_centers = (unc_bins[:-1] + unc_bins[1:]) / 2
            
            accuracy_by_uncertainty = []
            for i in range(len(unc_bins) - 1):
                mask = (unc_np >= unc_bins[i]) & (unc_np < unc_bins[i + 1])
                if np.sum(mask) > 0:
                    if pred_np.ndim == 3:
                        pred_labels = np.argmax(pred_np, axis=-1)
                    else:
                        pred_labels = pred_np
                    
                    accuracy = np.mean(pred_labels[mask] == target_np[mask])
                    accuracy_by_uncertainty.append(accuracy)
                else:
                    accuracy_by_uncertainty.append(np.nan)
            
            axes[1, 1].plot(bin_centers, accuracy_by_uncertainty, 'o-')
            axes[1, 1].plot([0, 1], [0, 1], '--', color='gray', alpha=0.7)
            axes[1, 1].set_title("Uncertainty Calibration")
            axes[1, 1].set_xlabel("Predicted Uncertainty")
            axes[1, 1].set_ylabel("Actual Accuracy")
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
        
        # Plot 6: Uncertainty statistics
        if unc_np.ndim == 2:
            stats_text = f"""
            Mean: {np.mean(unc_np):.3f}
            Std: {np.std(unc_np):.3f}
            Min: {np.min(unc_np):.3f}
            Max: {np.max(unc_np):.3f}
            Median: {np.median(unc_np):.3f}
            """
            axes[1, 2].text(0.5, 0.5, stats_text, ha='center', va='center',
                           transform=axes[1, 2].transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[1, 2].set_title("Uncertainty Statistics")
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_quantum_circuit(
        self,
        circuit: torch.Tensor,
        gate_names: Optional[List[str]] = None,
        title: str = "Quantum Circuit",
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot quantum circuit visualization.
        
        Args:
            circuit: Circuit tensor representation
            gate_names: Names of quantum gates
            title: Plot title
            figsize: Figure size
        """
        # Convert to numpy
        if isinstance(circuit, torch.Tensor):
            circuit_np = circuit.detach().cpu().numpy()
        else:
            circuit_np = circuit
        
        # Get circuit dimensions
        if circuit_np.ndim == 3:
            # [layers, qubits, gate_dim]
            num_layers, num_qubits, gate_dim = circuit_np.shape
        else:
            raise ValueError("Circuit tensor must be 3-dimensional")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Create circuit grid
        y_positions = np.arange(num_qubits)
        x_positions = np.arange(num_layers)
        
        # Plot gates
        for layer in range(num_layers):
            for qubit in range(num_qubits):
                gate = circuit_np[layer, qubit]
                
                # Determine gate type based on matrix properties
                if np.allclose(gate, np.eye(2)):
                    gate_type = "I"
                    color = "lightgray"
                elif np.allclose(gate, np.array([[0, 1], [1, 0]])):
                    gate_type = "X"
                    color = "red"
                elif np.allclose(gate, np.array([[1, 1], [1, -1]]) / np.sqrt(2)):
                    gate_type = "H"
                    color = "blue"
                else:
                    gate_type = "U"
                    color = "orange"
                
                # Plot gate
                ax.add_patch(plt.Rectangle((layer - 0.4, qubit - 0.4), 0.8, 0.8,
                                         facecolor=color, edgecolor='black', alpha=0.7))
                ax.text(layer, qubit, gate_type, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Customize plot
        ax.set_xlim(-0.5, num_layers - 0.5)
        ax.set_ylim(-0.5, num_qubits - 0.5)
        ax.set_xlabel("Circuit Layer")
        ax.set_ylabel("Qubit")
        ax.set_title("Quantum Circuit Diagram")
        ax.grid(True, alpha=0.3)
        
        # Set ticks
        ax.set_xticks(x_positions)
        ax.set_yticks(y_positions)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_quantum_state(
        self,
        quantum_state: torch.Tensor,
        title: str = "3D Quantum State",
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot 3D visualization of quantum state.
        
        Args:
            quantum_state: Quantum state tensor
            title: Plot title
            figsize: Figure size
        """
        # Convert to numpy
        if isinstance(quantum_state, torch.Tensor):
            state_np = quantum_state.detach().cpu().numpy()
        else:
            state_np = quantum_state
        
        # Ensure 3D data
        if state_np.ndim == 2:
            # [seq_len, dim] -> add batch dimension
            state_np = state_np[np.newaxis, :, :]
        
        if state_np.ndim != 3:
            raise ValueError("Quantum state must be 2D or 3D")
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot first batch
        batch_idx = 0
        x = np.arange(state_np.shape[1])  # Sequence positions
        y = np.arange(state_np.shape[2])  # Dimensions
        
        X, Y = np.meshgrid(x, y)
        Z = state_np[batch_idx]
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        # Customize plot
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Dimension")
        ax.set_zlabel("State Value")
        ax.set_title(f"{title} (Batch {batch_idx})")
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        return fig
