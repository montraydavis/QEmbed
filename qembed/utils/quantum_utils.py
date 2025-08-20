"""
Quantum math utilities for quantum-enhanced embeddings.

This module provides various mathematical utilities for working
with quantum-inspired components and operations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Union
import math


class QuantumUtils:
    """
    Utility class for quantum-inspired operations.
    
    Provides mathematical functions and operations commonly
    used in quantum computing and quantum-inspired models.
    """
    
    @staticmethod
    def create_superposition(
        states: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create superposition of quantum states.
        
        Args:
            states: Tensor of quantum states [batch, seq_len, num_states, dim]
            weights: Optional weights for superposition [batch, seq_len, num_states]
            
        Returns:
            Superposition state [batch, seq_len, dim]
        """
        if weights is None:
            # Equal superposition
            weights = torch.ones(states.shape[:-1], device=states.device) / states.size(-2)
        
        # Normalize weights
        weights = F.softmax(weights, dim=-1)
        
        # Create superposition
        superposition = torch.sum(states * weights.unsqueeze(-1), dim=-2)
        
        return superposition
    
    @staticmethod
    def measure_superposition(
        superposition: torch.Tensor,
        measurement_basis: torch.Tensor,
        noise_level: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Measure superposition state in given basis.
        
        Args:
            superposition: Superposition state to measure
            measurement_basis: Measurement basis vectors
            noise_level: Level of measurement noise
            
        Returns:
            Tuple of (measured_state, measurement_probabilities)
        """
        # Project onto measurement basis
        basis_norm = F.normalize(measurement_basis, p=2, dim=-1)
        superposition_norm = F.normalize(superposition, p=2, dim=-1)
        
        # Compute projection coefficients
        projections = torch.matmul(superposition_norm, basis_norm.transpose(-2, -1))
        
        # Add measurement noise
        if noise_level > 0:
            noise = torch.randn_like(projections) * noise_level
            projections = projections + noise
        
        # Convert to probabilities
        probabilities = F.softmax(projections, dim=-1)
        
        # Create measured state
        measured_state = torch.matmul(probabilities, basis_norm)
        
        return measured_state, probabilities
    
    @staticmethod
    def create_bell_state(
        state1: torch.Tensor,
        state2: torch.Tensor,
        bell_type: str = "phi_plus"
    ) -> torch.Tensor:
        """
        Create Bell state from two quantum states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            bell_type: Type of Bell state ('phi_plus', 'phi_minus', 'psi_plus', 'psi_minus')
            
        Returns:
            Bell state tensor
        """
        # Normalize input states
        state1_norm = F.normalize(state1, p=2, dim=-1)
        state2_norm = F.normalize(state2, p=2, dim=-1)
        
        if bell_type == "phi_plus":
            # |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
            bell_state = (state1_norm + state2_norm) / math.sqrt(2)
        elif bell_type == "phi_minus":
            # |Φ⁻⟩ = (|00⟩ - |11⟩) / √2
            bell_state = (state1_norm - state2_norm) / math.sqrt(2)
        elif bell_type == "psi_plus":
            # |Ψ⁺⟩ = (|01⟩ + |10⟩) / √2
            bell_state = (state1_norm + state2_norm) / math.sqrt(2)
        elif bell_type == "psi_minus":
            # |Ψ⁻⟩ = (|01⟩ - |10⟩) / √2
            bell_state = (state1_norm - state2_norm) / math.sqrt(2)
        else:
            raise ValueError(f"Unknown Bell state type: {bell_type}")
        
        return bell_state
    
    @staticmethod
    def compute_fidelity(
        state1: torch.Tensor,
        state2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fidelity between two quantum states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Fidelity values
        """
        # Normalize states
        state1_norm = F.normalize(state1, p=2, dim=-1)
        state2_norm = F.normalize(state2, p=2, dim=-1)
        
        # Compute overlap
        overlap = torch.abs(torch.sum(state1_norm * state2_norm, dim=-1))
        
        # Fidelity is the square of the overlap
        fidelity = overlap ** 2
        
        return fidelity
    
    @staticmethod
    def compute_entanglement_entropy(
        density_matrix: torch.Tensor,
        partition: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute entanglement entropy of a quantum state.
        
        Args:
            density_matrix: Density matrix representation
            partition: Partition of subsystems for partial trace
            
        Returns:
            Entanglement entropy values
        """
        if density_matrix.dim() == 2:
            # Single density matrix
            eigenvalues = torch.linalg.eigvals(density_matrix).real
            # Remove zero eigenvalues
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            # Compute von Neumann entropy
            entropy = -torch.sum(eigenvalues * torch.log(eigenvalues + 1e-10))
            return entropy
        else:
            # Multiple density matrices
            entropies = []
            for i in range(density_matrix.size(0)):
                entropy = QuantumUtils.compute_entanglement_entropy(density_matrix[i])
                entropies.append(entropy)
            return torch.stack(entropies)
    
    @staticmethod
    def create_ghz_state(
        states: List[torch.Tensor],
        ghz_type: str = "plus"
    ) -> torch.Tensor:
        """
        Create GHZ (Greenberger-Horne-Zeilinger) state.
        
        Args:
            states: List of quantum states
            ghz_type: Type of GHZ state ('plus' or 'minus')
            
        Returns:
            GHZ state tensor
        """
        if len(states) < 2:
            raise ValueError("GHZ state requires at least 2 states")
        
        # Normalize all states
        normalized_states = [F.normalize(state, p=2, dim=-1) for state in states]
        
        if ghz_type == "plus":
            # |GHZ⁺⟩ = (|0...0⟩ + |1...1⟩) / √2
            ghz_state = sum(normalized_states) / math.sqrt(len(states))
        elif ghz_type == "minus":
            # |GHZ⁻⟩ = (|0...0⟩ - |1...1⟩) / √2
            ghz_state = (normalized_states[0] - sum(normalized_states[1:])) / math.sqrt(len(states))
        else:
            raise ValueError(f"Unknown GHZ state type: {ghz_type}")
        
        return ghz_state
    
    @staticmethod
    def apply_quantum_gate(
        state: torch.Tensor,
        gate: torch.Tensor,
        target_qubits: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Apply quantum gate to quantum state.
        
        Args:
            state: Quantum state to transform
            gate: Quantum gate matrix
            target_qubits: Target qubit indices (if None, apply to all)
            
        Returns:
            Transformed quantum state
        """
        if target_qubits is None:
            # Apply gate to entire state
            transformed_state = torch.matmul(state, gate)
        else:
            # Apply gate to specific qubits
            # This is a simplified implementation
            transformed_state = state.clone()
            for qubit_idx in target_qubits:
                if qubit_idx < state.size(-1):
                    # Apply gate to specific dimension
                    gate_expanded = gate.unsqueeze(0).expand(state.size(0), -1, -1)
                    transformed_state = torch.matmul(transformed_state, gate_expanded)
        
        return transformed_state
    
    @staticmethod
    def compute_quantum_distance(
        state1: torch.Tensor,
        state2: torch.Tensor,
        distance_type: str = "fidelity"
    ) -> torch.Tensor:
        """
        Compute distance between quantum states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            distance_type: Type of distance metric
            
        Returns:
            Distance values
        """
        if distance_type == "fidelity":
            return 1 - QuantumUtils.compute_fidelity(state1, state2)
        elif distance_type == "trace":
            # Trace distance
            diff = state1 - state2
            trace_distance = torch.sum(torch.abs(diff), dim=-1)
            return trace_distance
        elif distance_type == "euclidean":
            # Euclidean distance
            return torch.norm(state1 - state2, dim=-1)
        elif distance_type == "cosine":
            # Cosine distance
            cos_sim = F.cosine_similarity(state1, state2, dim=-1)
            return 1 - cos_sim
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")
    
    @staticmethod
    def create_quantum_circuit(
        num_qubits: int,
        circuit_depth: int,
        gate_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Create a random quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            circuit_depth: Depth of the circuit
            gate_types: Types of gates to use
            
        Returns:
            Circuit representation as tensor
        """
        if gate_types is None:
            gate_types = ["H", "X", "Y", "Z", "CNOT"]
        
        # Create random circuit
        circuit = []
        for layer in range(circuit_depth):
            layer_gates = []
            for qubit in range(num_qubits):
                gate_type = np.random.choice(gate_types)
                if gate_type == "H":
                    # Hadamard gate
                    gate = torch.tensor([[1, 1], [1, -1]], dtype=torch.float32) / math.sqrt(2)
                elif gate_type == "X":
                    # Pauli-X gate
                    gate = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
                elif gate_type == "Y":
                    # Pauli-Y gate
                    gate = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.float32)
                elif gate_type == "Z":
                    # Pauli-Z gate
                    gate = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)
                else:
                    # Identity gate
                    gate = torch.eye(2, dtype=torch.float32)
                
                layer_gates.append(gate)
            
            # Combine gates in layer
            layer_tensor = torch.stack(layer_gates)
            circuit.append(layer_tensor)
        
        return torch.stack(circuit)
    
    @staticmethod
    def quantum_fourier_transform(
        state: torch.Tensor,
        inverse: bool = False
    ) -> torch.Tensor:
        """
        Apply quantum Fourier transform to quantum state.
        
        Args:
            state: Quantum state to transform
            inverse: Whether to apply inverse transform
            
        Returns:
            Transformed quantum state
        """
        n = state.size(-1)
        
        # Create Fourier transform matrix
        omega = torch.exp(2j * math.pi / n)
        if inverse:
            omega = torch.conj(omega)
        
        # Build transformation matrix
        fourier_matrix = torch.zeros(n, n, dtype=torch.complex64)
        for i in range(n):
            for j in range(n):
                fourier_matrix[i, j] = omega ** (i * j)
        
        if inverse:
            fourier_matrix = fourier_matrix / math.sqrt(n)
        else:
            fourier_matrix = fourier_matrix / math.sqrt(n)
        
        # Apply transformation
        if state.dtype == torch.complex64:
            transformed_state = torch.matmul(state, fourier_matrix)
        else:
            # Convert to complex if needed
            state_complex = state.to(torch.complex64)
            transformed_state = torch.matmul(state_complex, fourier_matrix)
        
        return transformed_state
