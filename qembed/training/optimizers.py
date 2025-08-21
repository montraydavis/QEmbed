"""
Quantum-aware optimizers for training quantum-enhanced models.

This module implements optimizers that are specifically designed
for models with quantum-inspired components.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple
import math
from torch.optim import Optimizer


class QuantumOptimizer(optim.Optimizer):
    """
    Quantum-aware optimizer base class.
    
    Extends standard optimizers with quantum-specific features
    like superposition-aware learning rates and entanglement
    correlation updates.
    """
    
    def __init__(
        self,
        params,
        base_optimizer: str = "adam",
        base_lr: float = 1e-4,
        quantum_lr_multiplier: float = 1.0,
        superposition_schedule: str = "linear",
        entanglement_update_freq: int = 10
    ):
        """
        Initialize quantum optimizer.
        
        Args:
            params: Model parameters
            base_optimizer: Base optimizer type ('adam', 'sgd', 'adamw')
            base_lr: Base learning rate
            quantum_lr_multiplier: Multiplier for quantum parameters
            superposition_schedule: Schedule for superposition training
            entanglement_update_freq: Frequency of entanglement updates
        """
        self.base_lr = base_lr
        self.quantum_lr_multiplier = quantum_lr_multiplier
        self.superposition_schedule = superposition_schedule
        self.entanglement_update_freq = entanglement_update_freq
        self.step_count = 0
        
        # self.base_optimizer: Optimizer | None = None

        # Create base optimizer
        if base_optimizer == "adam":
            self.base_optimizer = optim.Adam(params, lr=base_lr)
        elif base_optimizer == "sgd":
            self.base_optimizer = optim.SGD(params, lr=base_lr)
        elif base_optimizer == "adamw":
            self.base_optimizer = optim.AdamW(params, lr=base_lr)
        else:
            raise ValueError(f"Unsupported base optimizer: {base_optimizer}")
        
        # Separate parameter groups for quantum components
        self.quantum_params = []
        self.classical_params = []
        self._categorize_parameters(params)
        
        # Initialize quantum-specific state
        self.quantum_state = self._init_quantum_state()
        
        super().__init__(params, {})
    
    def _categorize_parameters(self, params):
        """Categorize parameters as quantum or classical."""
        for param_group in params:
            for param in param_group['params']:
                if self._is_quantum_parameter(param):
                    self.quantum_params.append(param)
                else:
                    self.classical_params.append(param)
    
    def _is_quantum_parameter(self, param) -> bool:
        """Determine if a parameter is quantum-related."""
        # Try to get parameter name from various sources
        param_name = None
        
        # Check if parameter has a name attribute
        if hasattr(param, 'name') and param.name is not None:
            param_name = param.name
        # Check if parameter has a _name attribute (PyTorch internal)
        elif hasattr(param, '_name') and param._name is not None:
            param_name = param._name
        # Fall back to string representation
        else:
            param_name = str(param)
        
        # If we still don't have a name, use a default
        if param_name is None or param_name == '':
            param_name = 'unknown_param'
        
        quantum_keywords = ['quantum', 'superposition', 'entanglement', 'measurement']
        return any(keyword in param_name.lower() for keyword in quantum_keywords)
    
    def _init_quantum_state(self) -> Dict[str, Any]:
        """Initialize quantum-specific optimizer state."""
        return {
            'superposition_phase': 0.0,
            'entanglement_strength': 1.0,
            'measurement_noise': 0.0,
            'collapse_probability': 0.0
        }
    
    def step(self, closure=None):
        """Perform optimization step."""
        # Update quantum state
        self._update_quantum_state()
        
        # Apply quantum-specific learning rate adjustments
        self._adjust_learning_rates()
        
        # Perform base optimizer step
        loss = None
        if closure is not None:
            loss = closure()
        
        self.base_optimizer.step()
        self.step_count += 1
        
        return loss
    
    def _update_quantum_state(self):
        """Update quantum optimizer state."""
        # Update superposition phase
        if self.superposition_schedule == "linear":
            self.quantum_state['superposition_phase'] = min(1.0, self.step_count / 1000)
        elif self.superposition_schedule == "cyclic":
            self.quantum_state['superposition_phase'] = 0.5 + 0.5 * math.sin(self.step_count * 0.01)
        elif self.superposition_schedule == "exponential":
            self.quantum_state['superposition_phase'] = 1 - math.exp(-self.step_count / 500)
        
        # Update entanglement strength
        if self.step_count % self.entanglement_update_freq == 0:
            self.quantum_state['entanglement_strength'] = 0.5 + 0.5 * math.cos(self.step_count * 0.001)
        
        # Update measurement noise
        self.quantum_state['measurement_noise'] = max(0.0, 0.1 - self.step_count * 1e-5)
        
        # Update collapse probability
        self.quantum_state['collapse_probability'] = self.quantum_state['superposition_phase']
    
    def _adjust_learning_rates(self):
        """Adjust learning rates for quantum parameters."""
        for param_group in self.base_optimizer.param_groups:
            for param in param_group['params']:
                if param in self.quantum_params:
                    # Apply quantum-specific learning rate adjustments
                    quantum_lr = self.base_lr * self.quantum_lr_multiplier
                    
                    # Adjust based on superposition phase
                    phase_factor = 1.0 + 0.5 * math.sin(self.quantum_state['superposition_phase'] * math.pi)
                    adjusted_lr = quantum_lr * phase_factor
                    
                    param_group['lr'] = adjusted_lr
    
    def get_quantum_state(self) -> Dict[str, Any]:
        """Get current quantum optimizer state."""
        return self.quantum_state.copy()
    
    def set_quantum_state(self, state: Dict[str, Any]):
        """Set quantum optimizer state."""
        self.quantum_state.update(state)


class SuperpositionOptimizer(optim.Optimizer):
    """
    Optimizer specifically designed for training superposition states.
    
    Handles the unique challenges of training models that maintain
    quantum superposition during training.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        superposition_weight: float = 0.1,
        collapse_weight: float = 0.05,
        phase_schedule: str = "linear"
    ):
        """
        Initialize superposition optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            superposition_weight: Weight for superposition regularization
            collapse_weight: Weight for collapse regularization
            phase_schedule: Schedule for superposition phase
        """
        self.lr = lr
        self.superposition_weight = superposition_weight
        self.collapse_weight = collapse_weight
        self.phase_schedule = phase_schedule
        self.step_count = 0
        
        # Parameter groups
        self.superposition_params = []
        self.collapse_params = []
        self._categorize_superposition_parameters(params)
        
        # Initialize state
        self.state = self._init_state()
        
        super().__init__(params, {})
    
    def _categorize_superposition_parameters(self, params):
        """Categorize parameters for superposition training."""
        for param_group in params:
            for param in param_group['params']:
                if self._is_superposition_parameter(param):
                    self.superposition_params.append(param)
                elif self._is_collapse_parameter(param):
                    self.collapse_params.append(param)
    
    def _is_superposition_parameter(self, param) -> bool:
        """Determine if parameter is related to superposition."""
        param_name = str(param)
        return 'superposition' in param_name.lower() or 'state' in param_name.lower()
    
    def _is_collapse_parameter(self, param) -> bool:
        """Determine if parameter is related to collapse."""
        param_name = str(param)
        return 'collapse' in param_name.lower() or 'measurement' in param_name.lower()
    
    def _init_state(self) -> Dict[str, Any]:
        """Initialize optimizer state."""
        return {
            'superposition_phase': 0.0,
            'collapse_phase': 0.0,
            'superposition_momentum': {},
            'collapse_momentum': {}
        }
    
    def step(self, closure=None):
        """Perform optimization step."""
        # Update phases
        self._update_phases()
        
        # Update parameters
        self._update_superposition_parameters()
        self._update_collapse_parameters()
        
        self.step_count += 1
        
        loss = None
        if closure is not None:
            loss = closure()
        
        return loss
    
    def _update_phases(self):
        """Update superposition and collapse phases."""
        if self.phase_schedule == "linear":
            self.state['superposition_phase'] = max(0.0, 1.0 - self.step_count / 1000)
            self.state['collapse_phase'] = min(1.0, self.step_count / 1000)
        elif self.phase_schedule == "cyclic":
            phase = self.step_count * 0.01
            self.state['superposition_phase'] = 0.5 + 0.5 * math.cos(phase)
            self.state['collapse_phase'] = 0.5 - 0.5 * math.cos(phase)
    
    def _update_superposition_parameters(self):
        """Update superposition-related parameters."""
        for param in self.superposition_params:
            if param.grad is None:
                continue
            
            # Initialize momentum
            if param not in self.state['superposition_momentum']:
                self.state['superposition_momentum'][param] = torch.zeros_like(param)
            
            momentum = self.state['superposition_momentum'][param]
            
            # Update with superposition-aware learning rate
            phase_factor = self.state['superposition_phase']
            lr = self.lr * (1.0 + phase_factor)
            
            # Update parameter
            param.data.add_(momentum, alpha=-lr)
            
            # Update momentum
            momentum.mul_(0.9).add_(param.grad, alpha=0.1)
    
    def _update_collapse_parameters(self):
        """Update collapse-related parameters."""
        for param in self.collapse_params:
            if param.grad is None:
                continue
            
            # Initialize momentum
            if param not in self.state['collapse_momentum']:
                self.state['collapse_momentum'][param] = torch.zeros_like(param)
            
            momentum = self.state['collapse_momentum'][param]
            
            # Update with collapse-aware learning rate
            phase_factor = self.state['collapse_phase']
            lr = self.lr * (1.0 + phase_factor)
            
            # Update parameter
            param.data.add_(momentum, alpha=-lr)
            
            # Update momentum
            momentum.mul_(0.9).add_(param.grad, alpha=0.1)


class EntanglementOptimizer(optim.Optimizer):
    """
    Optimizer for training entanglement correlations.
    
    Handles the training of parameters that control quantum
    entanglement between different model components.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        entanglement_strength: float = 1.0,
        correlation_weight: float = 0.1,
        update_frequency: int = 5
    ):
        """
        Initialize entanglement optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            entanglement_strength: Strength of entanglement updates
            correlation_weight: Weight for correlation updates
            update_frequency: Frequency of entanglement updates
        """
        self.lr = lr
        self.entanglement_strength = entanglement_strength
        self.correlation_weight = correlation_weight
        self.update_frequency = update_frequency
        self.step_count = 0
        
        # Parameter groups
        self.entanglement_params = []
        self._categorize_entanglement_parameters(params)
        
        # Initialize state
        self.state = self._init_state()
        
        super().__init__(params, {})
    
    def _categorize_entanglement_parameters(self, params):
        """Categorize parameters for entanglement training."""
        for param_group in params:
            for param in param_group['params']:
                if self._is_entanglement_parameter(param):
                    self.entanglement_params.append(param)
    
    def _is_entanglement_parameter(self, param) -> bool:
        """Determine if parameter is related to entanglement."""
        param_name = str(param)
        return 'entanglement' in param_name.lower() or 'correlation' in param_name.lower()
    
    def _init_state(self) -> Dict[str, Any]:
        """Initialize optimizer state."""
        return {
            'entanglement_momentum': {},
            'correlation_matrix': None,
            'last_update_step': 0
        }
    
    def step(self, closure=None):
        """Perform optimization step."""
        # Update entanglement parameters
        if self.step_count % self.update_frequency == 0:
            self._update_entanglement_parameters()
            self.state['last_update_step'] = self.step_count
        
        self.step_count += 1
        
        loss = None
        if closure is not None:
            loss = closure()
        
        return loss
    
    def _update_entanglement_parameters(self):
        """Update entanglement-related parameters."""
        for param in self.entanglement_params:
            if param.grad is None:
                continue
            
            # Initialize momentum
            if param not in self.state['entanglement_momentum']:
                self.state['entanglement_momentum'][param] = torch.zeros_like(param)
            
            momentum = self.state['entanglement_momentum'][param]
            
            # Compute entanglement-aware gradient
            entanglement_grad = self._compute_entanglement_gradient(param)
            
            # Update parameter
            param.data.add_(momentum, alpha=-self.lr * self.entanglement_strength)
            
            # Update momentum
            momentum.mul_(0.9).add_(entanglement_grad, alpha=0.1)
    
    def _compute_entanglement_gradient(self, param) -> torch.Tensor:
        """Compute gradient with entanglement considerations."""
        # Combine original gradient with entanglement regularization
        original_grad = param.grad
        
        # Add entanglement regularization gradient
        if param.dim() >= 2:
            # Encourage non-trivial entanglement patterns
            identity = torch.eye(param.size(-1), device=param.device)
            if param.dim() > 2:
                identity = identity.unsqueeze(0).expand_as(param)
            
            entanglement_reg_grad = 2 * (param - identity)
            combined_grad = original_grad + self.correlation_weight * entanglement_reg_grad
        else:
            combined_grad = original_grad
        
        return combined_grad
    
    def get_entanglement_state(self) -> Dict[str, Any]:
        """Get current entanglement optimizer state."""
        return {
            'step_count': self.step_count,
            'last_update_step': self.state['last_update_step'],
            'entanglement_strength': self.entanglement_strength
        }
