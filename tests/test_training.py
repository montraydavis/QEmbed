"""
Tests for training utilities.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from qembed.training.quantum_trainer import QuantumTrainer
from qembed.training.losses import (
    QuantumLoss, 
    SuperpositionLoss, 
    EntanglementLoss, 
    UncertaintyLoss
)
from qembed.training.optimizers import (
    QuantumOptimizer, 
    SuperpositionOptimizer, 
    EntanglementOptimizer
)


class TestQuantumLoss:
    """Test cases for QuantumLoss class."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create a QuantumLoss instance for testing."""
        return QuantumLoss(
            base_loss='cross_entropy',
            quantum_regularization=0.1,
            uncertainty_regularization=0.05
        )
    
    @pytest.fixture
    def predictions(self):
        """Create sample predictions."""
        return torch.randn(2, 10, 100)
    
    @pytest.fixture
    def targets(self):
        """Create sample targets."""
        return torch.randint(0, 100, (2, 10))
    
    @pytest.fixture
    def uncertainty(self):
        """Create sample uncertainty."""
        return torch.rand(2, 10)
    
    def test_initialization(self, loss_fn):
        """Test proper initialization."""
        assert loss_fn.base_loss == 'cross_entropy'
        assert loss_fn.quantum_regularization == 0.1
        assert loss_fn.uncertainty_regularization == 0.05
        
        # Check that base loss function exists
        assert hasattr(loss_fn, 'base_loss_fn')
    
    def test_forward(self, loss_fn, predictions, targets, uncertainty):
        """Test forward pass."""
        loss = loss_fn(
            predictions=predictions,
            targets=targets,
            uncertainty=uncertainty
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss > 0  # Loss should be positive
    
    def test_forward_without_uncertainty(self, loss_fn, predictions, targets):
        """Test forward pass without uncertainty."""
        loss = loss_fn(
            predictions=predictions,
            targets=targets
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss > 0
    
    def test_different_base_losses(self):
        """Test initialization with different base losses."""
        for base_loss in ['cross_entropy', 'mse', 'mae']:
            loss_fn = QuantumLoss(
                base_loss=base_loss,
                quantum_regularization=0.1
            )
            assert loss_fn.base_loss == base_loss
            assert hasattr(loss_fn, 'base_loss_fn')
    
    def test_zero_regularization(self):
        """Test with zero regularization."""
        loss_fn = QuantumLoss(
            base_loss='cross_entropy',
            quantum_regularization=0.0,
            uncertainty_regularization=0.0
        )
        
        predictions = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))
        
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss > 0


class TestSuperpositionLoss:
    """Test cases for SuperpositionLoss class."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create a SuperpositionLoss instance for testing."""
        return SuperpositionLoss(
            base_loss='cross_entropy',
            superposition_regularization=0.1,
            collapse_regularization=0.05
        )
    
    @pytest.fixture
    def predictions(self):
        """Create sample predictions."""
        return torch.randn(2, 10, 100)
    
    @pytest.fixture
    def targets(self):
        """Create sample targets."""
        return torch.randint(0, 100, (2, 10))
    
    @pytest.fixture
    def superposition_states(self):
        """Create sample superposition states."""
        return torch.randn(2, 10, 4, 100)  # 4 quantum states
    
    def test_initialization(self, loss_fn):
        """Test proper initialization."""
        assert loss_fn.base_loss == 'cross_entropy'
        assert loss_fn.superposition_regularization == 0.1
        assert loss_fn.collapse_regularization == 0.05
        
        # Check that base loss function exists
        assert hasattr(loss_fn, 'base_loss_fn')
    
    def test_forward(self, loss_fn, predictions, targets, superposition_states):
        """Test forward pass."""
        loss = loss_fn(
            predictions=predictions,
            targets=targets,
            superposition_states=superposition_states
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss > 0
    
    def test_forward_without_superposition(self, loss_fn, predictions, targets):
        """Test forward pass without superposition states."""
        loss = loss_fn(
            predictions=predictions,
            targets=targets
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss > 0
    
    def test_superposition_quality(self, loss_fn, predictions, targets, superposition_states):
        """Test superposition quality regularization."""
        loss = loss_fn(
            predictions=predictions,
            targets=targets,
            superposition_states=superposition_states
        )
        
        # Loss should be higher with superposition regularization
        base_loss = loss_fn.base_loss_fn(predictions, targets)
        assert loss >= base_loss


class TestEntanglementLoss:
    """Test cases for EntanglementLoss class."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create an EntanglementLoss instance for testing."""
        return EntanglementLoss(
            base_loss='cross_entropy',
            entanglement_regularization=0.1,
            correlation_regularization=0.05
        )
    
    @pytest.fixture
    def predictions(self):
        """Create sample predictions."""
        return torch.randn(2, 10, 100)
    
    @pytest.fixture
    def targets(self):
        """Create sample targets."""
        return torch.randint(0, 100, (2, 10))
    
    @pytest.fixture
    def entanglement_matrix(self):
        """Create sample entanglement matrix."""
        return torch.randn(2, 10, 10)  # Correlation matrix
    
    def test_initialization(self, loss_fn):
        """Test proper initialization."""
        assert loss_fn.base_loss == 'cross_entropy'
        assert loss_fn.entanglement_regularization == 0.1
        assert loss_fn.correlation_regularization == 0.05
        
        # Check that base loss function exists
        assert hasattr(loss_fn, 'base_loss_fn')
    
    def test_forward(self, loss_fn, predictions, targets, entanglement_matrix):
        """Test forward pass."""
        loss = loss_fn(
            predictions=predictions,
            targets=targets,
            entanglement_matrix=entanglement_matrix
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss > 0
    
    def test_forward_without_entanglement(self, loss_fn, predictions, targets):
        """Test forward pass without entanglement matrix."""
        loss = loss_fn(
            predictions=predictions,
            targets=targets
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss > 0


class TestUncertaintyLoss:
    """Test cases for UncertaintyLoss class."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create an UncertaintyLoss instance for testing."""
        return UncertaintyLoss(
            base_loss='cross_entropy',
            uncertainty_regularization=0.1,
            calibration_regularization=0.05
        )
    
    @pytest.fixture
    def predictions(self):
        """Create sample predictions."""
        return torch.randn(2, 10, 100)
    
    @pytest.fixture
    def targets(self):
        """Create sample targets."""
        return torch.randint(0, 100, (2, 10))
    
    @pytest.fixture
    def uncertainty(self):
        """Create sample uncertainty."""
        return torch.rand(2, 10)
    
    def test_initialization(self, loss_fn):
        """Test proper initialization."""
        assert loss_fn.base_loss == 'cross_entropy'
        assert loss_fn.uncertainty_regularization == 0.1
        assert loss_fn.calibration_regularization == 0.05
        
        # Check that base loss function exists
        assert hasattr(loss_fn, 'base_loss_fn')
    
    def test_forward(self, loss_fn, predictions, targets, uncertainty):
        """Test forward pass."""
        loss = loss_fn(
            predictions=predictions,
            targets=targets,
            uncertainty=uncertainty
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss > 0
    
    def test_forward_without_uncertainty(self, loss_fn, predictions, targets):
        """Test forward pass without uncertainty."""
        loss = loss_fn(
            predictions=predictions,
            targets=targets
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss > 0
    
    def test_uncertainty_calibration(self, loss_fn, predictions, targets, uncertainty):
        """Test uncertainty calibration regularization."""
        loss = loss_fn(
            predictions=predictions,
            targets=targets,
            uncertainty=uncertainty
        )
        
        # Loss should be higher with uncertainty regularization
        base_loss = loss_fn.base_loss_fn(predictions, targets)
        assert loss >= base_loss


class TestQuantumOptimizer:
    """Test cases for QuantumOptimizer class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a QuantumOptimizer instance for testing."""
        model = Mock()
        model.parameters.return_value = [torch.randn(10, 10)]
        
        return QuantumOptimizer(
            model.parameters(),
            lr=0.001,
            quantum_lr_factor=1.5,
            uncertainty_threshold=0.5
        )
    
    def test_initialization(self, optimizer):
        """Test proper initialization."""
        assert optimizer.quantum_lr_factor == 1.5
        assert optimizer.uncertainty_threshold == 0.5
        
        # Check that base optimizer exists
        assert hasattr(optimizer, 'optimizer')
    
    def test_step(self, optimizer):
        """Test optimizer step."""
        # Mock loss
        loss = Mock()
        loss.backward.return_value = None
        
        # This should not raise an error
        try:
            optimizer.step()
            optimizer.zero_grad()
        except Exception as e:
            pytest.fail(f"Optimizer step failed: {e}")
    
    def test_quantum_lr_adjustment(self):
        """Test quantum learning rate adjustment."""
        model = Mock()
        model.parameters.return_value = [torch.randn(10, 10)]
        
        optimizer = QuantumOptimizer(
            model.parameters(),
            lr=0.001,
            quantum_lr_factor=2.0,
            uncertainty_threshold=0.5
        )
        
        # Check that quantum learning rate is adjusted
        assert optimizer.quantum_lr_factor == 2.0


class TestSuperpositionOptimizer:
    """Test cases for SuperpositionOptimizer class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a SuperpositionOptimizer instance for testing."""
        model = Mock()
        model.parameters.return_value = [torch.randn(10, 10)]
        
        return SuperpositionOptimizer(
            model.parameters(),
            lr=0.001,
            superposition_schedule='linear',
            collapse_threshold=0.3
        )
    
    def test_initialization(self, optimizer):
        """Test proper initialization."""
        assert optimizer.superposition_schedule == 'linear'
        assert optimizer.collapse_threshold == 0.3
        
        # Check that base optimizer exists
        assert hasattr(optimizer, 'optimizer')
    
    def test_step(self, optimizer):
        """Test optimizer step."""
        # Mock loss
        loss = Mock()
        loss.backward.return_value = None
        
        # This should not raise an error
        try:
            optimizer.step()
            optimizer.zero_grad()
        except Exception as e:
            pytest.fail(f"Optimizer step failed: {e}")
    
    def test_different_schedules(self):
        """Test different superposition schedules."""
        model = Mock()
        model.parameters.return_value = [torch.randn(10, 10)]
        
        for schedule in ['linear', 'exponential', 'cosine']:
            optimizer = SuperpositionOptimizer(
                model.parameters(),
                lr=0.001,
                superposition_schedule=schedule
            )
            assert optimizer.superposition_schedule == schedule


class TestEntanglementOptimizer:
    """Test cases for EntanglementOptimizer class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create an EntanglementOptimizer instance for testing."""
        model = Mock()
        model.parameters.return_value = [torch.randn(10, 10)]
        
        return EntanglementOptimizer(
            model.parameters(),
            lr=0.001,
            entanglement_strength=0.8,
            correlation_threshold=0.6
        )
    
    def test_initialization(self, optimizer):
        """Test proper initialization."""
        assert optimizer.entanglement_strength == 0.8
        assert optimizer.correlation_threshold == 0.6
        
        # Check that base optimizer exists
        assert hasattr(optimizer, 'optimizer')
    
    def test_step(self, optimizer):
        """Test optimizer step."""
        # Mock loss
        loss = Mock()
        loss.backward.return_value = None
        
        # This should not raise an error
        try:
            optimizer.step()
            optimizer.zero_grad()
        except Exception as e:
            pytest.fail(f"Optimizer step failed: {e}")


class TestQuantumTrainer:
    """Test cases for QuantumTrainer class."""
    
    @pytest.fixture
    def model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.parameters.return_value = [torch.randn(10, 10)]
        model.train.return_value = None
        model.eval.return_value = None
        model.forward.return_value = torch.randn(2, 10, 100)
        model.get_uncertainty.return_value = torch.rand(2, 10)
        return model
    
    @pytest.fixture
    def optimizer(self):
        """Create a mock optimizer for testing."""
        optimizer = Mock()
        optimizer.zero_grad.return_value = None
        optimizer.step.return_value = None
        return optimizer
    
    @pytest.fixture
    def loss_fn(self):
        """Create a mock loss function for testing."""
        loss_fn = Mock()
        loss_fn.return_value = torch.tensor(1.0)
        return loss_fn
    
    @pytest.fixture
    def trainer(self, model, optimizer, loss_fn):
        """Create a QuantumTrainer instance for testing."""
        return QuantumTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device='cpu'
        )
    
    def test_initialization(self, trainer, model, optimizer, loss_fn):
        """Test proper initialization."""
        assert trainer.model == model
        assert trainer.optimizer == optimizer
        assert trainer.loss_fn == loss_fn
        assert trainer.device == 'cpu'
        assert trainer.quantum_training_config == {}
    
    def test_initialization_with_config(self, model, optimizer, loss_fn):
        """Test initialization with quantum training config."""
        config = {
            'uncertainty_regularization': 0.1,
            'superposition_schedule': 'linear',
            'entanglement_training': True
        }
        
        trainer = QuantumTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device='cpu',
            quantum_training_config=config
        )
        
        assert trainer.quantum_training_config == config
    
    def test_train_epoch(self, trainer):
        """Test training epoch."""
        # Mock dataloader
        dataloader = Mock()
        dataloader.__iter__.return_value = [
            (torch.randn(2, 10), torch.randint(0, 100, (2, 10)))
        ]
        
        # Mock context
        context = torch.randn(2, 10, 100)
        
        # Train epoch
        metrics = trainer.train_epoch(
            dataloader=dataloader,
            context=context
        )
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'uncertainty' in metrics
    
    def test_validate(self, trainer):
        """Test validation."""
        # Mock dataloader
        dataloader = Mock()
        dataloader.__iter__.return_value = [
            (torch.randn(2, 10), torch.randint(0, 100, (2, 10)))
        ]
        
        # Mock context
        context = torch.randn(2, 10, 100)
        
        # Validate
        metrics = trainer.validate(
            dataloader=dataloader,
            context=context
        )
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'uncertainty' in metrics
    
    def test_train(self, trainer):
        """Test full training loop."""
        # Mock dataloaders
        train_dataloader = Mock()
        train_dataloader.__iter__.return_value = [
            (torch.randn(2, 10), torch.randint(0, 100, (2, 10)))
        ]
        
        val_dataloader = Mock()
        val_dataloader.__iter__.return_value = [
            (torch.randn(2, 10), torch.randint(0, 100, (2, 10)))
        ]
        
        # Mock context
        context = torch.randn(2, 10, 100)
        
        # Train
        history = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=2,
            context=context
        )
        
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'train_uncertainty' in history
        assert 'val_uncertainty' in history
    
    def test_save_and_load(self, trainer):
        """Test model saving and loading."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
        
        try:
            # Save model
            trainer.save_checkpoint(checkpoint_path, epoch=1, metrics={'loss': 1.0})
            
            # Check that file exists
            assert os.path.exists(checkpoint_path)
            
            # Load model
            loaded_trainer = QuantumTrainer.load_checkpoint(
                checkpoint_path,
                model=trainer.model,
                optimizer=trainer.optimizer,
                device='cpu'
            )
            
            assert isinstance(loaded_trainer, QuantumTrainer)
            assert loaded_trainer.model == trainer.model
            assert loaded_trainer.optimizer == trainer.optimizer
            
        finally:
            # Clean up
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)
    
    def test_quantum_training_features(self, trainer):
        """Test quantum-specific training features."""
        # Set quantum training config
        trainer.quantum_training_config = {
            'uncertainty_regularization': 0.1,
            'superposition_schedule': 'linear',
            'entanglement_training': True
        }
        
        # Mock dataloader
        dataloader = Mock()
        dataloader.__iter__.return_value = [
            (torch.randn(2, 10), torch.randint(0, 100, (2, 10)))
        ]
        
        # Mock context
        context = torch.randn(2, 10, 100)
        
        # Train epoch with quantum features
        metrics = trainer.train_epoch(
            dataloader=dataloader,
            context=context
        )
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'uncertainty' in metrics
    
    def test_uncertainty_regularization(self, trainer):
        """Test uncertainty regularization in training."""
        # Set uncertainty regularization
        trainer.quantum_training_config = {
            'uncertainty_regularization': 0.2
        }
        
        # Mock dataloader
        dataloader = Mock()
        dataloader.__iter__.return_value = [
            (torch.randn(2, 10), torch.randint(0, 100, (2, 10)))
        ]
        
        # Mock context
        context = torch.randn(2, 10, 100)
        
        # Train epoch
        metrics = trainer.train_epoch(
            dataloader=dataloader,
            context=context
        )
        
        assert 'uncertainty' in metrics
        assert metrics['uncertainty'] >= 0
    
    def test_superposition_schedule(self, trainer):
        """Test superposition schedule in training."""
        # Set superposition schedule
        trainer.quantum_training_config = {
            'superposition_schedule': 'exponential'
        }
        
        # Mock dataloader
        dataloader = Mock()
        dataloader.__iter__.return_value = [
            (torch.randn(2, 10), torch.randint(0, 100, (2, 10)))
        ]
        
        # Mock context
        context = torch.randn(2, 10, 100)
        
        # Train epoch
        metrics = trainer.train_epoch(
            dataloader=dataloader,
            context=context
        )
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics


if __name__ == "__main__":
    pytest.main([__file__])
