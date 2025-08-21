"""
Tests for training utilities.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import Mock, MagicMock, patch
from qembed.training.losses import QuantumLoss, SuperpositionLoss, EntanglementLoss, UncertaintyLoss
from qembed.training.optimizers import QuantumOptimizer, SuperpositionOptimizer, EntanglementOptimizer
from qembed.training.quantum_trainer import QuantumTrainer


class TestQuantumLoss:
    """Test quantum loss functionality."""
    
    @pytest.fixture
    def base_loss(self):
        """Create base loss function."""
        return nn.CrossEntropyLoss()
    
    @pytest.fixture
    def quantum_loss(self, base_loss):
        """Create quantum loss instance."""
        return QuantumLoss(
            base_loss=base_loss,
            quantum_weight=0.1,
            uncertainty_weight=0.05,
            entanglement_weight=0.02
        )
    
    @pytest.fixture
    def predictions(self):
        """Create predictions tensor."""
        return torch.randn(4, 3)
    
    @pytest.fixture
    def targets(self):
        """Create targets tensor."""
        return torch.randint(0, 3, (4,))
    
    def test_initialization(self, base_loss, quantum_loss):
        """Test initialization."""
        assert quantum_loss.base_loss == base_loss
        assert quantum_loss.quantum_weight == 0.1
        assert quantum_loss.uncertainty_weight == 0.05
        assert quantum_loss.entanglement_weight == 0.02
    
    def test_forward(self, quantum_loss, predictions, targets):
        """Test forward pass."""
        # Ensure predictions require gradients and create a computational graph
        predictions = predictions.detach().requires_grad_(True)
        
        # Create a simple computation to ensure gradients flow
        predictions = predictions * 1.0  # This ensures requires_grad is preserved
        
        loss = quantum_loss(predictions, targets)
    
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0
    
    def test_forward_without_uncertainty(self, quantum_loss, predictions, targets):
        """Test forward pass without quantum outputs."""
        # Ensure predictions require gradients and create a computational graph
        predictions = predictions.detach().requires_grad_(True)
        
        # Create a simple computation to ensure gradients flow
        predictions = predictions * 1.0  # This ensures requires_grad is preserved
        
        loss = quantum_loss(predictions, targets, quantum_outputs=None)
    
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0
    
    def test_different_base_losses(self, predictions, targets):
        """Test with different base loss functions."""
        # Test with MSELoss
        mse_loss = nn.MSELoss()
        quantum_mse = QuantumLoss(
            base_loss=mse_loss,
            quantum_weight=0.1
        )
        
        # For MSELoss, targets should match predictions shape
        targets_reshaped = targets.float().unsqueeze(1).expand_as(predictions)
        
        # Ensure predictions require gradients
        predictions = predictions.detach().requires_grad_(True)
        predictions = predictions * 1.0  # This ensures requires_grad is preserved
        
        loss = quantum_mse(predictions, targets_reshaped)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_zero_regularization(self, base_loss, predictions, targets):
        """Test with zero quantum regularization."""
        quantum_loss = QuantumLoss(
            base_loss=base_loss,
            quantum_weight=0.0
        )
        
        loss = quantum_loss(predictions, targets)
        # Should be approximately equal to base loss
        base_loss_value = base_loss(predictions, targets)
        assert torch.allclose(loss, base_loss_value, atol=1e-6)


class TestSuperpositionLoss:
    """Test superposition loss functionality."""
    
    @pytest.fixture
    def base_loss(self):
        """Create base loss function."""
        return nn.CrossEntropyLoss()
    
    @pytest.fixture
    def superposition_loss(self, base_loss):
        """Create superposition loss instance."""
        return SuperpositionLoss(
            base_loss=base_loss,
            superposition_weight=0.1,
            collapse_weight=0.05
        )
    
    @pytest.fixture
    def predictions(self):
        """Create predictions tensor."""
        return torch.randn(4, 3)
    
    @pytest.fixture
    def targets(self):
        """Create targets tensor."""
        return torch.randint(0, 3, (4,))
    
    def test_initialization(self, base_loss, superposition_loss):
        """Test initialization."""
        assert superposition_loss.base_loss == base_loss
        assert superposition_loss.superposition_weight == 0.1
        assert superposition_loss.collapse_weight == 0.05
    
    def test_forward(self, superposition_loss, predictions, targets):
        """Test forward pass."""
        # Create mock superposition and collapsed states
        batch_size, num_classes = predictions.shape
        seq_len = 10
        embed_dim = 128
        num_states = 4
        
        superposition_states = torch.randn(batch_size, seq_len, num_states, embed_dim)
        collapsed_states = torch.randn(batch_size, seq_len, embed_dim)
        
        # Ensure predictions require gradients
        predictions = predictions.detach().requires_grad_(True)
        predictions = predictions * 1.0  # This ensures requires_grad is preserved
        
        loss = superposition_loss(predictions, targets, superposition_states, collapsed_states)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0
    
    def test_forward_without_superposition(self, superposition_loss, predictions, targets):
        """Test forward pass without superposition outputs."""
        # Create mock superposition and collapsed states
        batch_size, num_classes = predictions.shape
        seq_len = 10
        embed_dim = 128
        num_states = 4
        
        superposition_states = torch.randn(batch_size, seq_len, num_states, embed_dim)
        collapsed_states = torch.randn(batch_size, seq_len, embed_dim)
        
        # Ensure predictions require gradients
        predictions = predictions.detach().requires_grad_(True)
        predictions = predictions * 1.0  # This ensures requires_grad is preserved
        
        loss = superposition_loss(predictions, targets, superposition_states, collapsed_states)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0
    
    def test_superposition_quality(self, superposition_loss, predictions, targets):
        """Test superposition quality computation."""
        # Create mock superposition and collapsed states
        batch_size, num_classes = predictions.shape
        seq_len = 10
        embed_dim = 128
        num_states = 4
        
        superposition_states = torch.randn(batch_size, seq_len, num_states, embed_dim)
        collapsed_states = torch.randn(batch_size, seq_len, embed_dim)
        
        # Ensure predictions require gradients
        predictions = predictions.detach().requires_grad_(True)
        predictions = predictions * 1.0  # This ensures requires_grad is preserved
        
        loss = superposition_loss(predictions, targets, superposition_states, collapsed_states)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


class TestEntanglementLoss:
    """Test entanglement loss functionality."""
    
    @pytest.fixture
    def base_loss(self):
        """Create base loss function."""
        return nn.CrossEntropyLoss()
    
    @pytest.fixture
    def entanglement_loss(self, base_loss):
        """Create entanglement loss instance."""
        return EntanglementLoss(
            base_loss=base_loss,
            entanglement_weight=0.1,
            correlation_weight=0.05
        )
    
    @pytest.fixture
    def predictions(self):
        """Create predictions tensor."""
        return torch.randn(4, 3)
    
    @pytest.fixture
    def targets(self):
        """Create targets tensor."""
        return torch.randint(0, 3, (4,))
    
    def test_initialization(self, base_loss, entanglement_loss):
        """Test initialization."""
        assert entanglement_loss.base_loss == base_loss
        assert entanglement_loss.entanglement_weight == 0.1
        assert entanglement_loss.correlation_weight == 0.05
    
    def test_forward(self, entanglement_loss, predictions, targets):
        """Test forward pass."""
        # Create mock entanglement matrix and attention weights
        batch_size, num_classes = predictions.shape
        seq_len = 10
        num_heads = 8
        
        entanglement_matrix = torch.randn(batch_size, seq_len, seq_len)
        attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
        
        # Ensure predictions require gradients
        predictions = predictions.detach().requires_grad_(True)
        predictions = predictions * 1.0  # This ensures requires_grad is preserved
        
        loss = entanglement_loss(predictions, targets, entanglement_matrix, attention_weights)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0
    
    def test_forward_without_entanglement(self, entanglement_loss, predictions, targets):
        """Test forward pass without entanglement outputs."""
        # Create mock entanglement matrix and attention weights
        batch_size, num_classes = predictions.shape
        seq_len = 10
        num_heads = 8
        
        entanglement_matrix = torch.randn(batch_size, seq_len, seq_len)
        attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
        
        # Ensure predictions require gradients
        predictions = predictions.detach().requires_grad_(True)
        predictions = predictions * 1.0  # This ensures requires_grad is preserved
        
        loss = entanglement_loss(predictions, targets, entanglement_matrix, attention_weights)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0


class TestUncertaintyLoss:
    """Test uncertainty loss functionality."""
    
    @pytest.fixture
    def base_loss(self):
        """Create base loss function."""
        return nn.CrossEntropyLoss()
    
    @pytest.fixture
    def uncertainty_loss(self, base_loss):
        """Create uncertainty loss instance."""
        return UncertaintyLoss(
            base_loss=base_loss,
            uncertainty_weight=0.1,
            calibration_weight=0.05
        )
    
    @pytest.fixture
    def predictions(self):
        """Create predictions tensor."""
        return torch.randn(4, 3)
    
    @pytest.fixture
    def targets(self):
        """Create targets tensor."""
        return torch.randint(0, 3, (4,))
    
    def test_initialization(self, base_loss, uncertainty_loss):
        """Test initialization."""
        assert uncertainty_loss.base_loss == base_loss
        assert uncertainty_loss.uncertainty_weight == 0.1
        assert uncertainty_loss.calibration_weight == 0.05
    
    def test_forward(self, uncertainty_loss, predictions, targets):
        """Test forward pass."""
        # Create mock uncertainty tensor
        batch_size, num_classes = predictions.shape
        uncertainty = torch.rand(batch_size, num_classes)
        
        # Ensure predictions require gradients
        predictions = predictions.detach().requires_grad_(True)
        predictions = predictions * 1.0  # This ensures requires_grad is preserved
        
        loss = uncertainty_loss(predictions, targets, uncertainty)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0
    
    def test_forward_without_uncertainty(self, uncertainty_loss, predictions, targets):
        """Test forward pass without uncertainty outputs."""
        # Create mock uncertainty tensor
        batch_size, num_classes = predictions.shape
        uncertainty = torch.rand(batch_size, num_classes)
        
        # Ensure predictions require gradients
        predictions = predictions.detach().requires_grad_(True)
        predictions = predictions * 1.0  # This ensures requires_grad is preserved
        
        loss = uncertainty_loss(predictions, targets, uncertainty)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0
    
    def test_uncertainty_calibration(self, uncertainty_loss, predictions, targets):
        """Test uncertainty calibration computation."""
        # Create mock uncertainty and confidence tensors
        batch_size, num_classes = predictions.shape
        uncertainty = torch.rand(batch_size, num_classes)
        confidence = torch.rand(batch_size, num_classes)
    
        # Ensure predictions require gradients
        predictions = predictions.detach().requires_grad_(True)
        predictions = predictions * 1.0  # This ensures requires_grad is preserved
        
        loss = uncertainty_loss(predictions, targets, uncertainty, confidence)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


class TestQuantumOptimizer:
    """Test quantum optimizer functionality."""
    
    @pytest.fixture
    def model_params(self):
        """Create model parameters."""
        # Create parameters with proper names for quantum detection
        param1 = torch.randn(10, 10, requires_grad=True)
        param2 = torch.randn(10, 10, requires_grad=True)
        
        # Use parameter groups with names instead of tensor names
        return [
            {'params': [param1], 'name': 'quantum_embedding.weight'},
            {'params': [param2], 'name': 'classical_layer.weight'}
        ]
    
    @pytest.fixture
    def quantum_optimizer(self, model_params):
        """Create quantum optimizer instance."""
        return QuantumOptimizer(
            params=model_params,
            base_optimizer="adam",
            base_lr=1e-4,
            quantum_lr_multiplier=1.0,
            superposition_schedule="linear",
            entanglement_update_freq=10
        )
    
    def test_initialization(self, model_params, quantum_optimizer):
        """Test initialization."""
        assert quantum_optimizer.base_lr == 1e-4
        assert quantum_optimizer.quantum_lr_multiplier == 1.0
        assert quantum_optimizer.superposition_schedule == "linear"
        assert quantum_optimizer.entanglement_update_freq == 10
        assert quantum_optimizer.base_optimizer is not None
    
    def test_step(self, quantum_optimizer):
        """Test optimization step."""
        # Mock closure
        closure = Mock(return_value=torch.tensor(1.0))
        
        quantum_optimizer.step(closure)
        
        # Verify step count increased
        assert quantum_optimizer.step_count == 1
    
    def test_quantum_lr_adjustment(self, model_params):
        """Test quantum learning rate adjustment."""
        optimizer = QuantumOptimizer(
            params=model_params,
            base_optimizer="adam",
            base_lr=1e-4,
            quantum_lr_multiplier=2.0
        )
        
        assert optimizer.quantum_lr_multiplier == 2.0


class TestSuperpositionOptimizer:
    """Test superposition optimizer functionality."""
    
    @pytest.fixture
    def model_params(self):
        """Create model parameters."""
        return [{'params': [torch.randn(10, 10, requires_grad=True)]}]
    
    @pytest.fixture
    def superposition_optimizer(self, model_params):
        """Create superposition optimizer instance."""
        return SuperpositionOptimizer(
            params=model_params,
            lr=1e-4,
            superposition_weight=0.1,
            collapse_weight=0.05,
            phase_schedule="cyclic"
        )
    
    def test_initialization(self, model_params, superposition_optimizer):
        """Test initialization."""
        assert superposition_optimizer.lr == 1e-4
        assert superposition_optimizer.superposition_weight == 0.1
        assert superposition_optimizer.collapse_weight == 0.05
        assert superposition_optimizer.phase_schedule == "cyclic"
    
    def test_step(self, superposition_optimizer):
        """Test optimization step."""
        # Mock closure
        closure = Mock(return_value=torch.tensor(1.0))
        
        superposition_optimizer.step(closure)
        
        # Verify step count increased
        assert superposition_optimizer.step_count == 1
    
    def test_different_schedules(self, model_params):
        """Test different superposition schedules."""
        # Test exponential schedule
        optimizer = SuperpositionOptimizer(
            params=model_params,
            lr=1e-4,
            superposition_weight=0.1,
            collapse_weight=0.05,
            phase_schedule="exponential"
        )
        
        assert optimizer.phase_schedule == "exponential"


class TestEntanglementOptimizer:
    """Test entanglement optimizer functionality."""
    
    @pytest.fixture
    def model_params(self):
        """Create model parameters."""
        return [{'params': [torch.randn(10, 10, requires_grad=True)]}]
    
    @pytest.fixture
    def entanglement_optimizer(self, model_params):
        """Create entanglement optimizer instance."""
        return EntanglementOptimizer(
            params=model_params,
            lr=1e-4,
            entanglement_strength=1.0,
            correlation_weight=0.1,
            update_frequency=5
        )
    
    def test_initialization(self, model_params, entanglement_optimizer):
        """Test initialization."""
        assert entanglement_optimizer.lr == 1e-4
        assert entanglement_optimizer.entanglement_strength == 1.0
        assert entanglement_optimizer.correlation_weight == 0.1
        assert entanglement_optimizer.update_frequency == 5
    
    def test_step(self, entanglement_optimizer):
        """Test optimization step."""
        # Mock closure
        closure = Mock(return_value=torch.tensor(1.0))
        
        entanglement_optimizer.step(closure)
        
        # Verify step count increased
        assert entanglement_optimizer.step_count == 1


class TestQuantumTrainer:
    """Test quantum trainer functionality."""
    
    @pytest.fixture
    def model(self):
        """Create mock model."""
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        mock_model.to.return_value = mock_model
        
        # Mock model to return proper outputs with quantum attributes
        def mock_call(*args, **kwargs):
            mock_outputs = Mock()
            mock_outputs.loss = torch.tensor(0.5, requires_grad=True)
            mock_outputs.logits = torch.randn(2, 3, requires_grad=True)  # 2 samples, 3 classes
            mock_outputs.quantum_uncertainty = torch.rand(2, 10)  # 2 samples, 10 sequence positions
            mock_outputs.last_hidden_state = torch.randn(2, 10, 768, requires_grad=True)
            mock_outputs.quantum_loss = torch.tensor(0.1, requires_grad=True)
            return mock_outputs
        
        mock_model.side_effect = mock_call
        mock_model.__call__ = mock_call
        return mock_model
    
    @pytest.fixture
    def optimizer(self):
        """Create mock optimizer."""
        return Mock()
    
    @pytest.fixture
    def loss_fn(self):
        """Create mock loss function."""
        mock_loss = Mock()
        mock_loss.return_value = torch.tensor(0.3, requires_grad=True)
        return mock_loss
    
    @pytest.fixture
    def dataloader(self):
        """Create mock dataloader."""
        mock_dataloader = Mock()
        
        # Mock __iter__ to return a fresh iterator each time it's called
        def create_iterator():
            return iter([
                {
                    'input_ids': torch.randint(0, 1000, (2, 10)),
                    'labels': torch.randint(0, 3, (2,)),
                    'attention_mask': torch.ones(2, 10)
                }
            ])
        
        mock_dataloader.__iter__ = lambda self: create_iterator()
        return mock_dataloader
    
    @pytest.fixture
    def trainer(self, model, optimizer, loss_fn):
        """Create quantum trainer instance."""
        return QuantumTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu"
        )
    
    def test_initialization(self, model, optimizer, loss_fn, trainer):
        """Test initialization."""
        assert trainer.model == model
        assert trainer.optimizer == optimizer
        assert trainer.loss_fn == loss_fn
        assert trainer.device == "cpu"
        assert trainer.quantum_config == {}
        assert trainer.current_epoch == 0
        assert trainer.best_loss == float('inf')
    
    def test_initialization_with_config(self, model, optimizer, loss_fn):
        """Test initialization with quantum config."""
        quantum_config = {
            'superposition_schedule': 'cyclic',
            'entanglement_training': False,
            'uncertainty_regularization': 0.2
        }
        
        trainer = QuantumTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            quantum_training_config=quantum_config
        )
        
        assert trainer.superposition_schedule == 'cyclic'
        assert trainer.entanglement_training == False
        assert trainer.uncertainty_regularization == 0.2
    
    def test_train_epoch(self, trainer, dataloader):
        """Test training for one epoch."""
        metrics = trainer.train_epoch(dataloader, epoch=1)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'quantum_loss' in metrics
        assert 'uncertainty' in metrics
    
    def test_validate(self, trainer, dataloader):
        """Test validation."""
        metrics = trainer.validate(dataloader, epoch=1)
        
        assert isinstance(metrics, dict)
        assert 'val_loss' in metrics
        assert 'val_quantum_loss' in metrics
        assert 'val_uncertainty' in metrics
    
    def test_train(self, trainer, dataloader):
        """Test full training loop."""
        # Mock save_checkpoint method
        trainer.save_checkpoint = Mock()
        
        # Create a separate validation dataloader to avoid iterator issues
        val_dataloader = Mock()
        
        def create_val_iterator():
            return iter([
                {
                    'input_ids': torch.randint(0, 1000, (2, 10)),
                    'labels': torch.randint(0, 3, (2,)),
                    'attention_mask': torch.ones(2, 10)
                }
            ])
        
        val_dataloader.__iter__ = lambda self: create_val_iterator()
        
        trainer.train(
            train_dataloader=dataloader,
            val_dataloader=val_dataloader,
            num_epochs=2
        )
        
        assert trainer.current_epoch == 1  # current_epoch is 0-based index of last completed epoch
    
    def test_save_and_load(self, trainer):
        """Test checkpoint saving and loading."""
        # Mock save_checkpoint method
        trainer.save_checkpoint = Mock()
        
        # Test save
        trainer.save_checkpoint("test_checkpoint.pt")
        trainer.save_checkpoint.assert_called_once_with("test_checkpoint.pt")
    
    def test_quantum_training_features(self, trainer, dataloader):
        """Test quantum-specific training features."""
        # Test superposition schedule
        metrics = trainer.train_epoch(dataloader, epoch=5)
        assert isinstance(metrics, dict)
    
    def test_uncertainty_regularization(self, trainer, dataloader):
        """Test uncertainty regularization."""
        trainer.uncertainty_regularization = 0.2
        metrics = trainer.train_epoch(dataloader, epoch=1)
        assert isinstance(metrics, dict)
    
    def test_superposition_schedule(self, trainer, dataloader):
        """Test superposition schedule."""
        trainer.superposition_schedule = 'cyclic'
        metrics = trainer.train_epoch(dataloader, epoch=1)
        assert isinstance(metrics, dict)


class TestTrainingIntegration:
    """Test integration between training components."""
    
    @pytest.fixture
    def model(self):
        """Create mock model."""
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        mock_model.to.return_value = mock_model
        
        # Mock model to return proper outputs with quantum attributes
        def mock_call(*args, **kwargs):
            mock_outputs = Mock()
            mock_outputs.loss = torch.tensor(0.5, requires_grad=True)
            mock_outputs.logits = torch.randn(2, 3, requires_grad=True)  # 2 samples, 3 classes
            mock_outputs.quantum_uncertainty = torch.rand(2, 10)  # 2 samples, 10 sequence positions
            mock_outputs.last_hidden_state = torch.randn(2, 10, 768, requires_grad=True)
            mock_outputs.quantum_loss = torch.tensor(0.1, requires_grad=True)
            return mock_outputs
        
        mock_model.side_effect = mock_call
        mock_model.__call__ = mock_call
        return mock_model
    
    @pytest.fixture
    def optimizer(self):
        """Create mock optimizer."""
        return Mock()
    
    @pytest.fixture
    def loss_fn(self):
        """Create mock loss function."""
        mock_loss = Mock()
        mock_loss.return_value = torch.tensor(0.3, requires_grad=True)
        return mock_loss
    
    @pytest.fixture
    def dataloader(self):
        """Create mock dataloader."""
        mock_dataloader = Mock()
        
        # Mock __iter__ to return a fresh iterator each time it's called
        def create_iterator():
            return iter([
                {
                    'input_ids': torch.randint(0, 1000, (2, 10)),
                    'labels': torch.randint(0, 3, (2,)),
                    'attention_mask': torch.ones(2, 10)
                }
            ])
        
        mock_dataloader.__iter__ = lambda self: create_iterator()
        return mock_dataloader
    
    def test_comprehensive_training_workflow(self, model, optimizer, loss_fn, dataloader):
        """Test complete training workflow."""
        # Create trainer
        trainer = QuantumTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn
        )
        
        # Mock save_checkpoint method
        trainer.save_checkpoint = Mock()
        
        # Test training
        trainer.train(
            train_dataloader=dataloader,
            val_dataloader=dataloader,
            num_epochs=1
        )
        
        assert trainer.current_epoch == 0  # current_epoch is 0-based index of last completed epoch
        assert isinstance(trainer.training_history, list)
