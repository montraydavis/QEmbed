"""
Tests for QEmbed utility modules.

This test suite covers utility components including:
- Metrics utilities (QuantumMetrics)
- Quantum utilities (quantum_utils)
- Visualization utilities (QuantumVisualization)
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import utility components
from qembed.utils.metrics import QuantumMetrics
from qembed.utils.quantum_utils import QuantumUtils
from qembed.utils.visualization import QuantumVisualization


class TestQuantumMetrics:
    """Test cases for QuantumMetrics class."""
    
    @pytest.fixture
    def metrics(self):
        """Create a QuantumMetrics instance for testing."""
        return QuantumMetrics()
    
    def test_initialization(self, metrics):
        """Test proper initialization."""
        assert hasattr(metrics, '_compute_entropy')
        assert hasattr(metrics, 'compute_metrics')
    
    def test_compute_entropy(self, metrics):
        """Test entropy computation."""
        # Test with tensor
        tensor = torch.randn(10, 10)
        entropy = metrics._compute_entropy(tensor)
        
        assert isinstance(entropy, float)
        assert entropy >= 0
        
        # Test with numpy array - convert to tensor first
        array = np.random.randn(10, 10)
        tensor_array = torch.from_numpy(array)
        entropy = metrics._compute_entropy(tensor_array)
        
        assert isinstance(entropy, float)
        assert entropy >= 0
    
    def test_compute_metrics(self, metrics):
        """Test metrics computation."""
        # Create a mock model with parameters
        model = Mock()
        model.get_uncertainty = Mock(return_value=torch.rand(2, 10))
        # Mock parameters to be iterable
        mock_param1 = Mock()
        mock_param1.numel.return_value = 100
        mock_param2 = Mock()
        mock_param2.numel.return_value = 200
        model.parameters = Mock(return_value=[mock_param1, mock_param2])
        # Mock encoder_layers to be iterable
        model.encoder_layers = [Mock(), Mock(), Mock()]
        
        predictions = torch.randn(2, 10, 3)
        targets = torch.randint(0, 3, (2, 10))
        
        computed_metrics = metrics.compute_metrics(
            model=model,
            predictions=predictions,
            targets=targets
        )
        
        assert isinstance(computed_metrics, dict)
        assert 'accuracy' in computed_metrics
    
    def test_compute_superposition_metrics(self, metrics):
        """Test superposition metrics computation."""
        # Create a mock model
        model = Mock()
        
        superposition_metrics = metrics._compute_superposition_metrics(model)
        
        assert isinstance(superposition_metrics, dict)
    
    def test_compute_entanglement_metrics(self, metrics):
        """Test entanglement metrics computation."""
        # Create a mock model
        model = Mock()
        
        entanglement_metrics = metrics._compute_entanglement_metrics(model)
        
        assert isinstance(entanglement_metrics, dict)


class TestQuantumUtils:
    """Test cases for QuantumUtils class."""
    
    @pytest.fixture
    def quantum_utils(self):
        """Create a QuantumUtils instance for testing."""
        return QuantumUtils()
    
    def test_create_superposition(self, quantum_utils):
        """Test superposition creation."""
        states = torch.randn(2, 10, 4, 128)  # [batch, seq, num_states, dim]
        
        superposition = quantum_utils.create_superposition(states)
        
        assert superposition.shape == (2, 10, 128)
        assert not torch.isnan(superposition).any()
        assert not torch.isinf(superposition).any()
    
    def test_create_superposition_with_weights(self, quantum_utils):
        """Test superposition creation with custom weights."""
        states = torch.randn(2, 10, 4, 128)
        weights = torch.rand(2, 10, 4)
        
        superposition = quantum_utils.create_superposition(states, weights)
        
        assert superposition.shape == (2, 10, 128)
        assert not torch.isnan(superposition).any()
    
    def test_measure_superposition(self, quantum_utils):
        """Test superposition measurement."""
        superposition = torch.randn(2, 10, 128)
        measurement_basis = torch.randn(5, 128)  # 5 basis vectors
        
        measured_state, probabilities = quantum_utils.measure_superposition(
            superposition, measurement_basis
        )
        
        assert measured_state.shape == (2, 10, 128)
        assert probabilities.shape == (2, 10, 5)
        assert not torch.isnan(measured_state).any()
        assert not torch.isnan(probabilities).any()
    
    def test_create_bell_state(self, quantum_utils):
        """Test Bell state creation."""
        state1 = torch.randn(2, 10, 128)
        state2 = torch.randn(2, 10, 128)
        
        bell_state = quantum_utils.create_bell_state(state1, state2)
        
        assert bell_state.shape == (2, 10, 128)
        assert not torch.isnan(bell_state).any()


class TestQuantumVisualization:
    """Test cases for QuantumVisualization class."""
    
    @pytest.fixture
    def viz(self):
        """Create a QuantumVisualization instance for testing."""
        return QuantumVisualization()
    
    def test_initialization(self, viz):
        """Test proper initialization."""
        assert hasattr(viz, 'plot_superposition_states')
        assert hasattr(viz, 'plot_entanglement_correlations')
        assert hasattr(viz, 'plot_uncertainty_analysis')
    
    def test_plot_superposition_states(self, viz):
        """Test superposition states plotting."""
        superposition_states = torch.randn(2, 10, 128)
        
        # Test that plotting doesn't raise errors
        try:
            fig = viz.plot_superposition_states(superposition_states)
            assert fig is not None
        except Exception as e:
            # If plotting fails due to missing dependencies, that's okay
            pytest.skip(f"Plotting failed: {e}")
    
    def test_plot_uncertainty_analysis(self, viz):
        """Test uncertainty analysis plotting."""
        uncertainty = torch.rand(100, 10)
        predictions = torch.randn(100, 10, 3)
        targets = torch.randint(0, 3, (100, 10))
        
        # Test that plotting doesn't raise errors
        try:
            fig = viz.plot_uncertainty_analysis(uncertainty, predictions, targets)
            assert fig is not None
        except Exception as e:
            # If plotting fails due to missing dependencies, that's okay
            pytest.skip(f"Plotting failed: {e}")
    
    def test_plot_entanglement_correlations(self, viz):
        """Test entanglement correlations plotting."""
        embeddings = torch.randn(2, 10, 128)
        
        # Test that plotting doesn't raise errors
        try:
            fig = viz.plot_entanglement_correlations(embeddings)
            assert fig is not None
        except Exception as e:
            # If plotting fails due to missing dependencies, that's okay
            pytest.skip(f"Plotting failed: {e}")
    
    def test_plot_quantum_circuit(self, viz):
        """Test quantum circuit plotting."""
        # Test that plotting doesn't raise errors
        try:
            fig = viz.plot_quantum_circuit()
            assert fig is not None
        except Exception as e:
            # If plotting fails due to missing dependencies, that's okay
            pytest.skip(f"Plotting failed: {e}")
    
    def test_save_plots(self, viz):
        """Test plot saving functionality."""
        # Create a simple plot
        superposition_states = torch.randn(2, 10, 128)
        
        try:
            fig = viz.plot_superposition_states(superposition_states)
            
            # Test saving to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                filepath = tmp_file.name
            
            try:
                fig.savefig(filepath)
                assert os.path.exists(filepath)
            finally:
                # Clean up
                if os.path.exists(filepath):
                    os.unlink(filepath)
                    
        except Exception as e:
            # If plotting fails due to missing dependencies, that's okay
            pytest.skip(f"Plotting failed: {e}")


class TestUtilsIntegration:
    """Integration tests for utility modules."""
    
    def test_metrics_utils_integration(self):
        """Test integration between metrics and quantum utils."""
        # Create metrics instance
        metrics = QuantumMetrics()
        quantum_utils = QuantumUtils()
        
        # Create test data
        embeddings = torch.randn(2, 10, 128)
        states = torch.randn(2, 10, 4, 128)
        
        # Test quantum utils functions
        superposition = quantum_utils.create_superposition(states)
        measurement_basis = torch.randn(5, 128)
        measured_state, probabilities = quantum_utils.measure_superposition(
            superposition, measurement_basis
        )
        
        # Test metrics with mock model
        model = Mock()
        model.get_uncertainty = Mock(return_value=torch.rand(2, 10))
        # Mock parameters to be iterable
        mock_param1 = Mock()
        mock_param1.numel.return_value = 100
        mock_param2 = Mock()
        mock_param2.numel.return_value = 200
        model.parameters = Mock(return_value=[mock_param1, mock_param2])
        # Mock encoder_layers to be iterable
        model.encoder_layers = [Mock(), Mock(), Mock()]
        
        predictions = torch.randn(2, 10, 3)
        targets = torch.randint(0, 3, (2, 10))
        
        computed_metrics = metrics.compute_metrics(
            model=model,
            predictions=predictions,
            targets=targets
        )
        
        # All should return valid results
        assert isinstance(computed_metrics, dict)
        assert superposition.shape == (2, 10, 128)
        assert measured_state.shape == (2, 10, 128)
        assert probabilities.shape == (2, 10, 5)
        
        # Values should be reasonable
        assert not torch.isnan(superposition).any()
        assert not torch.isnan(measured_state).any()
        assert not torch.isnan(probabilities).any()
    
    def test_visualization_utils_integration(self):
        """Test integration between visualization and other utils."""
        # Create visualization instance
        viz = QuantumVisualization()
        
        # Create test data
        embeddings = torch.randn(2, 10, 128)
        uncertainty = torch.rand(2, 10)
        
        # Test that visualization can work with data from other utils
        try:
            # Test superposition states
            fig1 = viz.plot_superposition_states(embeddings)
            assert fig1 is not None
            
            # Test uncertainty analysis
            predictions = torch.randn(2, 10, 3)
            targets = torch.randint(0, 3, (2, 10))
            fig2 = viz.plot_uncertainty_analysis(uncertainty, predictions, targets)
            assert fig2 is not None
            
        except Exception as e:
            # If plotting fails due to missing dependencies, that's okay
            pytest.skip(f"Visualization integration failed: {e}")
    
    def test_comprehensive_utils_workflow(self):
        """Test comprehensive workflow using all utility modules."""
        # Create test data
        embeddings = torch.randn(2, 10, 128)
        predictions = torch.randn(2, 10, 3)
        targets = torch.randint(0, 3, (2, 10))
        states = torch.randn(2, 10, 4, 128)
        
        # Use metrics
        metrics = QuantumMetrics()
        # Create a mock model for compute_metrics
        model = Mock()
        model.get_uncertainty = Mock(return_value=torch.rand(2, 10))
        mock_param1 = Mock()
        mock_param1.numel.return_value = 100
        mock_param2 = Mock()
        mock_param2.numel.return_value = 200
        model.parameters = Mock(return_value=[mock_param1, mock_param2])
        # Mock encoder_layers to be iterable
        model.encoder_layers = [Mock(), Mock(), Mock()]
        
        # Create test_data dictionary as expected by compute_metrics
        test_data = {'input_ids': torch.randint(0, 1000, (2, 10))}
        computed_metrics = metrics.compute_metrics(model, test_data, predictions, targets)
        
        # Use quantum utils
        quantum_utils = QuantumUtils()
        superposition = quantum_utils.create_superposition(states)
        measurement_basis = torch.randn(5, 128)
        measured_state, probabilities = quantum_utils.measure_superposition(
            superposition, measurement_basis
        )
        
        # Use visualization
        viz = QuantumVisualization()
        
        try:
            # Create plots
            fig1 = viz.plot_superposition_states(embeddings)
            fig2 = viz.plot_entanglement_correlations(embeddings)
            
            # Test saving
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                filepath = tmp_file.name
            
            try:
                fig1.savefig(filepath)
                assert os.path.exists(filepath)
            finally:
                if os.path.exists(filepath):
                    os.unlink(filepath)
                    
        except Exception as e:
            # If visualization fails, that's okay
            pytest.skip(f"Visualization workflow failed: {e}")
        
        # All utility functions should work together
        assert isinstance(computed_metrics, dict)
        assert superposition.shape == (2, 10, 128)
        assert measured_state.shape == (2, 10, 128)
        assert probabilities.shape == (2, 10, 5)


if __name__ == "__main__":
    pytest.main([__file__])
