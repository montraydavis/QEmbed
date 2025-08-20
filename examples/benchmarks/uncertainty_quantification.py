"""
Uncertainty Quantification Benchmark

This benchmark evaluates the uncertainty quantification capabilities of
quantum-enhanced models compared to classical approaches.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import scipy.stats as stats

from qembed.core.quantum_embeddings import QuantumEmbeddings
from qembed.models.quantum_bert import QuantumBERT
from qembed.utils.metrics import QuantumMetrics
from qembed.utils.visualization import QuantumVisualization


@dataclass
class UncertaintySample:
    """Uncertainty quantification sample."""
    input_text: str
    true_label: int
    confidence: float
    uncertainty: float
    model_prediction: int


@dataclass
class UncertaintyResult:
    """Uncertainty quantification result."""
    model_name: str
    calibration_error: float
    expected_calibration_error: float
    uncertainty_correlation: float
    coverage_accuracy: float
    sharpness: float
    inference_time: float
    memory_usage: float
    avg_uncertainty: float
    uncertainty_std: float


class UncertaintyBenchmark:
    """Uncertainty quantification benchmark suite."""
    
    def __init__(self, device: str = 'cpu', num_samples: int = 1000):
        """
        Initialize the benchmark suite.
        
        Args:
            device: Device to run benchmarks on ('cpu' or 'cuda')
            num_samples: Number of samples to generate for testing
        """
        self.device = device
        self.num_samples = num_samples
        self.results = []
        self.metrics = QuantumMetrics()
        self.viz = QuantumVisualization()
        
        # Create synthetic dataset for uncertainty testing
        self.uncertainty_data = self._create_uncertainty_dataset()
        
        print(f"✅ Initialized Uncertainty Benchmark on {device.upper()}")
        print(f"📊 Dataset size: {len(self.uncertainty_data)} samples")
    
    def _create_uncertainty_dataset(self) -> List[UncertaintySample]:
        """Create synthetic dataset for uncertainty testing."""
        samples = []
        
        # Create samples with varying difficulty levels
        for i in range(self.num_samples):
            # Generate synthetic text (simplified)
            text_length = np.random.randint(10, 50)
            text = f"Sample text {i} with length {text_length} for uncertainty testing"
            
            # Generate true label
            true_label = np.random.randint(0, 10)
            
            # Generate confidence based on difficulty
            difficulty = np.random.random()
            if difficulty < 0.3:  # Easy samples
                confidence = np.random.beta(8, 2)  # High confidence
            elif difficulty < 0.7:  # Medium samples
                confidence = np.random.beta(3, 3)  # Medium confidence
            else:  # Hard samples
                confidence = np.random.beta(2, 8)  # Low confidence
            
            # Calculate uncertainty
            uncertainty = 1.0 - confidence
            
            # Generate model prediction (with some errors)
            if np.random.random() < confidence:
                model_prediction = true_label
            else:
                model_prediction = np.random.randint(0, 10)
            
            sample = UncertaintySample(
                input_text=text,
                true_label=true_label,
                confidence=confidence,
                uncertainty=uncertainty,
                model_prediction=model_prediction
            )
            samples.append(sample)
        
        return samples
    
    def _create_quantum_embeddings_model(self, num_states: int = 4) -> QuantumEmbeddings:
        """Create a quantum embeddings model."""
        return QuantumEmbeddings(
            vocab_size=10000,
            embedding_dim=128,
            num_states=num_states,
            dropout=0.1
        ).to(self.device)
    
    def _create_quantum_bert_model(self, num_states: int = 4) -> QuantumBERT:
        """Create a quantum BERT model."""
        return QuantumBERT(
            vocab_size=10000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_quantum_states=num_states,
            intermediate_size=1024,
            dropout=0.1
        ).to(self.device)
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Simple tokenization for demonstration."""
        # Simple word-based tokenization
        words = text.lower().split()
        vocab = {"<pad>": 0, "<unk>": 1}
        
        # Build vocabulary
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
        
        # Convert words to IDs
        token_ids = [vocab.get(w, vocab["<unk>"]) for w in words]
        
        return torch.tensor([token_ids], device=self.device)
    
    def _compute_calibration_error(
        self, 
        confidences: np.ndarray, 
        accuracies: np.ndarray,
        num_bins: int = 10
    ) -> Tuple[float, float]:
        """Compute calibration error and expected calibration error."""
        # Bin the confidence scores
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_errors = []
        expected_errors = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_confidences = confidences[in_bin]
                bin_accuracies = accuracies[in_bin]
                
                # Average confidence and accuracy in this bin
                avg_confidence = np.mean(bin_confidences)
                avg_accuracy = np.mean(bin_accuracies)
                
                # Calibration error
                calibration_error = np.abs(avg_confidence - avg_accuracy)
                calibration_errors.append(calibration_error)
                
                # Expected calibration error (binomial confidence interval)
                expected_error = np.sqrt(avg_accuracy * (1 - avg_accuracy) / bin_size)
                expected_errors.append(expected_error)
        
        # Return average calibration error and expected error
        avg_calibration_error = np.mean(calibration_errors) if calibration_errors else 0.0
        avg_expected_error = np.mean(expected_errors) if expected_errors else 0.0
        
        return avg_calibration_error, avg_expected_error
    
    def _compute_uncertainty_correlation(
        self, 
        uncertainties: np.ndarray, 
        errors: np.ndarray
    ) -> float:
        """Compute correlation between uncertainty and prediction errors."""
        # Convert to binary errors
        binary_errors = (errors > 0).astype(float)
        
        # Compute correlation
        correlation = np.corrcoef(uncertainties, binary_errors)[0, 1]
        
        # Handle NaN values
        if np.isnan(correlation):
            correlation = 0.0
        
        return correlation
    
    def _compute_coverage_accuracy(
        self, 
        uncertainties: np.ndarray, 
        errors: np.ndarray,
        confidence_levels: List[float] = None
    ) -> float:
        """Compute coverage accuracy at different confidence levels."""
        if confidence_levels is None:
            confidence_levels = [0.5, 0.8, 0.9, 0.95]
        
        coverage_scores = []
        
        for confidence_level in confidence_levels:
            # Find samples with uncertainty below threshold
            threshold = 1.0 - confidence_level
            covered_samples = uncertainties <= threshold
            
            if np.sum(covered_samples) > 0:
                # Calculate accuracy of covered samples
                covered_errors = errors[covered_samples]
                coverage_accuracy = 1.0 - np.mean(covered_errors > 0)
                coverage_scores.append(coverage_accuracy)
            else:
                coverage_scores.append(0.0)
        
        # Return average coverage accuracy
        return np.mean(coverage_scores)
    
    def _compute_sharpness(self, uncertainties: np.ndarray) -> float:
        """Compute sharpness (inverse of uncertainty spread)."""
        # Lower uncertainty spread = higher sharpness
        uncertainty_std = np.std(uncertainties)
        sharpness = 1.0 / (1.0 + uncertainty_std)
        return sharpness
    
    def benchmark_quantum_embeddings(self, num_states: int = 4) -> UncertaintyResult:
        """Benchmark quantum embeddings uncertainty quantification."""
        print(f"🔬 Benchmarking Quantum Embeddings Uncertainty ({num_states} states)...")
        
        # Create model
        model = self._create_quantum_embeddings_model(num_states)
        
        # Prepare data
        all_confidences = []
        all_uncertainties = []
        all_errors = []
        
        # Measure inference time
        start_time = time.time()
        
        for sample in self.uncertainty_data:
            # Tokenize text
            token_ids = self._tokenize_text(sample.input_text)
            
            # Get embeddings and uncertainty
            embeddings = model(token_ids, collapse=True)
            model_uncertainty = model.get_uncertainty(token_ids)
            
            # Extract uncertainty value
            uncertainty_value = model_uncertainty[0, 0].item()
            
            # Calculate confidence
            confidence = 1.0 - uncertainty_value
            
            # Calculate error
            error = 1 if sample.model_prediction != sample.true_label else 0
            
            all_confidences.append(confidence)
            all_uncertainties.append(uncertainty_value)
            all_errors.append(error)
        
        inference_time = time.time() - start_time
        
        # Convert to numpy arrays
        confidences = np.array(all_confidences)
        uncertainties = np.array(all_uncertainties)
        errors = np.array(all_errors)
        
        # Calculate metrics
        calibration_error, expected_calibration_error = self._compute_calibration_error(
            confidences, 1 - errors
        )
        uncertainty_correlation = self._compute_uncertainty_correlation(uncertainties, errors)
        coverage_accuracy = self._compute_coverage_accuracy(uncertainties, errors)
        sharpness = self._compute_sharpness(uncertainties)
        
        # Memory usage
        memory_usage = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        result = UncertaintyResult(
            model_name=f"QuantumEmbeddings_{num_states}states",
            calibration_error=calibration_error,
            expected_calibration_error=expected_calibration_error,
            uncertainty_correlation=uncertainty_correlation,
            coverage_accuracy=coverage_accuracy,
            sharpness=sharpness,
            inference_time=inference_time,
            memory_usage=memory_usage,
            avg_uncertainty=np.mean(uncertainties),
            uncertainty_std=np.std(uncertainties)
        )
        
        self.results.append(result)
        print(f"✅ Completed: {calibration_error:.4f} calibration error, {inference_time:.3f}s")
        
        return result
    
    def benchmark_quantum_bert(self, num_states: int = 4) -> UncertaintyResult:
        """Benchmark quantum BERT uncertainty quantification."""
        print(f"🧠 Benchmarking Quantum BERT Uncertainty ({num_states} states)...")
        
        # Create model
        model = self._create_quantum_bert_model(num_states)
        
        # Prepare data
        all_confidences = []
        all_uncertainties = []
        all_errors = []
        
        # Measure inference time
        start_time = time.time()
        
        for sample in self.uncertainty_data:
            # Tokenize text
            token_ids = self._tokenize_text(sample.input_text)
            attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
            
            # Get model outputs and uncertainty
            outputs = model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                collapse=True
            )
            model_uncertainty = model.get_uncertainty(token_ids)
            
            # Extract uncertainty value
            uncertainty_value = model_uncertainty[0, 0].item()
            
            # Calculate confidence
            confidence = 1.0 - uncertainty_value
            
            # Calculate error
            error = 1 if sample.model_prediction != sample.true_label else 0
            
            all_confidences.append(confidence)
            all_uncertainties.append(uncertainty_value)
            all_errors.append(error)
        
        inference_time = time.time() - start_time
        
        # Convert to numpy arrays
        confidences = np.array(all_confidences)
        uncertainties = np.array(all_uncertainties)
        errors = np.array(all_errors)
        
        # Calculate metrics
        calibration_error, expected_calibration_error = self._compute_calibration_error(
            confidences, 1 - errors
        )
        uncertainty_correlation = self._compute_uncertainty_correlation(uncertainties, errors)
        coverage_accuracy = self._compute_coverage_accuracy(uncertainties, errors)
        sharpness = self._compute_sharpness(uncertainties)
        
        # Memory usage
        memory_usage = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        result = UncertaintyResult(
            model_name=f"QuantumBERT_{num_states}states",
            calibration_error=calibration_error,
            expected_calibration_error=expected_calibration_error,
            uncertainty_correlation=uncertainty_correlation,
            coverage_accuracy=coverage_accuracy,
            sharpness=sharpness,
            inference_time=inference_time,
            memory_usage=memory_usage,
            avg_uncertainty=np.mean(uncertainties),
            uncertainty_std=np.std(uncertainties)
        )
        
        self.results.append(result)
        print(f"✅ Completed: {calibration_error:.4f} calibration error, {inference_time:.3f}s")
        
        return result
    
    def benchmark_classical_baseline(self) -> UncertaintyResult:
        """Benchmark classical baseline uncertainty quantification."""
        print("📊 Benchmarking Classical Baseline Uncertainty...")
        
        # Use the synthetic data as baseline
        all_confidences = []
        all_uncertainties = []
        all_errors = []
        
        start_time = time.time()
        
        for sample in self.uncertainty_data:
            all_confidences.append(sample.confidence)
            all_uncertainties.append(sample.uncertainty)
            all_errors.append(1 if sample.model_prediction != sample.true_label else 0)
        
        inference_time = time.time() - start_time
        
        # Convert to numpy arrays
        confidences = np.array(all_confidences)
        uncertainties = np.array(all_uncertainties)
        errors = np.array(all_errors)
        
        # Calculate metrics
        calibration_error, expected_calibration_error = self._compute_calibration_error(
            confidences, 1 - errors
        )
        uncertainty_correlation = self._compute_uncertainty_correlation(uncertainties, errors)
        coverage_accuracy = self._compute_coverage_accuracy(uncertainties, errors)
        sharpness = self._compute_sharpness(uncertainties)
        
        result = UncertaintyResult(
            model_name="ClassicalBaseline",
            calibration_error=calibration_error,
            expected_calibration_error=expected_calibration_error,
            uncertainty_correlation=uncertainty_correlation,
            coverage_accuracy=coverage_accuracy,
            sharpness=sharpness,
            inference_time=inference_time,
            memory_usage=0.0,
            avg_uncertainty=np.mean(uncertainties),
            uncertainty_std=np.std(uncertainties)
        )
        
        self.results.append(result)
        print(f"✅ Completed: {calibration_error:.4f} calibration error, {inference_time:.3f}s")
        
        return result
    
    def run_all_benchmarks(self) -> List[UncertaintyResult]:
        """Run all uncertainty quantification benchmarks."""
        print("\n🚀 Starting Uncertainty Quantification Benchmark Suite")
        print("=" * 60)
        
        # Run classical baseline
        self.benchmark_classical_baseline()
        
        # Run quantum embeddings with different numbers of states
        for num_states in [2, 4, 8]:
            self.benchmark_quantum_embeddings(num_states)
        
        # Run quantum BERT with different numbers of states
        for num_states in [2, 4, 8]:
            self.benchmark_quantum_bert(num_states)
        
        print("\n🎉 All uncertainty benchmarks completed!")
        return self.results
    
    def generate_report(self, output_dir: str = "uncertainty_benchmark_results"):
        """Generate comprehensive uncertainty benchmark report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📊 Generating uncertainty benchmark report in {output_path}")
        
        # Create summary table
        summary_data = []
        for result in self.results:
            summary_data.append({
                'Model': result.model_name,
                'Calibration Error': f"{result.calibration_error:.4f}",
                'Expected Calibration Error': f"{result.expected_calibration_error:.4f}",
                'Uncertainty Correlation': f"{result.uncertainty_correlation:.4f}",
                'Coverage Accuracy': f"{result.coverage_accuracy:.4f}",
                'Sharpness': f"{result.sharpness:.4f}",
                'Inference Time (s)': f"{result.inference_time:.3f}",
                'Memory (MB)': f"{result.memory_usage:.2f}",
                'Avg Uncertainty': f"{result.avg_uncertainty:.4f}",
                'Uncertainty Std': f"{result.uncertainty_std:.4f}"
            })
        
        # Save results as JSON
        results_json = {
            'benchmark_info': {
                'task': 'Uncertainty Quantification',
                'device': self.device,
                'dataset_size': len(self.uncertainty_data),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': [vars(result) for result in self.results]
        }
        
        with open(output_path / 'uncertainty_benchmark_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Create uncertainty comparison plots
        self._create_uncertainty_plots(output_path)
        
        # Create calibration plots
        self._create_calibration_plots(output_path)
        
        # Save summary table
        import pandas as pd
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path / 'uncertainty_benchmark_summary.csv', index=False)
        
        print(f"✅ Report generated successfully!")
        print(f"   • JSON results: {output_path / 'uncertainty_benchmark_results.json'}")
        print(f"   • Summary CSV: {output_path / 'uncertainty_benchmark_summary.csv'}")
        print(f"   • Uncertainty plots: {output_path / 'uncertainty_comparison.png'}")
        print(f"   • Calibration plots: {output_path / 'calibration_analysis.png'}")
    
    def _create_uncertainty_plots(self, output_path: Path):
        """Create uncertainty comparison plots."""
        # Extract data for plotting
        model_names = [result.model_name for result in self.results]
        calibration_errors = [result.calibration_error for result in self.results]
        uncertainty_correlations = [result.uncertainty_correlation for result in self.results]
        coverage_accuracies = [result.coverage_accuracy for result in self.results]
        sharpness_scores = [result.sharpness for result in self.results]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Calibration error comparison
        bars1 = ax1.bar(range(len(model_names)), calibration_errors, color='red', alpha=0.7)
        ax1.set_title('Calibration Error Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Calibration Error (Lower is Better)')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Uncertainty correlation comparison
        bars2 = ax2.bar(range(len(model_names)), uncertainty_correlations, color='blue', alpha=0.7)
        ax2.set_title('Uncertainty Correlation Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Correlation (Higher is Better)')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Coverage accuracy comparison
        bars3 = ax3.bar(range(len(model_names)), coverage_accuracies, color='green', alpha=0.7)
        ax3.set_title('Coverage Accuracy Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Coverage Accuracy (Higher is Better)')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Sharpness comparison
        bars4 = ax4.bar(range(len(model_names)), sharpness_scores, color='orange', alpha=0.7)
        ax4.set_title('Sharpness Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Sharpness (Higher is Better)')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax = bar.axes
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'uncertainty_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_calibration_plots(self, output_path: Path):
        """Create calibration analysis plots."""
        # Create reliability diagram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Calibration error vs expected error
        model_names = [result.model_name for result in self.results]
        calibration_errors = [result.calibration_error for result in self.results]
        expected_errors = [result.expected_calibration_error for result in self.results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, calibration_errors, width, label='Calibration Error', color='red', alpha=0.7)
        bars2 = ax1.bar(x + width/2, expected_errors, width, label='Expected Error', color='blue', alpha=0.7)
        
        ax1.set_title('Calibration Error vs Expected Error', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Error Value')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Uncertainty statistics
        avg_uncertainties = [result.avg_uncertainty for result in self.results]
        uncertainty_stds = [result.uncertainty_std for result in self.results]
        
        bars3 = ax2.bar(x - width/2, avg_uncertainties, width, label='Average Uncertainty', color='green', alpha=0.7)
        bars4 = ax2.bar(x + width/2, uncertainty_stds, width, label='Uncertainty Std', color='purple', alpha=0.7)
        
        ax2.set_title('Uncertainty Statistics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Uncertainty Value')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax = bar.axes
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'calibration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main uncertainty benchmark execution function."""
    print("🌟 QEmbed Uncertainty Quantification Benchmark")
    print("=" * 60)
    print("This benchmark evaluates uncertainty quantification capabilities:")
    print("• Calibration error analysis")
    print("• Uncertainty correlation with errors")
    print("• Coverage accuracy assessment")
    print("• Sharpness evaluation")
    print("• Classical vs quantum comparison")
    print("=" * 60)
    
    # Check device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device.upper()}")
    
    try:
        # Create and run benchmark
        benchmark = UncertaintyBenchmark(device=device, num_samples=1000)
        results = benchmark.run_all_benchmarks()
        
        # Generate comprehensive report
        benchmark.generate_report()
        
        # Print summary
        print("\n📊 Uncertainty Benchmark Summary")
        print("=" * 60)
        print(f"{'Model':<25} {'Cal Error':<10} {'Correlation':<12} {'Coverage':<10} {'Sharpness':<10}")
        print("-" * 60)
        
        for result in results:
            print(f"{result.model_name:<25} {result.calibration_error:<10.4f} {result.uncertainty_correlation:<12.4f} {result.coverage_accuracy:<10.4f} {result.sharpness:<10.4f}")
        
        print("\n🎉 Uncertainty benchmark completed successfully!")
        print("📁 Check the 'uncertainty_benchmark_results' directory for detailed reports and visualizations.")
        
    except Exception as e:
        print(f"\n❌ Error during uncertainty benchmark: {e}")
        print("This might be due to missing dependencies or device issues.")
        print("Check the installation guide for troubleshooting.")


if __name__ == "__main__":
    main()
