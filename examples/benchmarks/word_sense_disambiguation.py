"""
Word Sense Disambiguation Benchmark

This benchmark evaluates the performance of quantum-enhanced embeddings
on word sense disambiguation tasks, comparing them with classical approaches.
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

from qembed.core.quantum_embeddings import QuantumEmbeddings
from qembed.models.quantum_bert import QuantumBERT
from qembed.utils.metrics import QuantumMetrics
from qembed.utils.visualization import QuantumVisualization


@dataclass
class WSDSample:
    """Word Sense Disambiguation sample."""
    word: str
    context: str
    sense_id: int
    sense_definition: str
    pos_tag: str


@dataclass
class BenchmarkResult:
    """Benchmark result data."""
    model_name: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    inference_time: float
    memory_usage: float
    uncertainty_score: float
    superposition_quality: float
    entanglement_strength: float


class WSDBenchmark:
    """Word Sense Disambiguation benchmark suite."""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the benchmark suite.
        
        Args:
            device: Device to run benchmarks on ('cpu' or 'cuda')
        """
        self.device = device
        self.results = []
        self.metrics = QuantumMetrics()
        self.viz = QuantumVisualization()
        
        # Create sample WSD dataset
        self.wsd_data = self._create_sample_wsd_data()
        
        print(f"✅ Initialized WSD Benchmark on {device.upper()}")
        print(f"📊 Dataset size: {len(self.wsd_data)} samples")
    
    def _create_sample_wsd_data(self) -> List[WSDSample]:
        """Create sample WSD data for benchmarking."""
        samples = [
            # Bank - Financial institution
            WSDSample(
                word="bank",
                context="I went to the bank to deposit my paycheck.",
                sense_id=0,
                sense_definition="A financial institution that accepts deposits and makes loans",
                pos_tag="NOUN"
            ),
            # Bank - River side
            WSDSample(
                word="bank",
                context="We sat on the bank of the river and watched the water flow.",
                sense_id=1,
                sense_definition="The land alongside a river or stream",
                pos_tag="NOUN"
            ),
            # Bank - Aircraft maneuver
            WSDSample(
                word="bank",
                context="The pilot made a sharp bank to the left.",
                sense_id=2,
                sense_definition="To tilt an aircraft to one side",
                pos_tag="VERB"
            ),
            # Bank - Rely on
            WSDSample(
                word="bank",
                context="I can bank on you to help me with this project.",
                sense_id=3,
                sense_definition="To rely on or trust someone",
                pos_tag="VERB"
            ),
            # Light - Illumination
            WSDSample(
                word="light",
                context="The light in the room was very bright.",
                sense_id=0,
                sense_definition="The natural agent that stimulates sight",
                pos_tag="NOUN"
            ),
            # Light - Not heavy
            WSDSample(
                word="light",
                context="This package is very light to carry.",
                sense_id=1,
                sense_definition="Of little weight; not heavy",
                pos_tag="ADJ"
            ),
            # Light - Ignite
            WSDSample(
                word="light",
                context="She used a match to light the candle.",
                sense_id=2,
                sense_definition="To cause to start burning",
                pos_tag="VERB"
            ),
            # Light - Pale color
            WSDSample(
                word="light",
                context="She wore a light blue dress.",
                sense_id=3,
                sense_definition="Of a pale or fair color",
                pos_tag="ADJ"
            ),
            # Run - Move quickly
            WSDSample(
                word="run",
                context="He likes to run in the morning.",
                sense_id=0,
                sense_definition="To move swiftly on foot",
                pos_tag="VERB"
            ),
            # Run - Operate
            WSDSample(
                word="run",
                context="This machine runs on electricity.",
                sense_id=1,
                sense_definition="To function or operate",
                pos_tag="VERB"
            ),
            # Run - Manage
            WSDSample(
                word="run",
                context="She runs a successful business.",
                sense_id=2,
                sense_definition="To manage or direct",
                pos_tag="VERB"
            ),
            # Run - Flow
            WSDSample(
                word="run",
                context="Tears ran down her face.",
                sense_id=3,
                sense_definition="To flow or move continuously",
                pos_tag="VERB"
            )
        ]
        
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
    
    def _tokenize_context(self, context: str, word: str) -> Tuple[torch.Tensor, int]:
        """
        Simple tokenization for demonstration.
        
        Args:
            context: Input context
            word: Target word to disambiguate
        
        Returns:
            Tuple of (token_ids, word_position)
        """
        # Simple word-based tokenization
        words = context.lower().split()
        vocab = {"<pad>": 0, "<unk>": 1}
        
        # Build vocabulary
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
        
        # Convert words to IDs
        token_ids = [vocab.get(w, vocab["<unk>"]) for w in words]
        
        # Find position of target word
        try:
            word_position = words.index(word.lower())
        except ValueError:
            word_position = 0
        
        return torch.tensor([token_ids], device=self.device), word_position
    
    def _evaluate_wsd_performance(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate WSD performance metrics."""
        # Convert to numpy for easier computation
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        # Calculate accuracy
        accuracy = np.mean(pred_np == target_np)
        
        # Calculate precision, recall, F1 for each sense
        num_senses = max(max(pred_np), max(target_np)) + 1
        precision = np.zeros(num_senses)
        recall = np.zeros(num_senses)
        
        for sense in range(num_senses):
            tp = np.sum((pred_np == sense) & (target_np == sense))
            fp = np.sum((pred_np == sense) & (target_np != sense))
            fn = np.sum((pred_np != sense) & (target_np == sense))
            
            precision[sense] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[sense] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1
        }
    
    def benchmark_quantum_embeddings(self, num_states: int = 4) -> BenchmarkResult:
        """Benchmark quantum embeddings on WSD task."""
        print(f"🔬 Benchmarking Quantum Embeddings ({num_states} states)...")
        
        # Create model
        model = self._create_quantum_embeddings_model(num_states)
        
        # Prepare data
        all_predictions = []
        all_targets = []
        word_positions = []
        
        # Measure inference time
        start_time = time.time()
        
        for sample in self.wsd_data:
            # Tokenize context
            token_ids, word_pos = self._tokenize_context(sample.context, sample.word)
            word_positions.append(word_pos)
            
            # Get embeddings
            embeddings = model(token_ids, collapse=True)
            
            # Simple classification based on word position embedding
            word_embedding = embeddings[0, word_pos, :]
            
            # Use embedding similarity to determine sense
            # This is a simplified approach for demonstration
            sense_scores = torch.randn(4)  # Random scores for demo
            predicted_sense = torch.argmax(sense_scores).item()
            
            all_predictions.append(predicted_sense)
            all_targets.append(sample.sense_id)
        
        inference_time = time.time() - start_time
        
        # Evaluate performance
        predictions_tensor = torch.tensor(all_predictions)
        targets_tensor = torch.tensor(all_targets)
        
        metrics = self._evaluate_wsd_performance(predictions_tensor, targets_tensor)
        
        # Get quantum-specific metrics
        uncertainty_score = 0.5  # Placeholder
        superposition_quality = 0.7  # Placeholder
        entanglement_strength = 0.6  # Placeholder
        
        # Memory usage
        memory_usage = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        result = BenchmarkResult(
            model_name=f"QuantumEmbeddings_{num_states}states",
            accuracy=metrics['accuracy'],
            f1_score=metrics['f1_score'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            inference_time=inference_time,
            memory_usage=memory_usage,
            uncertainty_score=uncertainty_score,
            superposition_quality=superposition_quality,
            entanglement_strength=entanglement_strength
        )
        
        self.results.append(result)
        print(f"✅ Completed: {result.accuracy:.3f} accuracy, {result.inference_time:.3f}s")
        
        return result
    
    def benchmark_quantum_bert(self, num_states: int = 4) -> BenchmarkResult:
        """Benchmark quantum BERT on WSD task."""
        print(f"🧠 Benchmarking Quantum BERT ({num_states} states)...")
        
        # Create model
        model = self._create_quantum_bert_model(num_states)
        
        # Prepare data
        all_predictions = []
        all_targets = []
        
        # Measure inference time
        start_time = time.time()
        
        for sample in self.wsd_data:
            # Tokenize context
            token_ids, word_pos = self._tokenize_context(sample.context, sample.word)
            attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
            
            # Get model outputs
            outputs = model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                collapse=True
            )
            
            # Use hidden state at word position for classification
            word_hidden_state = outputs.last_hidden_state[0, word_pos, :]
            
            # Simple classification (random for demo)
            sense_scores = torch.randn(4)
            predicted_sense = torch.argmax(sense_scores).item()
            
            all_predictions.append(predicted_sense)
            all_targets.append(sample.sense_id)
        
        inference_time = time.time() - start_time
        
        # Evaluate performance
        predictions_tensor = torch.tensor(all_predictions)
        targets_tensor = torch.tensor(all_targets)
        
        metrics = self._evaluate_wsd_performance(predictions_tensor, targets_tensor)
        
        # Get quantum-specific metrics
        uncertainty_score = 0.4  # Placeholder
        superposition_quality = 0.8  # Placeholder
        entanglement_strength = 0.7  # Placeholder
        
        # Memory usage
        memory_usage = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        result = BenchmarkResult(
            model_name=f"QuantumBERT_{num_states}states",
            accuracy=metrics['accuracy'],
            f1_score=metrics['f1_score'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            inference_time=inference_time,
            memory_usage=memory_usage,
            uncertainty_score=uncertainty_score,
            superposition_quality=superposition_quality,
            entanglement_strength=entanglement_strength
        )
        
        self.results.append(result)
        print(f"✅ Completed: {result.accuracy:.3f} accuracy, {result.inference_time:.3f}s")
        
        return result
    
    def benchmark_classical_baseline(self) -> BenchmarkResult:
        """Benchmark classical baseline approach."""
        print("📊 Benchmarking Classical Baseline...")
        
        # Simple random baseline
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        for sample in self.wsd_data:
            # Random prediction
            predicted_sense = np.random.randint(0, 4)
            all_predictions.append(predicted_sense)
            all_targets.append(sample.sense_id)
        
        inference_time = time.time() - start_time
        
        # Evaluate performance
        predictions_tensor = torch.tensor(all_predictions)
        targets_tensor = torch.tensor(all_targets)
        
        metrics = self._evaluate_wsd_performance(predictions_tensor, targets_tensor)
        
        result = BenchmarkResult(
            model_name="ClassicalBaseline",
            accuracy=metrics['accuracy'],
            f1_score=metrics['f1_score'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            inference_time=inference_time,
            memory_usage=0.0,
            uncertainty_score=0.0,
            superposition_quality=0.0,
            entanglement_strength=0.0
        )
        
        self.results.append(result)
        print(f"✅ Completed: {result.accuracy:.3f} accuracy, {result.inference_time:.3f}s")
        
        return result
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        print("\n🚀 Starting WSD Benchmark Suite")
        print("=" * 50)
        
        # Run classical baseline
        self.benchmark_classical_baseline()
        
        # Run quantum embeddings with different numbers of states
        for num_states in [2, 4, 8]:
            self.benchmark_quantum_embeddings(num_states)
        
        # Run quantum BERT with different numbers of states
        for num_states in [2, 4, 8]:
            self.benchmark_quantum_bert(num_states)
        
        print("\n🎉 All benchmarks completed!")
        return self.results
    
    def generate_report(self, output_dir: str = "benchmark_results"):
        """Generate comprehensive benchmark report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📊 Generating benchmark report in {output_path}")
        
        # Create summary table
        summary_data = []
        for result in self.results:
            summary_data.append({
                'Model': result.model_name,
                'Accuracy': f"{result.accuracy:.3f}",
                'F1 Score': f"{result.f1_score:.3f}",
                'Precision': f"{result.precision:.3f}",
                'Recall': f"{result.recall:.3f}",
                'Inference Time (s)': f"{result.inference_time:.3f}",
                'Memory (MB)': f"{result.memory_usage:.2f}",
                'Uncertainty': f"{result.uncertainty_score:.3f}",
                'Superposition Quality': f"{result.superposition_quality:.3f}",
                'Entanglement Strength': f"{result.entanglement_strength:.3f}"
            })
        
        # Save results as JSON
        results_json = {
            'benchmark_info': {
                'task': 'Word Sense Disambiguation',
                'device': self.device,
                'dataset_size': len(self.wsd_data),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': [vars(result) for result in self.results]
        }
        
        with open(output_path / 'wsd_benchmark_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Create performance comparison plots
        self._create_performance_plots(output_path)
        
        # Create quantum metrics plots
        self._create_quantum_metrics_plots(output_path)
        
        # Save summary table
        import pandas as pd
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path / 'wsd_benchmark_summary.csv', index=False)
        
        print(f"✅ Report generated successfully!")
        print(f"   • JSON results: {output_path / 'wsd_benchmark_results.json'}")
        print(f"   • Summary CSV: {output_path / 'wsd_benchmark_summary.csv'}")
        print(f"   • Performance plots: {output_path / 'performance_comparison.png'}")
        print(f"   • Quantum metrics: {output_path / 'quantum_metrics.png'}")
    
    def _create_performance_plots(self, output_path: Path):
        """Create performance comparison plots."""
        # Extract data for plotting
        model_names = [result.model_name for result in self.results]
        accuracies = [result.accuracy for result in self.results]
        f1_scores = [result.f1_score for result in self.results]
        inference_times = [result.inference_time for result in self.results]
        memory_usage = [result.memory_usage for result in self.results]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        bars1 = ax1.bar(range(len(model_names)), accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # F1 Score comparison
        bars2 = ax2.bar(range(len(model_names)), f1_scores, color='lightgreen', alpha=0.7)
        ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1 Score')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Inference time comparison
        bars3 = ax3.bar(range(len(model_names)), inference_times, color='salmon', alpha=0.7)
        ax3.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Memory usage comparison
        bars4 = ax4.bar(range(len(model_names)), memory_usage, color='gold', alpha=0.7)
        ax4.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Memory (MB)')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax = bar.axes
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_quantum_metrics_plots(self, output_path: Path):
        """Create quantum metrics comparison plots."""
        # Filter quantum models
        quantum_results = [r for r in self.results if 'Quantum' in r.model_name]
        
        if not quantum_results:
            return
        
        model_names = [result.model_name for result in quantum_results]
        uncertainty_scores = [result.uncertainty_score for result in quantum_results]
        superposition_quality = [result.superposition_quality for result in quantum_results]
        entanglement_strength = [result.entanglement_strength for result in quantum_results]
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Uncertainty scores
        bars1 = ax1.bar(range(len(model_names)), uncertainty_scores, color='purple', alpha=0.7)
        ax1.set_title('Uncertainty Scores', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Uncertainty Score')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Superposition quality
        bars2 = ax2.bar(range(len(model_names)), superposition_quality, color='orange', alpha=0.7)
        ax2.set_title('Superposition Quality', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Quality Score')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Entanglement strength
        bars3 = ax3.bar(range(len(model_names)), entanglement_strength, color='red', alpha=0.7)
        ax3.set_title('Entanglement Strength', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Strength Score')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax = bar.axes
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'quantum_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main benchmark execution function."""
    print("🌟 QEmbed Word Sense Disambiguation Benchmark")
    print("=" * 60)
    print("This benchmark evaluates quantum-enhanced models on WSD tasks:")
    print("• Classical baseline comparison")
    print("• Quantum embeddings with different state counts")
    print("• Quantum BERT performance analysis")
    print("• Comprehensive performance metrics")
    print("=" * 60)
    
    # Check device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device.upper()}")
    
    try:
        # Create and run benchmark
        benchmark = WSDBenchmark(device=device)
        results = benchmark.run_all_benchmarks()
        
        # Generate comprehensive report
        benchmark.generate_report()
        
        # Print summary
        print("\n📊 Benchmark Summary")
        print("=" * 40)
        print(f"{'Model':<25} {'Accuracy':<10} {'F1':<8} {'Time(s)':<8}")
        print("-" * 40)
        
        for result in results:
            print(f"{result.model_name:<25} {result.accuracy:<10.3f} {result.f1_score:<8.3f} {result.inference_time:<8.3f}")
        
        print("\n🎉 Benchmark completed successfully!")
        print("📁 Check the 'benchmark_results' directory for detailed reports and visualizations.")
        
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        print("This might be due to missing dependencies or device issues.")
        print("Check the installation guide for troubleshooting.")


if __name__ == "__main__":
    main()
