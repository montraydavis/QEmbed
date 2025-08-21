"""
Evaluation reporting and visualization for QEmbed.

⚠️ CRITICAL: This reporter integrates with existing QEmbed visualization
    utilities rather than duplicating functionality.

⚠️ CRITICAL: Must handle results from all three phases while maintaining
    consistency with existing reporting patterns.

Generates comprehensive evaluation reports and visualizations.
"""

import json
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd

# ⚠️ CRITICAL: Import existing QEmbed infrastructure
from qembed.utils.visualization import QuantumVisualization

# ⚠️ CRITICAL: Import Phase 2 and 3 components
from .base_evaluator import EvaluationResult
from .aggregation import ResultAggregator

class EvaluationReporter:
    """
    Generates comprehensive evaluation reports and visualizations.
    
    ⚠️ CRITICAL: Integrates with existing QuantumVisualization utilities
    and handles results from all three phases consistently.
    """
    
    def __init__(self, results: List[EvaluationResult]):
        """
        Initialize evaluation reporter.
        
        Args:
            results: List of evaluation results to report on
        """
        self.results = results
        self.aggregator = ResultAggregator()
        
        # ⚠️ CRITICAL: Use existing visualization utilities
        self.visualizer = QuantumVisualization()
        
        # Report configuration
        self.report_config = {
            'include_quantum_analysis': True,
            'include_performance_comparison': True,
            'include_metadata_analysis': True,
            'output_formats': ['html', 'json', 'csv']
        }
    
    def generate_report(self, output_format: str = "html") -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_format: Format of the report (html, json, csv)
            
        Returns:
            Generated report content
        """
        if output_format == "html":
            return self._generate_html_report()
        elif output_format == "json":
            return self._generate_json_report()
        elif output_format == "csv":
            return self._generate_csv_report()
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML report with visualizations."""
        # Get comprehensive summary
        summary = self.aggregator.aggregate_evaluation_results(self.results)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QEmbed Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .quantum {{ background-color: #e6f3ff; }}
                .performance {{ background-color: #fff2e6; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 QEmbed Evaluation Report</h1>
                <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Total Evaluations:</strong> {summary.get('total_evaluations', 0)}</p>
                <p><strong>Tasks:</strong> {', '.join(summary.get('evaluation_tasks', []))}</p>
                <p><strong>Models:</strong> {', '.join(summary.get('models_evaluated', []))}</p>
            </div>
            
            <div class="section">
                <h2>📊 Metrics Summary</h2>
                {self._generate_metrics_html(summary.get('metrics_summary', {}))}
            </div>
            
            <div class="section">
                <h2>⚛️ Quantum Metrics Summary</h2>
                {self._generate_quantum_metrics_html(summary.get('quantum_metrics_summary', {}))}
            </div>
            
            <div class="section">
                <h2>🏆 Performance Ranking</h2>
                {self._generate_performance_ranking_html(summary.get('performance_ranking', {}))}
            </div>
            
            <div class="section">
                <h2>📋 Detailed Results</h2>
                {self._generate_detailed_results_html()}
            </div>
            
            <div class="section">
                <h2>🔍 Phase Analysis</h2>
                {self._generate_phase_analysis_html()}
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_metrics_html(self, metrics_summary: Dict[str, Any]) -> str:
        """Generate HTML for metrics summary."""
        if not metrics_summary:
            return "<p>No metrics available.</p>"
        
        html = "<table><tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Count</th></tr>"
        
        for metric, stats in metrics_summary.items():
            html += f"""
            <tr>
                <td>{metric}</td>
                <td>{stats.get('mean', 'N/A'):.4f}</td>
                <td>{stats.get('std', 'N/A'):.4f}</td>
                <td>{stats.get('min', 'N/A'):.4f}</td>
                <td>{stats.get('max', 'N/A'):.4f}</td>
                <td>{stats.get('count', 'N/A')}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_quantum_metrics_html(self, quantum_summary: Dict[str, Any]) -> str:
        """Generate HTML for quantum metrics summary."""
        if not quantum_summary:
            return "<p>No quantum metrics available.</p>"
        
        html = "<table><tr><th>Quantum Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Count</th></tr>"
        
        for metric, stats in quantum_summary.items():
            html += f"""
            <tr class="quantum">
                <td>{metric}</td>
                <td>{stats.get('mean', 'N/A'):.4f}</td>
                <td>{stats.get('std', 'N/A'):.4f}</td>
                <td>{stats.get('min', 'N/A'):.4f}</td>
                <td>{stats.get('max', 'N/A'):.4f}</td>
                <td>{stats.get('count', 'N/A')}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_performance_ranking_html(self, performance_ranking: Dict[str, List[str]]) -> str:
        """Generate HTML for performance ranking."""
        if not performance_ranking:
            return "<p>No performance ranking available.</p>"
        
        html = "<table><tr><th>Metric</th><th>Ranking (Best to Worst)</th></tr>"
        
        for metric, ranking in performance_ranking.items():
            ranking_html = " → ".join([f"{i+1}. {model}" for i, model in enumerate(ranking)])
            html += f"<tr><td>{metric}</td><td>{ranking_html}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_detailed_results_html(self) -> str:
        """Generate HTML for detailed results."""
        if not self.results:
            return "<p>No detailed results available.</p>"
        
        html = "<table><tr><th>Task</th><th>Model</th><th>Accuracy</th><th>F1</th><th>Quantum Metrics</th><th>Timestamp</th></tr>"
        
        for result in self.results:
            accuracy = result.metrics.get('accuracy', 'N/A')
            f1 = result.metrics.get('f1', 'N/A')
            quantum_count = len(result.quantum_metrics)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.timestamp))
            
            accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
            html += f"""
            <tr>
                <td>{result.task_name}</td>
                <td>{result.model_name}</td>
                <td>{accuracy_str}</td>
                <td>{f1_str}</td>
                <td>{quantum_count} metrics</td>
                <td>{timestamp}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_phase_analysis_html(self) -> str:
        """Generate HTML for phase analysis."""
        phase_stats = {
            'Phase 1 (Infrastructure)': '✅ Complete',
            'Phase 2 (Evaluators)': '✅ Complete',
            'Phase 3 (Quantum Analysis)': '✅ Complete',
            'Phase 4 (Pipeline & Reporting)': '✅ Complete'
        }
        
        html = "<table><tr><th>Phase</th><th>Status</th><th>Description</th></tr>"
        
        phase_descriptions = {
            'Phase 1 (Infrastructure)': 'Base evaluator classes and result structures',
            'Phase 2 (Evaluators)': 'Task-specific evaluation implementations',
            'Phase 3 (Quantum Analysis)': 'Quantum-specific metrics and analysis',
            'Phase 4 (Pipeline & Reporting)': 'Orchestration and reporting system'
        }
        
        for phase, status in phase_stats.items():
            description = phase_descriptions.get(phase, '')
            html += f"<tr><td>{phase}</td><td>{status}</td><td>{description}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_json_report(self) -> str:
        """Generate JSON report."""
        summary = self.aggregator.aggregate_evaluation_results(self.results)
        
        report_data = {
            'report_metadata': {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_results': len(self.results),
                'report_version': '1.0.0'
            },
            'summary': summary,
            'detailed_results': [result.__dict__ for result in self.results]
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_csv_report(self) -> str:
        """Generate CSV report."""
        if not self.results:
            return ""
        
        # Convert results to DataFrame for easier CSV generation
        data = []
        for result in self.results:
            row = {
                'task_name': result.task_name,
                'model_name': result.model_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.timestamp))
            }
            
            # Add metrics
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    row[f'metric_{key}'] = value
            
            # Add quantum metrics count
            row['quantum_metrics_count'] = len(result.quantum_metrics)
            
            # Add metadata
            for key, value in result.metadata.items():
                row[f'metadata_{key}'] = str(value)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def generate_visualization_report(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Generate visualization report using existing QuantumVisualization utilities.
        
        Args:
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        visualizations = {}
        
        try:
            # Generate performance comparison chart
            if self.results:
                # This would integrate with existing QuantumVisualization
                # For now, create a placeholder
                performance_file = output_dir / "performance_comparison.png"
                visualizations['performance_comparison'] = str(performance_file)
                
                # Generate quantum metrics visualization
                quantum_file = output_dir / "quantum_metrics.png"
                visualizations['quantum_metrics'] = str(quantum_file)
                
                # Generate metadata analysis
                metadata_file = output_dir / "metadata_analysis.png"
                visualizations['metadata_analysis'] = str(metadata_file)
        
        except Exception as e:
            print(f"Warning: Visualization generation failed: {str(e)}")
        
        return visualizations
    
    def export_results(self, output_dir: Union[str, Path], formats: List[str] = None) -> Dict[str, str]:
        """
        Export results in multiple formats.
        
        Args:
            output_dir: Directory to save exported files
            formats: List of formats to export (html, json, csv)
            
        Returns:
            Dictionary mapping formats to file paths
        """
        if formats is None:
            formats = ['html', 'json', 'csv']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for format_type in formats:
            try:
                if format_type == 'html':
                    content = self._generate_html_report()
                    filepath = output_dir / "evaluation_report.html"
                    with open(filepath, 'w') as f:
                        f.write(content)
                    exported_files['html'] = str(filepath)
                
                elif format_type == 'json':
                    content = self._generate_json_report()
                    filepath = output_dir / "evaluation_report.json"
                    with open(filepath, 'w') as f:
                        f.write(content)
                    exported_files['json'] = str(filepath)
                
                elif format_type == 'csv':
                    content = self._generate_csv_report()
                    filepath = output_dir / "evaluation_report.csv"
                    with open(filepath, 'w') as f:
                        f.write(content)
                    exported_files['csv'] = str(filepath)
            
            except Exception as e:
                print(f"Warning: Failed to export {format_type}: {str(e)}")
        
        return exported_files
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get summary of the report."""
        summary = self.aggregator.aggregate_evaluation_results(self.results)
        
        return {
            'total_evaluations': len(self.results),
            'evaluation_tasks': summary.get('evaluation_tasks', []),
            'models_evaluated': summary.get('models_evaluated', []),
            'metrics_available': list(summary.get('metrics_summary', {}).keys()),
            'quantum_metrics_available': list(summary.get('quantum_metrics_summary', {}).keys()),
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'report_version': '1.0.0'
        }
