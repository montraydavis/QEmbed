"""
Model comparison utilities for QEmbed.

⚠️ CRITICAL: This comparator integrates with existing QEmbed infrastructure
    and provides comprehensive comparison across all three phases.

Provides statistical significance testing, performance ranking,
and A/B testing frameworks for model evaluation.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
import pandas as pd
import time
import json

# ⚠️ CRITICAL: Import existing QEmbed infrastructure
from .base_evaluator import EvaluationResult
from .aggregation import ResultAggregator

class ModelComparator:
    """
    Comprehensive model comparison utilities.
    
    ⚠️ CRITICAL: Integrates with existing evaluation infrastructure
    and provides statistical analysis across all three phases.
    """
    
    def __init__(self):
        """Initialize model comparator."""
        self.aggregator = ResultAggregator()
        
        # Comparison configuration
        self.comparison_config = {
            'significance_level': 0.05,
            'statistical_tests': ['ttest', 'mannwhitney', 'wilcoxon'],
            'include_quantum_analysis': True,
            'include_metadata_analysis': True
        }
    
    def compare_models(
        self,
        results: List[EvaluationResult],
        model_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models across specified metrics.
        
        Args:
            results: List of evaluation results
            model_names: Specific models to compare (if None, compare all)
            metrics: Specific metrics to compare (if None, compare all)
            
        Returns:
            Comprehensive comparison results
        """
        if not results:
            return {}
        
        # Filter results by model names if specified
        if model_names:
            results = [r for r in results if r.model_name in model_names]
        
        if not results:
            return {}
        
        # Get all available metrics
        if metrics is None:
            metrics = self._get_all_available_metrics(results)
        
        comparison_results = {
            'comparison_metadata': {
                'total_models': len(set(r.model_name for r in results)),
                'total_evaluations': len(results),
                'metrics_compared': metrics,
                'comparison_timestamp': time.time()
            },
            'performance_comparison': self._compare_performance(results, metrics),
            'statistical_significance': self._compute_statistical_significance(results, metrics),
            'quantum_analysis_comparison': self._compare_quantum_analysis(results),
            'metadata_comparison': self._compare_metadata(results),
            'ranking_analysis': self._compute_ranking_analysis(results, metrics)
        }
        
        return comparison_results
    
    def _get_all_available_metrics(self, results: List[EvaluationResult]) -> List[str]:
        """Get all available metrics across results."""
        metric_keys = set()
        for result in results:
            metric_keys.update(result.metrics.keys())
        return list(metric_keys)
    
    def _compare_performance(
        self, 
        results: List[EvaluationResult], 
        metrics: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare performance across specified metrics."""
        performance_comparison = {}
        
        for metric in metrics:
            metric_results = {}
            
            # Group results by model
            model_groups = {}
            for result in results:
                if metric in result.metrics and isinstance(result.metrics[metric], (int, float)):
                    model_name = result.model_name
                    if model_name not in model_groups:
                        model_groups[model_name] = []
                    model_groups[model_name].append(result.metrics[metric])
            
            # Compute statistics for each model
            for model_name, values in model_groups.items():
                if values:
                    metric_results[model_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values),
                        'median': float(np.median(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75))
                    }
            
            performance_comparison[metric] = metric_results
        
        return performance_comparison
    
    def _compute_statistical_significance(
        self, 
        results: List[EvaluationResult], 
        metrics: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute statistical significance between models for each metric."""
        significance_results = {}
        
        for metric in metrics:
            metric_significance = {}
            
            # Group results by model
            model_groups = {}
            for result in results:
                if metric in result.metrics and isinstance(result.metrics[metric], (int, float)):
                    model_name = result.model_name
                    if model_name not in model_groups:
                        model_groups[model_name] = []
                    model_groups[model_name].append(result.metrics[metric])
            
            # Compute pairwise significance tests
            model_names = list(model_groups.keys())
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    if len(model_groups[model1]) > 1 and len(model_groups[model2]) > 1:
                        comparison_key = f"{model1}_vs_{model2}"
                        
                        # Perform statistical tests
                        t_stat, t_pvalue = ttest_ind(model_groups[model1], model_groups[model2])
                        u_stat, u_pvalue = mannwhitneyu(model_groups[model1], model_groups[model2])
                        
                        # Wilcoxon signed-rank test (if same number of samples)
                        w_stat, w_pvalue = None, None
                        if len(model_groups[model1]) == len(model_groups[model2]):
                            try:
                                w_stat, w_pvalue = wilcoxon(model_groups[model1], model_groups[model2])
                            except:
                                pass
                        
                        metric_significance[comparison_key] = {
                            't_test': {
                                'statistic': float(t_stat),
                                'p_value': float(t_pvalue),
                                'significant': t_pvalue < self.comparison_config['significance_level']
                            },
                            'mann_whitney': {
                                'statistic': float(u_stat),
                                'p_value': float(u_pvalue),
                                'significant': u_pvalue < self.comparison_config['significance_level']
                            }
                        }
                        
                        if w_stat is not None:
                            metric_significance[comparison_key]['wilcoxon'] = {
                                'statistic': float(w_stat),
                                'p_value': float(w_pvalue),
                                'significant': w_pvalue < self.comparison_config['significance_level']
                            }
            
            significance_results[metric] = metric_significance
        
        return significance_results
    
    def _compare_quantum_analysis(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compare quantum analysis results across models."""
        if not self.comparison_config['include_quantum_analysis']:
            return {}
        
        quantum_comparison = {}
        
        # Get all quantum metrics
        quantum_metrics = set()
        for result in results:
            quantum_metrics.update(result.quantum_metrics.keys())
        
        for metric in quantum_metrics:
            metric_results = {}
            
            # Group results by model
            model_groups = {}
            for result in results:
                if metric in result.quantum_metrics and isinstance(result.quantum_metrics[metric], (int, float)):
                    model_name = result.model_name
                    if model_name not in model_groups:
                        model_groups[model_name] = []
                    model_groups[model_name].append(result.quantum_metrics[metric])
            
            # Compute statistics for each model
            for model_name, values in model_groups.items():
                if values:
                    metric_results[model_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
            
            quantum_comparison[metric] = metric_results
        
        return quantum_comparison
    
    def _compare_metadata(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compare metadata across models."""
        if not self.comparison_config['include_metadata_analysis']:
            return {}
        
        metadata_comparison = {}
        
        # Get all metadata keys
        metadata_keys = set()
        for result in results:
            metadata_keys.update(result.metadata.keys())
        
        for key in metadata_keys:
            key_results = {}
            
            # Group results by model
            model_groups = {}
            for result in results:
                if key in result.metadata:
                    model_name = result.model_name
                    if model_name not in model_groups:
                        model_groups[model_name] = []
                    model_groups[model_name].append(result.metadata[key])
            
            # Analyze metadata for each model
            for model_name, values in model_groups.items():
                if values:
                    # Handle different metadata types
                    if all(isinstance(v, (int, float)) for v in values):
                        key_results[model_name] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'count': len(values)
                        }
                    else:
                        # For non-numeric metadata, collect unique values
                        key_results[model_name] = {
                            'unique_values': list(set(str(v) for v in values)),
                            'count': len(values)
                        }
            
            metadata_comparison[key] = key_results
        
        return metadata_comparison
    
    def _compute_ranking_analysis(
        self, 
        results: List[EvaluationResult], 
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compute comprehensive ranking analysis."""
        ranking_analysis = {}
        
        for metric in metrics:
            # Get all values for this metric
            metric_values = []
            for result in results:
                if metric in result.metrics and isinstance(result.metrics[metric], (int, float)):
                    metric_values.append((result.model_name, result.metrics[metric]))
            
            if metric_values:
                # Sort by metric value (higher is better for most metrics)
                sorted_values = sorted(metric_values, key=lambda x: x[1], reverse=True)
                
                ranking_analysis[metric] = {
                    'ranking': [model for model, _ in sorted_values],
                    'values': {model: value for model, value in sorted_values},
                    'best_model': sorted_values[0][0] if sorted_values else None,
                    'worst_model': sorted_values[-1][0] if sorted_values else None,
                    'performance_gap': sorted_values[0][1] - sorted_values[-1][1] if len(sorted_values) > 1 else 0
                }
        
        return ranking_analysis
    
    def perform_ab_test(
        self,
        model_a_results: List[EvaluationResult],
        model_b_results: List[EvaluationResult],
        metric: str,
        test_type: str = 'ttest'
    ) -> Dict[str, Any]:
        """
        Perform A/B testing between two models.
        
        Args:
            model_a_results: Results for model A
            model_b_results: Results for model B
            metric: Metric to compare
            test_type: Type of statistical test
            
        Returns:
            A/B test results
        """
        # Extract metric values
        a_values = []
        b_values = []
        
        for result in model_a_results:
            if metric in result.metrics and isinstance(result.metrics[metric], (int, float)):
                a_values.append(result.metrics[metric])
        
        for result in model_b_results:
            if metric in result.metrics and isinstance(result.metrics[metric], (int, float)):
                b_values.append(result.metrics[metric])
        
        if not a_values or not b_values:
            return {'error': 'Insufficient data for A/B testing'}
        
        # Perform statistical test
        if test_type == 'ttest':
            statistic, p_value = ttest_ind(a_values, b_values)
            test_name = 'Independent t-test'
        elif test_type == 'mannwhitney':
            statistic, p_value = mannwhitneyu(a_values, b_values)
            test_name = 'Mann-Whitney U test'
        elif test_type == 'wilcoxon':
            if len(a_values) == len(b_values):
                statistic, p_value = wilcoxon(a_values, b_values)
                test_name = 'Wilcoxon signed-rank test'
            else:
                return {'error': 'Wilcoxon test requires equal sample sizes'}
        else:
            return {'error': f'Unsupported test type: {test_type}'}
        
        # Determine winner
        a_mean = np.mean(a_values)
        b_mean = np.mean(b_values)
        
        if a_mean > b_mean:
            winner = 'Model A'
            improvement = ((a_mean - b_mean) / b_mean) * 100
        else:
            winner = 'Model B'
            improvement = ((b_mean - a_mean) / a_mean) * 100
        
        return {
            'test_type': test_name,
            'model_a': {
                'mean': float(a_mean),
                'std': float(np.std(a_values)),
                'count': len(a_values)
            },
            'model_b': {
                'mean': float(b_mean),
                'std': float(np.std(b_values)),
                'count': len(b_values)
            },
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.comparison_config['significance_level'],
            'winner': winner,
            'improvement_percent': float(improvement),
            'effect_size': float(abs(a_mean - b_mean) / np.sqrt((np.var(a_values) + np.var(b_values)) / 2))
        }
    
    def generate_comparison_report(
        self,
        results: List[EvaluationResult],
        output_format: str = "html"
    ) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            results: List of evaluation results
            output_format: Format of the report
            
        Returns:
            Generated comparison report
        """
        comparison_data = self.compare_models(results)
        
        if output_format == "html":
            return self._generate_comparison_html(comparison_data)
        elif output_format == "json":
            return json.dumps(comparison_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_comparison_html(self, comparison_data: Dict[str, Any]) -> str:
        """Generate HTML comparison report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QEmbed Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .significant {{ background-color: #d4edda; }}
                .not_significant {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 QEmbed Model Comparison Report</h1>
                <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Models Compared:</strong> {comparison_data['comparison_metadata']['total_models']}</p>
                <p><strong>Metrics Compared:</strong> {len(comparison_data['comparison_metadata']['metrics_compared'])}</p>
            </div>
            
            <div class="section">
                <h2>📊 Performance Comparison</h2>
                {self._generate_performance_comparison_html(comparison_data.get('performance_comparison', {}))}
            </div>
            
            <div class="section">
                <h2>📈 Statistical Significance</h2>
                {self._generate_significance_html(comparison_data.get('statistical_significance', {}))}
            </div>
            
            <div class="section">
                <h2>🏆 Ranking Analysis</h2>
                {self._generate_ranking_html(comparison_data.get('ranking_analysis', {}))}
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_performance_comparison_html(self, performance_data: Dict[str, Any]) -> str:
        """Generate HTML for performance comparison."""
        if not performance_data:
            return "<p>No performance data available.</p>"
        
        html = ""
        for metric, model_data in performance_data.items():
            html += f"<h3>{metric}</h3>"
            html += "<table><tr><th>Model</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Count</th></tr>"
            
            for model_name, stats in model_data.items():
                html += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{stats.get('mean', 'N/A'):.4f}</td>
                    <td>{stats.get('std', 'N/A'):.4f}</td>
                    <td>{stats.get('min', 'N/A'):.4f}</td>
                    <td>{stats.get('max', 'N/A'):.4f}</td>
                    <td>{stats.get('count', 'N/A')}</td>
                </tr>
                """
            
            html += "</table><br>"
        
        return html
    
    def _generate_significance_html(self, significance_data: Dict[str, Any]) -> str:
        """Generate HTML for statistical significance."""
        if not significance_data:
            return "<p>No significance data available.</p>"
        
        html = ""
        for metric, comparisons in significance_data.items():
            html += f"<h3>{metric}</h3>"
            html += "<table><tr><th>Comparison</th><th>Test</th><th>Statistic</th><th>P-value</th><th>Significant</th></tr>"
            
            for comparison, tests in comparisons.items():
                for test_name, test_data in tests.items():
                    if test_name != 'wilcoxon' or 'wilcoxon' in tests:
                        significant_class = 'significant' if test_data['significant'] else 'not_significant'
                        html += f"""
                        <tr class="{significant_class}">
                            <td>{comparison}</td>
                            <td>{test_name.replace('_', ' ').title()}</td>
                            <td>{test_data['statistic']:.4f}</td>
                            <td>{test_data['p_value']:.4f}</td>
                            <td>{'Yes' if test_data['significant'] else 'No'}</td>
                        </tr>
                        """
            
            html += "</table><br>"
        
        return html
    
    def _generate_ranking_html(self, ranking_data: Dict[str, Any]) -> str:
        """Generate HTML for ranking analysis."""
        if not ranking_data:
            return "<p>No ranking data available.</p>"
        
        html = "<table><tr><th>Metric</th><th>Best Model</th><th>Worst Model</th><th>Performance Gap</th><th>Full Ranking</th></tr>"
        
        for metric, ranking_info in ranking_data.items():
            best = ranking_info.get('best_model', 'N/A')
            worst = ranking_info.get('worst_model', 'N/A')
            gap = ranking_info.get('performance_gap', 0)
            full_ranking = " → ".join(ranking_info.get('ranking', []))
            
            html += f"""
            <tr>
                <td>{metric}</td>
                <td>{best}</td>
                <td>{worst}</td>
                <td>{gap:.4f}</td>
                <td>{full_ranking}</td>
            </tr>
            """
        
        html += "</table>"
        return html
