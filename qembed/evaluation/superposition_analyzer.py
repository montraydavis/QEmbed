"""
Superposition state analysis for QEmbed.

⚠️ CRITICAL: This analyzer uses existing superposition logic from QuantumEmbeddings
    and integrates with existing collapse mechanisms.

Analyzes superposition state evolution and quantum interference effects.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# ⚠️ CRITICAL: Import existing QEmbed infrastructure
from qembed.core.quantum_embeddings import QuantumEmbeddings
from qembed.utils.metrics import QuantumMetrics

class SuperpositionAnalyzer:
    """
    Analyzer for superposition state analysis and quantum interference effects.
    
    ⚠️ CRITICAL: Integrates with existing QuantumEmbeddings superposition logic
    and follows Phase 2 integration patterns.
    """
    
    def __init__(self):
        """Initialize superposition analyzer."""
        self.quantum_metrics = QuantumMetrics()
    
    def analyze_superposition_states(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        superposition_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze superposition state characteristics.
        
        Args:
            embeddings: Embedding vectors [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask
            superposition_matrix: Optional superposition mixing matrix
            
        Returns:
            Dictionary of superposition state metrics
        """
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Basic superposition metrics
        superposition_metrics = {
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'embedding_dimension': embed_dim
        }
        
        # Compute superposition strength
        superposition_strength = self._compute_superposition_strength(embeddings)
        superposition_metrics.update(superposition_strength)
        
        # Compute superposition coherence
        superposition_coherence = self._compute_superposition_coherence(embeddings)
        superposition_metrics.update(superposition_coherence)
        
        # Compute superposition diversity
        superposition_diversity = self._compute_superposition_diversity(embeddings)
        superposition_metrics.update(superposition_diversity)
        
        # Analyze superposition matrix if provided
        if superposition_matrix is not None:
            matrix_analysis = self._analyze_superposition_matrix(superposition_matrix)
            superposition_metrics.update(matrix_analysis)
        
        return superposition_metrics
    
    def _compute_superposition_strength(
        self, 
        embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Compute superposition strength metrics."""
        # Compute variance across sequence dimension as superposition measure
        variance_across_seq = torch.var(embeddings, dim=1)  # [batch_size, embed_dim]
        mean_variance = torch.mean(variance_across_seq, dim=1)  # [batch_size]
        
        # Compute superposition strength as normalized variance
        embedding_norms = torch.norm(embeddings, p=2, dim=-1)  # [batch_size, seq_len]
        mean_norms = torch.mean(embedding_norms, dim=1)  # [batch_size]
        
        superposition_strength = mean_variance / (mean_norms ** 2 + 1e-8)
        
        return {
            'mean_superposition_strength': superposition_strength.mean().item(),
            'std_superposition_strength': superposition_strength.std().item(),
            'max_superposition_strength': superposition_strength.max().item(),
            'min_superposition_strength': superposition_strength.min().item()
        }
    
    def _compute_superposition_coherence(
        self, 
        embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Compute superposition coherence metrics."""
        batch_size, seq_len, embed_dim = embeddings.shape
        
        coherence_scores = []
        for i in range(batch_size):
            batch_embeddings = embeddings[i]  # [seq_len, embed_dim]
            
            # Compute coherence as consistency across sequence positions
            position_correlations = []
            for j in range(seq_len):
                for k in range(j + 1, seq_len):
                    pos_j = batch_embeddings[j]
                    pos_k = batch_embeddings[k]
                    
                    # Compute cosine similarity between positions
                    similarity = torch.dot(pos_j, pos_k) / (torch.norm(pos_j) * torch.norm(pos_k) + 1e-8)
                    position_correlations.append(similarity.item())
            
            if position_correlations:
                coherence_scores.append(np.mean(position_correlations))
        
        if coherence_scores:
            coherence_scores = np.array(coherence_scores)
            return {
                'mean_superposition_coherence': float(np.mean(coherence_scores)),
                'std_superposition_coherence': float(np.std(coherence_scores)),
                'superposition_coherence_entropy': float(stats.entropy(np.histogram(coherence_scores, bins=20)[0] + 1e-8))
            }
        else:
            return {
                'mean_superposition_coherence': 0.0,
                'std_superposition_coherence': 0.0,
                'superposition_coherence_entropy': 0.0
            }
    
    def _compute_superposition_diversity(
        self, 
        embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Compute superposition diversity metrics."""
        batch_size, seq_len, embed_dim = embeddings.shape
        
        diversity_scores = []
        for i in range(batch_size):
            batch_embeddings = embeddings[i]  # [seq_len, embed_dim]
            
            # Compute diversity as spread across embedding space
            # Use pairwise distances between sequence positions
            distances = pdist(batch_embeddings.cpu().numpy(), metric='euclidean')
            diversity_scores.append(np.mean(distances))
        
        diversity_scores = np.array(diversity_scores)
        
        return {
            'mean_superposition_diversity': float(np.mean(diversity_scores)),
            'std_superposition_diversity': float(np.std(diversity_scores)),
            'superposition_diversity_entropy': float(stats.entropy(np.histogram(diversity_scores, bins=20)[0] + 1e-8))
        }
    
    def _analyze_superposition_matrix(
        self, 
        superposition_matrix: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze superposition mixing matrix."""
        matrix_np = superposition_matrix.detach().cpu().numpy()
        
        # Compute matrix properties
        eigenvals, eigenvecs = np.linalg.eig(matrix_np)
        eigenvals = np.real(eigenvals)
        
        # Eigenvalue analysis
        eigenval_entropy = stats.entropy(np.abs(eigenvals) + 1e-8)
        eigenval_stability = 1.0 - np.std(np.abs(eigenvals))
        
        # Matrix rank and condition
        matrix_rank = np.linalg.matrix_rank(matrix_np)
        matrix_condition = np.linalg.cond(matrix_np)
        
        # Mixing efficiency
        identity = np.eye(matrix_np.shape[0])
        mixing_efficiency = np.linalg.norm(matrix_np - identity) / np.linalg.norm(identity)
        
        return {
            'eigenvalue_entropy': float(eigenval_entropy),
            'eigenvalue_stability': float(eigenval_stability),
            'matrix_rank': int(matrix_rank),
            'matrix_condition': float(matrix_condition),
            'mixing_efficiency': float(mixing_efficiency)
        }
    
    def analyze_collapse_mechanisms(
        self,
        embeddings: torch.Tensor,
        collapse_probabilities: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze superposition collapse mechanisms.
        
        Args:
            embeddings: Embedding vectors [batch_size, seq_len, embed_dim]
            collapse_probabilities: Optional collapse probabilities
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary of collapse mechanism metrics
        """
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        batch_size, seq_len, embed_dim = embeddings.shape
        
        collapse_metrics = {}
        
        # Analyze collapse patterns if probabilities provided
        if collapse_probabilities is not None:
            collapse_probs = collapse_probabilities.detach().cpu().numpy()
            
            collapse_metrics.update({
                'mean_collapse_probability': float(np.mean(collapse_probs)),
                'std_collapse_probability': float(np.std(collapse_probs)),
                'collapse_probability_entropy': float(stats.entropy(np.histogram(collapse_probs, bins=20)[0] + 1e-8))
            })
        
        # Analyze collapse effects on embeddings
        collapse_effects = self._analyze_collapse_effects(embeddings)
        collapse_metrics.update(collapse_effects)
        
        return collapse_metrics
    
    def _analyze_collapse_effects(
        self, 
        embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze effects of collapse on embedding structure."""
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Compute collapse stability (how much embeddings change across sequence)
        stability_scores = []
        for i in range(batch_size):
            batch_embeddings = embeddings[i]  # [seq_len, embed_dim]
            
            # Compute stability as consistency across sequence
            position_variations = []
            for j in range(seq_len - 1):
                current_pos = batch_embeddings[j]
                next_pos = batch_embeddings[j + 1]
                
                # Compute change magnitude
                change = torch.norm(next_pos - current_pos)
                position_variations.append(change.item())
            
            if position_variations:
                stability = 1.0 / (1.0 + np.mean(position_variations))
                stability_scores.append(stability)
        
        if stability_scores:
            stability_scores = np.array(stability_scores)
            return {
                'mean_collapse_stability': float(np.mean(stability_scores)),
                'std_collapse_stability': float(np.std(stability_scores)),
                'collapse_stability_entropy': float(stats.entropy(np.histogram(stability_scores, bins=20)[0] + 1e-8))
            }
        else:
            return {
                'mean_collapse_stability': 0.0,
                'std_collapse_stability': 0.0,
                'collapse_stability_entropy': 0.0
            }
    
    def analyze_quantum_interference(
        self,
        embeddings: torch.Tensor,
        interference_matrix: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze quantum interference effects.
        
        Args:
            embeddings: Embedding vectors [batch_size, seq_len, embed_dim]
            interference_matrix: Optional interference matrix
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary of interference analysis metrics
        """
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        batch_size, seq_len, embed_dim = embeddings.shape
        
        interference_metrics = {}
        
        # Analyze interference patterns
        interference_patterns = self._compute_interference_patterns(embeddings)
        interference_metrics.update(interference_patterns)
        
        # Analyze interference matrix if provided
        if interference_matrix is not None:
            matrix_analysis = self._analyze_interference_matrix(interference_matrix)
            interference_metrics.update(matrix_analysis)
        
        return interference_metrics
    
    def _compute_interference_patterns(
        self, 
        embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Compute quantum interference patterns."""
        batch_size, seq_len, embed_dim = embeddings.shape
        
        interference_scores = []
        for i in range(batch_size):
            batch_embeddings = embeddings[i]  # [seq_len, embed_dim]
            
            # Compute interference as cross-correlation between positions
            interference_score = 0.0
            for j in range(seq_len):
                for k in range(j + 1, seq_len):
                    pos_j = batch_embeddings[j]
                    pos_k = batch_embeddings[k]
                    
                    # Interference as normalized dot product
                    interference = torch.dot(pos_j, pos_k) / (torch.norm(pos_j) * torch.norm(pos_k) + 1e-8)
                    interference_score += interference.item()
            
            # Normalize by number of position pairs
            num_pairs = seq_len * (seq_len - 1) // 2
            interference_scores.append(interference_score / num_pairs)
        
        interference_scores = np.array(interference_scores)
        
        return {
            'mean_interference_strength': float(np.mean(interference_scores)),
            'std_interference_strength': float(np.std(interference_scores)),
            'interference_entropy': float(stats.entropy(np.histogram(interference_scores, bins=20)[0] + 1e-8))
        }
    
    def _analyze_interference_matrix(
        self, 
        interference_matrix: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze interference matrix properties."""
        matrix_np = interference_matrix.detach().cpu().numpy()
        
        # Matrix properties
        matrix_norm = np.linalg.norm(matrix_np)
        matrix_rank = np.linalg.matrix_rank(matrix_np)
        
        # Eigenvalue analysis
        eigenvals, _ = np.linalg.eig(matrix_np)
        eigenvals = np.real(eigenvals)
        
        eigenval_entropy = stats.entropy(np.abs(eigenvals) + 1e-8)
        eigenval_stability = 1.0 - np.std(np.abs(eigenvals))
        
        return {
            'interference_matrix_norm': float(matrix_norm),
            'interference_matrix_rank': int(matrix_rank),
            'interference_eigenvalue_entropy': float(eigenval_entropy),
            'interference_eigenvalue_stability': float(eigenval_stability)
        }
    
    def analyze_superposition_evolution(
        self,
        embeddings_sequence: List[torch.Tensor],
        time_steps: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze superposition state evolution over time.
        
        Args:
            embeddings_sequence: List of embeddings at different time steps
            time_steps: Optional time steps for temporal analysis
            
        Returns:
            Dictionary of evolution analysis metrics
        """
        if len(embeddings_sequence) < 2:
            return {}
        
        evolution_metrics = {}
        
        # Analyze superposition strength evolution
        superposition_strengths = []
        for embeddings in embeddings_sequence:
            strength_metrics = self._compute_superposition_strength(embeddings)
            superposition_strengths.append(strength_metrics['mean_superposition_strength'])
        
        superposition_strengths = np.array(superposition_strengths)
        
        # Evolution metrics
        evolution_stability = 1.0 - np.std(superposition_strengths)
        evolution_entropy = stats.entropy(np.abs(superposition_strengths) + 1e-8)
        
        # Detect quantum jumps (sudden large changes)
        if len(superposition_strengths) > 1:
            changes = np.abs(np.diff(superposition_strengths))
            jump_threshold = np.mean(changes) + 2 * np.std(changes)
            quantum_jumps = np.sum(changes > jump_threshold)
            evolution_metrics['superposition_quantum_jumps'] = int(quantum_jumps)
        
        evolution_metrics.update({
            'superposition_strength_evolution': superposition_strengths.tolist(),
            'superposition_evolution_stability': float(evolution_stability),
            'superposition_evolution_entropy': float(evolution_entropy)
        })
        
        # Time-dependent analysis if time steps provided
        if time_steps and len(time_steps) == len(embeddings_sequence):
            evolution_metrics['time_steps'] = time_steps
            evolution_metrics['temporal_analysis'] = True
        
        return evolution_metrics
    
    def generate_superposition_report(
        self,
        embeddings: torch.Tensor,
        superposition_matrix: Optional[torch.Tensor] = None,
        collapse_probabilities: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive superposition analysis report.
        
        Args:
            embeddings: Embedding vectors
            superposition_matrix: Optional superposition mixing matrix
            collapse_probabilities: Optional collapse probabilities
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing comprehensive superposition analysis
        """
        report = {}
        
        # Superposition state analysis
        report['superposition_states'] = self.analyze_superposition_states(
            embeddings, attention_mask, superposition_matrix
        )
        
        # Collapse mechanism analysis
        report['collapse_mechanisms'] = self.analyze_collapse_mechanisms(
            embeddings, collapse_probabilities, attention_mask
        )
        
        # Quantum interference analysis
        report['quantum_interference'] = self.analyze_quantum_interference(
            embeddings, None, attention_mask
        )
        
        # Summary statistics
        import time
        report['summary'] = {
            'total_samples': int(embeddings.numel()),
            'analysis_timestamp': str(time.time()),
            'embedding_shape': list(embeddings.shape)
        }
        
        return report
