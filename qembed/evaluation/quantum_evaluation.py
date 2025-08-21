"""
Quantum-specific evaluation metrics for QEmbed.

⚠️ CRITICAL: This class extends the existing QuantumMetrics class
    and integrates with existing QuantumEmbeddings infrastructure.

Provides metrics for analyzing quantum behavior including
coherence, superposition quality, and entanglement effects.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

# ⚠️ CRITICAL: Extend existing QuantumMetrics instead of duplicating
# Note: Use absolute imports to avoid circular dependency issues
from qembed.utils.metrics import QuantumMetrics as BaseQuantumMetrics
from qembed.core.quantum_embeddings import QuantumEmbeddings

class QuantumEvaluation(BaseQuantumMetrics):
    """
    Extended quantum evaluation metrics that build upon existing QuantumMetrics.
    
    ⚠️ CRITICAL: Extends existing functionality and integrates with
    QuantumEmbeddings for superposition analysis.
    
    ⚠️ CRITICAL: Uses Phase 2 integration patterns for consistency
    """
    
    def __init__(self):
        """Initialize quantum evaluation metrics."""
        super().__init__()
    
    @staticmethod
    def coherence_metrics(
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute quantum coherence metrics.
        
        Args:
            embeddings: Embedding vectors [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary of coherence metrics
        """
        if attention_mask is not None:
            # Apply attention mask
            embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        # Compute coherence as normalized variance
        variance = torch.var(embeddings, dim=1)  # [batch_size, embed_dim]
        mean_variance = torch.mean(variance, dim=1)  # [batch_size]
        
        # Normalize by embedding norm
        embedding_norms = torch.norm(embeddings, p=2, dim=-1)  # [batch_size, seq_len]
        mean_norms = torch.mean(embedding_norms, dim=1)  # [batch_size]
        
        coherence = mean_variance / (mean_norms ** 2 + 1e-8)
        
        return {
            'mean_coherence': coherence.mean().item(),
            'std_coherence': coherence.std().item(),
            'coherence_entropy': QuantumEvaluation._compute_entropy(coherence)
        }
    
    @staticmethod
    def superposition_quality(
        state_embeddings: torch.Tensor,
        superposition_matrix: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate superposition quality.
        
        Args:
            state_embeddings: Base state embeddings [vocab_size, num_states, embed_dim]
            superposition_matrix: Superposition mixing matrix [num_states, num_states]
            
        Returns:
            Dictionary of superposition quality metrics
        """
        # Compute superposition matrix properties
        eigenvals, eigenvecs = torch.linalg.eig(superposition_matrix)
        eigenvals = eigenvals.real
        
        # Superposition quality based on eigenvalue distribution
        eigenval_entropy = entropy(np.abs(eigenvals.cpu().numpy()) + 1e-8)
        
        # Compute mixing efficiency
        mixing_efficiency = torch.norm(superposition_matrix - torch.eye(superposition_matrix.size(0))) / torch.norm(torch.eye(superposition_matrix.size(0)))
        
        # State diversity
        state_diversity = torch.std(state_embeddings, dim=1).mean()
        
        return {
            'eigenvalue_entropy': eigenval_entropy,
            'mixing_efficiency': mixing_efficiency.item(),
            'state_diversity': state_diversity.item(),
            'superposition_rank': torch.linalg.matrix_rank(superposition_matrix).item()
        }
    
    @staticmethod
    def entanglement_quantification(
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Quantify entanglement effects.
        
        Args:
            embeddings: Embedding vectors [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary of entanglement metrics
        """
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Compute pairwise correlations
        correlations = []
        for i in range(batch_size):
            batch_embeddings = embeddings[i]  # [seq_len, embed_dim]
            
            # Compute correlation matrix
            corr_matrix = torch.corrcoef(batch_embeddings.T)  # [embed_dim, embed_dim]
            
            # Remove diagonal and compute average correlation
            mask = torch.eye(embed_dim, device=corr_matrix.device)
            off_diag_corr = corr_matrix[~mask.bool()]
            correlations.append(off_diag_corr.mean().item())
        
        correlations = np.array(correlations)
        
        # Entanglement measures
        mean_correlation = np.mean(correlations)
        correlation_entropy = entropy(np.abs(correlations) + 1e-8)
        
        # Compute mutual information between different embedding dimensions
        mutual_info_scores = []
        for i in range(min(10, embed_dim)):  # Sample dimensions for efficiency
            for j in range(i+1, min(10, embed_dim)):
                dim_i = embeddings[:, :, i].flatten().cpu().numpy()
                dim_j = embeddings[:, :, j].flatten().cpu().numpy()
                
                # Discretize for mutual information computation
                dim_i_binned = np.digitize(dim_i, bins=np.linspace(dim_i.min(), dim_i.max(), 10))
                dim_j_binned = np.digitize(dim_j, bins=np.linspace(dim_j.min(), dim_j.max(), 10))
                
                mi_score = mutual_info_score(dim_i_binned, dim_j_binned)
                mutual_info_scores.append(mi_score)
        
        return {
            'mean_correlation': mean_correlation,
            'correlation_entropy': correlation_entropy,
            'mean_mutual_info': np.mean(mutual_info_scores),
            'entanglement_strength': mean_correlation * np.mean(mutual_info_scores)
        }
    
    @staticmethod
    def quantum_state_evolution(
        embeddings_sequence: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Analyze quantum state evolution over time.
        
        Args:
            embeddings_sequence: List of embeddings at different time steps
            
        Returns:
            Dictionary of evolution metrics
        """
        if len(embeddings_sequence) < 2:
            return {}
        
        # Compute state transitions
        transitions = []
        for i in range(len(embeddings_sequence) - 1):
            current = embeddings_sequence[i]
            next_state = embeddings_sequence[i + 1]
            
            # Compute transition probability
            transition_prob = torch.cosine_similarity(
                current.flatten(), next_state.flatten(), dim=0
            )
            transitions.append(transition_prob.item())
        
        transitions = np.array(transitions)
        
        # Evolution metrics
        evolution_stability = np.std(transitions)
        evolution_entropy = entropy(np.abs(transitions) + 1e-8)
        
        # Detect quantum jumps (sudden large changes)
        jump_threshold = np.mean(transitions) + 2 * np.std(transitions)
        quantum_jumps = np.sum(transitions > jump_threshold)
        
        return {
            'evolution_stability': evolution_stability,
            'evolution_entropy': evolution_entropy,
            'quantum_jumps': quantum_jumps,
            'mean_transition_prob': np.mean(transitions),
            'transition_consistency': 1.0 - evolution_stability
        }
    
    @staticmethod
    def _compute_entropy(values: torch.Tensor) -> float:
        """Compute entropy of a tensor."""
        values_np = values.detach().cpu().numpy()
        return entropy(np.abs(values_np) + 1e-8)
    
    @staticmethod
    def quantum_interference_analysis(
        embeddings: torch.Tensor,
        interference_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze quantum interference effects.
        
        Args:
            embeddings: Embedding vectors [batch_size, seq_len, embed_dim]
            interference_matrix: Optional interference matrix
            
        Returns:
            Dictionary of interference metrics
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Compute interference patterns
        interference_patterns = []
        
        for i in range(batch_size):
            batch_embeddings = embeddings[i]  # [seq_len, embed_dim]
            
            # Compute interference as cross-correlation between different positions
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
            interference_patterns.append(interference_score / num_pairs)
        
        interference_patterns = np.array(interference_patterns)
        
        # Interference metrics
        mean_interference = np.mean(interference_patterns)
        interference_entropy = entropy(np.abs(interference_patterns) + 1e-8)
        
        # Compute interference stability
        interference_stability = 1.0 - np.std(interference_patterns)
        
        return {
            'mean_interference': mean_interference,
            'interference_entropy': interference_entropy,
            'interference_stability': interference_stability,
            'interference_patterns': interference_patterns.tolist()
        }
    
    @staticmethod
    def quantum_decoherence_analysis(
        embeddings: torch.Tensor,
        time_steps: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Analyze quantum decoherence effects.
        
        Args:
            embeddings: Embedding vectors [batch_size, seq_len, embed_dim]
            time_steps: Optional time steps for temporal analysis
            
        Returns:
            Dictionary of decoherence metrics
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Compute decoherence as loss of quantum coherence over sequence
        decoherence_scores = []
        
        for i in range(batch_size):
            batch_embeddings = embeddings[i]  # [seq_len, embed_dim]
            
            # Compute coherence at each position
            position_coherences = []
            for j in range(seq_len):
                pos_embedding = batch_embeddings[j]
                
                # Coherence as normalized variance across embedding dimensions
                coherence = torch.var(pos_embedding) / (torch.norm(pos_embedding) ** 2 + 1e-8)
                position_coherences.append(coherence.item())
            
            # Compute decoherence as decrease in coherence over sequence
            if len(position_coherences) > 1:
                initial_coherence = position_coherences[0]
                final_coherence = position_coherences[-1]
                decoherence_rate = (initial_coherence - final_coherence) / initial_coherence if initial_coherence > 0 else 0
                decoherence_scores.append(decoherence_rate)
        
        if not decoherence_scores:
            return {}
        
        decoherence_scores = np.array(decoherence_scores)
        
        return {
            'mean_decoherence_rate': np.mean(decoherence_scores),
            'decoherence_entropy': entropy(np.abs(decoherence_scores) + 1e-8),
            'decoherence_stability': 1.0 - np.std(decoherence_scores),
            'decoherence_patterns': decoherence_scores.tolist()
        }
