"""
Embedding quality evaluator for QEmbed.

⚠️ CRITICAL: This evaluator integrates with existing QEmbed infrastructure
    and uses existing QuantumEmbeddings class for analysis.

Evaluates embedding quality with semantic similarity metrics.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from .base_evaluator import BaseEvaluator, EvaluationResult
from .evaluation_metrics import EvaluationMetrics

# ⚠️ CRITICAL: Import existing QuantumEmbeddings for integration
from qembed.core.quantum_embeddings import QuantumEmbeddings

class EmbeddingEvaluator(BaseEvaluator):
    """
    Evaluator for embedding quality assessment.
    
    Analyzes embedding characteristics including semantic similarity,
    quantum properties, and coherence metrics.
    
    ⚠️ CRITICAL: Extends BaseEvaluator and integrates with existing QuantumMetrics
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        device: Optional[str] = None,
        similarity_metrics: List[str] = None
    ):
        """
        Initialize embedding evaluator.
        
        Args:
            model: Model to extract embeddings from
            device: Device to run evaluation on
            similarity_metrics: List of similarity metrics to compute
        """
        super().__init__(model, device)
        
        # ⚠️ CRITICAL: Use existing QuantumMetrics instance from parent class
        self.metrics_calculator = EvaluationMetrics()
        
        # Default similarity metrics
        self.similarity_metrics = similarity_metrics or [
            'cosine', 'euclidean', 'manhattan', 'dot_product'
        ]
    
    def evaluate(
        self, 
        dataloader: torch.utils.data.DataLoader,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate embedding quality.
        
        Args:
            dataloader: DataLoader containing evaluation data
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with embedding metrics
        """
        all_embeddings = []
        all_labels = []
        all_uncertainties = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self._evaluate_batch_internal(batch)
                
                # Collect results
                all_embeddings.append(outputs['embeddings'])
                if 'labels' in outputs and outputs['labels'] is not None:
                    all_labels.append(outputs['labels'])
                if 'uncertainty' in outputs and outputs['uncertainty'] is not None:
                    all_uncertainties.append(outputs['uncertainty'])
        
        # Concatenate results
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0) if all_labels else None
        uncertainties = torch.cat(all_uncertainties, dim=0) if all_uncertainties else None
        
        # Compute embedding metrics
        metrics = self._compute_embedding_metrics(embeddings, labels)
        
        # Compute quantum metrics if available
        quantum_metrics = {}
        if uncertainties is not None:
            # ⚠️ CRITICAL: Use parent class method to avoid conflicts
            quantum_metrics = self.compute_quantum_metrics({'quantum_uncertainty': uncertainties})
        
        # Add embedding-specific quantum analysis
        if hasattr(self.model, 'embeddings') and isinstance(self.model.embeddings, QuantumEmbeddings):
            quantum_metrics.update(self._analyze_quantum_embeddings(embeddings))
        
        # Create result
        result = EvaluationResult(
            task_name="embedding_quality",
            model_name=self.model.__class__.__name__,
            metrics=metrics,
            quantum_metrics=quantum_metrics,
            metadata={
                'embedding_dim': embeddings.size(-1),
                'num_samples': len(embeddings),
                'similarity_metrics': self.similarity_metrics
            }
        )
        
        # Store result
        self.current_result = result
        self.results.append(result)
        
        return result
    
    def _evaluate_batch_internal(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate a single batch of data.
        
        ⚠️ CRITICAL: This method handles both quantum BERT and legacy embedding models
        """
        # Extract batch components
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get embeddings (use last hidden state)
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
            embeddings = outputs.hidden_states[-1]
        else:
            # Fallback for models that don't return hidden states
            embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Get uncertainty if available
        uncertainty = None
        if hasattr(outputs, 'quantum_uncertainty'):
            uncertainty = outputs.quantum_uncertainty
        
        return {
            'embeddings': embeddings,
            'labels': labels,
            'uncertainty': uncertainty
        }
    
    def _compute_embedding_metrics(
        self, 
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute embedding quality metrics."""
        # ⚠️ CRITICAL: Use metrics calculator to avoid duplicating functionality
        metrics = self.metrics_calculator.embedding_metrics(embeddings, labels)
        
        # Add additional embedding-specific metrics
        metrics.update(self._compute_similarity_metrics(embeddings))
        metrics.update(self._compute_diversity_metrics(embeddings))
        
        if labels is not None:
            metrics.update(self._compute_supervised_metrics(embeddings, labels))
        
        return metrics
    
    def _compute_similarity_metrics(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Compute various similarity metrics between embeddings."""
        # Normalize embeddings for similarity computation
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(-2, -1))
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(similarity_matrix.size(-1), device=similarity_matrix.device)
        off_diagonal_similarities = similarity_matrix[~mask.bool()]
        
        metrics = {}
        
        if 'cosine' in self.similarity_metrics:
            metrics['mean_cosine_similarity'] = off_diagonal_similarities.mean().item()
            metrics['std_cosine_similarity'] = off_diagonal_similarities.std().item()
        
        if 'dot_product' in self.similarity_metrics:
            # Dot product without normalization
            dot_matrix = torch.matmul(embeddings, embeddings.transpose(-2, -1))
            dot_off_diagonal = dot_matrix[~mask.bool()]
            metrics['mean_dot_product'] = dot_off_diagonal.mean().item()
            metrics['std_dot_product'] = dot_off_diagonal.std().item()
        
        if 'euclidean' in self.similarity_metrics:
            # Compute pairwise Euclidean distances
            dist_matrix = torch.cdist(embeddings, embeddings, p=2)
            dist_off_diagonal = dist_matrix[~mask.bool()]
            metrics['mean_euclidean_distance'] = dist_off_diagonal.mean().item()
            metrics['std_euclidean_distance'] = dist_off_diagonal.std().item()
        
        if 'manhattan' in self.similarity_metrics:
            # Compute pairwise Manhattan distances
            manhattan_matrix = torch.cdist(embeddings, embeddings, p=1)
            manhattan_off_diagonal = manhattan_matrix[~mask.bool()]
            metrics['mean_manhattan_distance'] = manhattan_off_diagonal.mean().item()
            metrics['std_manhattan_distance'] = manhattan_off_diagonal.std().item()
        
        return metrics
    
    def _compute_diversity_metrics(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Compute diversity and coverage metrics."""
        # Compute embedding norms
        norms = torch.norm(embeddings, p=2, dim=-1)
        
        # Compute variance across different dimensions
        variance_across_samples = torch.var(embeddings, dim=0)  # [embed_dim]
        variance_across_features = torch.var(embeddings, dim=1)  # [batch_size]
        
        # Compute coverage (how much of the embedding space is utilized)
        mean_embedding = torch.mean(embeddings, dim=0)
        coverage = torch.norm(mean_embedding) / (torch.norm(embeddings, p=2, dim=0).mean() + 1e-8)
        
        return {
            'mean_embedding_norm': norms.mean().item(),
            'std_embedding_norm': norms.std().item(),
            'mean_variance_across_samples': variance_across_samples.mean().item(),
            'mean_variance_across_features': variance_across_features.mean().item(),
            'embedding_coverage': coverage.item()
        }
    
    def _compute_supervised_metrics(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute supervised metrics if labels are available."""
        # Convert to numpy for sklearn metrics
        embeddings_np = embeddings.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Compute intra-class and inter-class distances
        unique_labels = np.unique(labels_np)
        
        intra_class_distances = []
        inter_class_distances = []
        
        for label in unique_labels:
            # Intra-class distances
            class_mask = labels_np == label
            class_embeddings = embeddings_np[class_mask]
            
            if len(class_embeddings) > 1:
                # Compute pairwise distances within class
                class_distances = []
                for i in range(len(class_embeddings)):
                    for j in range(i + 1, len(class_embeddings)):
                        dist = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
                        class_distances.append(dist)
                
                if class_distances:
                    intra_class_distances.extend(class_distances)
            
            # Inter-class distances (to class centroid)
            class_centroid = np.mean(class_embeddings, axis=0)
            other_embeddings = embeddings_np[~class_mask]
            
            if len(other_embeddings) > 0:
                for other_emb in other_embeddings:
                    dist = np.linalg.norm(other_emb - class_centroid)
                    inter_class_distances.append(dist)
        
        metrics = {}
        
        if intra_class_distances:
            metrics['mean_intra_class_distance'] = np.mean(intra_class_distances)
            metrics['std_intra_class_distance'] = np.std(intra_class_distances)
        
        if inter_class_distances:
            metrics['mean_inter_class_distance'] = np.mean(inter_class_distances)
            metrics['std_inter_class_distance'] = np.std(inter_class_distances)
        
        # Compute separation ratio
        if intra_class_distances and inter_class_distances:
            separation_ratio = np.mean(inter_class_distances) / (np.mean(intra_class_distances) + 1e-8)
            metrics['class_separation_ratio'] = separation_ratio
        
        return metrics
    
    def _analyze_quantum_embeddings(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Analyze quantum-specific properties of embeddings."""
        # ⚠️ CRITICAL: Use existing QuantumMetrics instance for quantum analysis
        quantum_metrics = {}
        
        # Analyze superposition characteristics
        if hasattr(self.model, 'embeddings') and isinstance(self.model.embeddings, QuantumEmbeddings):
            # Use existing quantum embedding analysis methods
            quantum_metrics.update(self._analyze_superposition(embeddings))
        
        return quantum_metrics
    
    def evaluate_single_sample(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample for real-time inference.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary with embeddings and analysis
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare input
            inputs = {'input_ids': input_ids.to(self.device)}
            if attention_mask is not None:
                inputs['attention_mask'] = attention_mask.to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            
            # Get embeddings
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                embeddings = outputs.hidden_states[-1]
            else:
                embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Get uncertainty if available
            uncertainty = None
            if hasattr(outputs, 'quantum_uncertainty'):
                uncertainty = outputs.quantum_uncertainty
            
            # Compute single-sample metrics
            sample_metrics = self._compute_embedding_metrics(embeddings)
            
            return {
                'embeddings': embeddings.cpu().numpy(),
                'uncertainty': uncertainty.cpu().numpy() if uncertainty is not None else None,
                'metrics': sample_metrics
            }
    
    def get_embedding_clusters(
        self, 
        embeddings: torch.Tensor, 
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        Perform clustering analysis on embeddings.
        
        Args:
            embeddings: Embedding vectors
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with clustering results
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            
            # Reduce dimensionality for clustering
            pca = PCA(n_components=min(50, embeddings.size(-1)))
            embeddings_2d = pca.fit_transform(embeddings.cpu().numpy())
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_2d)
            
            # Compute cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_embeddings = embeddings_2d[cluster_mask]
                
                cluster_stats[f'cluster_{i}'] = {
                    'size': int(cluster_mask.sum()),
                    'centroid': kmeans.cluster_centers_[i].tolist(),
                    'variance': float(np.var(cluster_embeddings, axis=0).mean())
                }
            
            return {
                'cluster_labels': cluster_labels.tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'cluster_statistics': cluster_stats,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
            }
            
        except ImportError:
            return {'error': 'sklearn not available for clustering analysis'}
