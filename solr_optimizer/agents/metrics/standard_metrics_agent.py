"""
Standard Metrics Agent - Implementation of MetricsAgent for common IR metrics.

This module provides a concrete implementation of the MetricsAgent interface for
calculating common Information Retrieval metrics such as NDCG, Precision, and Recall.
"""

import math
import logging
from typing import Dict, List, Optional, Any

from solr_optimizer.agents.metrics.metrics_agent import MetricsAgent
from solr_optimizer.models.iteration_result import MetricResult

logger = logging.getLogger(__name__)


class StandardMetricsAgent(MetricsAgent):
    """
    Implementation of MetricsAgent for calculating common IR metrics.
    """
    
    def __init__(self):
        """Initialize the StandardMetricsAgent."""
        self.supported_metrics = {"ndcg", "dcg", "precision", "recall", "mrr", "err"}
    
    def calculate_metric(self, metric_name: str, results: List[str], 
                        judgments: Dict[str, int], depth: int) -> float:
        """
        Calculate a single relevance metric for a query.
        
        Args:
            metric_name: Name of the metric to calculate (e.g., 'ndcg', 'precision')
            results: List of document IDs in result order
            judgments: Dictionary of document ID to relevance judgment
            depth: Depth at which to calculate the metric (e.g., 10 for NDCG@10)
            
        Returns:
            The calculated metric value
        
        Raises:
            ValueError: If the metric is not supported
        """
        metric_name = metric_name.lower()
        
        if metric_name not in self.supported_metrics:
            raise ValueError(f"Unsupported metric: {metric_name}")
        
        # Truncate results to specified depth
        results = results[:depth]
        
        if metric_name == "ndcg":
            return self._calculate_ndcg(results, judgments, depth)
        elif metric_name == "dcg":
            return self._calculate_dcg(results, judgments, depth)
        elif metric_name == "precision":
            return self._calculate_precision(results, judgments)
        elif metric_name == "recall":
            return self._calculate_recall(results, judgments)
        elif metric_name == "mrr":
            return self._calculate_mrr(results, judgments)
        elif metric_name == "err":
            return self._calculate_err(results, judgments, depth)
        else:
            # This should never happen due to the check above, but just in case
            raise ValueError(f"Metric implementation missing: {metric_name}")
    
    def calculate_metrics(self, metrics: List[str], results_by_query: Dict[str, List[str]],
                         judgments_by_query: Dict[str, Dict[str, int]], 
                         depth: int) -> List[MetricResult]:
        """
        Calculate multiple metrics across a set of queries.
        
        Args:
            metrics: List of metrics to calculate
            results_by_query: Dictionary mapping query to list of result document IDs
            judgments_by_query: Dictionary mapping query to document judgments
            depth: Depth at which to calculate metrics
            
        Returns:
            List of MetricResult objects
        """
        metric_results = []
        
        for metric_name in metrics:
            per_query_values = {}
            total_value = 0.0
            query_count = 0
            
            for query, results in results_by_query.items():
                # Skip queries without judgments
                if query not in judgments_by_query:
                    logger.warning(f"Query '{query}' has no judgments, skipping metric calculation")
                    continue
                
                try:
                    value = self.calculate_metric(
                        metric_name, results, judgments_by_query[query], depth
                    )
                    per_query_values[query] = value
                    total_value += value
                    query_count += 1
                except Exception as e:
                    logger.error(f"Error calculating {metric_name} for query '{query}': {str(e)}")
            
            # Calculate mean across all queries
            mean_value = total_value / query_count if query_count > 0 else 0.0
            
            metric_results.append(
                MetricResult(
                    metric_name=f"{metric_name}@{depth}" if '@' not in metric_name else metric_name,
                    value=mean_value,
                    per_query=per_query_values
                )
            )
        
        return metric_results
    
    def normalize_judgments(self, judgments: Dict[str, Dict[str, Any]], 
                          scale: Optional[Dict[Any, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Normalize relevance judgments to a common scale.
        
        Args:
            judgments: Dictionary mapping query to document judgments
            scale: Optional mapping of input values to normalized values
            
        Returns:
            Normalized judgments
        """
        normalized = {}
        
        if scale is None:
            # Default scale: keep numeric values as is, convert non-numeric to binary
            for query, docs in judgments.items():
                normalized[query] = {}
                for doc_id, judgment in docs.items():
                    try:
                        # Try to convert to float
                        normalized[query][doc_id] = float(judgment)
                    except (ValueError, TypeError):
                        # If not numeric, treat as binary (relevant=1, not relevant=0)
                        normalized[query][doc_id] = 1.0 if judgment else 0.0
        else:
            # Apply custom scale
            for query, docs in judgments.items():
                normalized[query] = {}
                for doc_id, judgment in docs.items():
                    if judgment in scale:
                        normalized[query][doc_id] = scale[judgment]
                    else:
                        logger.warning(
                            f"Judgment value '{judgment}' not found in scale, defaulting to 0.0"
                        )
                        normalized[query][doc_id] = 0.0
        
        return normalized
    
    def get_supported_metrics(self) -> List[str]:
        """
        Get a list of metrics supported by this agent.
        
        Returns:
            List of supported metric names
        """
        return list(self.supported_metrics)
    
    def _calculate_dcg(self, results: List[str], judgments: Dict[str, int], depth: int) -> float:
        """
        Calculate Discounted Cumulative Gain (DCG).
        
        Args:
            results: List of document IDs in result order
            judgments: Dictionary of document ID to relevance judgment
            depth: Depth at which to calculate DCG
            
        Returns:
            DCG value
        """
        dcg = 0.0
        
        for i, doc_id in enumerate(results[:depth]):
            # Get relevance grade (default to 0 if not in judgments)
            rel = judgments.get(doc_id, 0)
            # Calculate gain using log2(i+2) to handle 0-indexed results
            # Most common formula: gain = (2^rel - 1) / log2(i+2)
            if rel > 0:  # Skip irrelevant documents for efficiency
                position = i + 1  # Convert to 1-indexed for formula
                dcg += (2 ** rel - 1) / math.log2(position + 1)
        
        return dcg
    
    def _calculate_ndcg(self, results: List[str], judgments: Dict[str, int], depth: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG).
        
        Args:
            results: List of document IDs in result order
            judgments: Dictionary of document ID to relevance judgment
            depth: Depth at which to calculate NDCG
            
        Returns:
            NDCG value
        """
        dcg = self._calculate_dcg(results, judgments, depth)
        
        # Calculate ideal DCG (IDCG) by sorting documents by relevance
        relevant_docs = [
            (doc_id, judgments[doc_id])
            for doc_id in judgments
            if judgments[doc_id] > 0
        ]
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Create ideal results list
        ideal_results = [doc_id for doc_id, _ in relevant_docs]
        idcg = self._calculate_dcg(ideal_results, judgments, depth)
        
        # Avoid division by zero
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _calculate_precision(self, results: List[str], judgments: Dict[str, int]) -> float:
        """
        Calculate Precision.
        
        Args:
            results: List of document IDs in result order
            judgments: Dictionary of document ID to relevance judgment
            
        Returns:
            Precision value
        """
        if not results:
            return 0.0
        
        # Count relevant documents in results
        relevant_count = sum(1 for doc_id in results if judgments.get(doc_id, 0) > 0)
        return relevant_count / len(results)
    
    def _calculate_recall(self, results: List[str], judgments: Dict[str, int]) -> float:
        """
        Calculate Recall.
        
        Args:
            results: List of document IDs in result order
            judgments: Dictionary of document ID to relevance judgment
            
        Returns:
            Recall value
        """
        # Count total relevant documents in judgments
        total_relevant = sum(1 for rel in judgments.values() if rel > 0)
        
        if total_relevant == 0:
            return 0.0
        
        # Count relevant documents in results
        relevant_in_results = sum(1 for doc_id in results if judgments.get(doc_id, 0) > 0)
        return relevant_in_results / total_relevant
    
    def _calculate_mrr(self, results: List[str], judgments: Dict[str, int]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            results: List of document IDs in result order
            judgments: Dictionary of document ID to relevance judgment
            
        Returns:
            MRR value
        """
        # Find the first relevant document
        for i, doc_id in enumerate(results):
            if judgments.get(doc_id, 0) > 0:
                # Return reciprocal of first relevant document rank (1-indexed)
                return 1.0 / (i + 1)
        
        # No relevant documents found
        return 0.0
    
    def _calculate_err(self, results: List[str], judgments: Dict[str, int], depth: int) -> float:
        """
        Calculate Expected Reciprocal Rank (ERR).
        
        Args:
            results: List of document IDs in result order
            judgments: Dictionary of document ID to relevance judgment
            depth: Depth at which to calculate ERR
            
        Returns:
            ERR value
        """
        # Simplified ERR implementation
        err = 0.0
        p_not_satisfied = 1.0  # Probability user not satisfied with previous results
        
        max_grade = max(judgments.values()) if judgments else 0
        
        if max_grade == 0:
            return 0.0
        
        for i, doc_id in enumerate(results[:depth]):
            # Get relevance grade (default to 0 if not in judgments)
            grade = judgments.get(doc_id, 0)
            
            # Convert grade to probability of relevance (0 to 1)
            p_relevant = grade / max_grade
            
            # Update ERR
            position = i + 1  # Convert to 1-indexed for formula
            err += p_not_satisfied * p_relevant / position
            
            # Update probability for next iteration
            p_not_satisfied *= (1 - p_relevant)
        
        return err
