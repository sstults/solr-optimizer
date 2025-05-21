"""
Metrics Agent - Interface for calculating relevance metrics.

This module defines the MetricsAgent interface which is responsible for
calculating relevance metrics based on query results and relevance judgments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from solr_optimizer.models.iteration_result import MetricResult


class MetricsAgent(ABC):
    """
    Agent responsible for calculating relevance metrics based on query results
    and judgments.
    """

    @abstractmethod
    def calculate_metric(
        self,
        metric_name: str,
        results: List[str],
        judgments: Dict[str, int],
        depth: int,
    ) -> float:
        """
        Calculate a single relevance metric for a query.

        Args:
            metric_name: Name of the metric to calculate
                         (e.g., 'ndcg', 'precision')
            results: List of document IDs in result order
            judgments: Dictionary of document ID to relevance judgment
            depth: Depth at which to calculate the metric
                   (e.g., 10 for NDCG@10)

        Returns:
            The calculated metric value
        """
        pass

    @abstractmethod
    def calculate_metrics(
        self,
        metrics: List[str],
        results_by_query: Dict[str, List[str]],
        judgments_by_query: Dict[str, Dict[str, int]],
        depth: int,
    ) -> List[MetricResult]:
        """
        Calculate multiple metrics across a set of queries.

        Args:
            metrics: List of metrics to calculate
            results_by_query: Dictionary mapping query to list of result
                              document IDs
            judgments_by_query: Dictionary mapping query to document judgments
            depth: Depth at which to calculate metrics

        Returns:
            List of MetricResult objects
        """
        pass

    @abstractmethod
    def normalize_judgments(
        self,
        judgments: Dict[str, Dict[str, Any]],
        scale: Optional[Dict[Any, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Normalize relevance judgments to a common scale.

        Args:
            judgments: Dictionary mapping query to document judgments
            scale: Optional mapping of input values to normalized values

        Returns:
            Normalized judgments
        """
        pass

    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """
        Get a list of metrics supported by this agent.

        Returns:
            List of supported metric names
        """
        pass
