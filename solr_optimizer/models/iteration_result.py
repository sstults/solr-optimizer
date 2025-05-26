"""
Iteration Result - Model class for experiment iteration results.

This module defines the IterationResult class that represents the outcome of
running queries with a specific configuration in an optimization experiment.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from solr_optimizer.models.query_config import QueryConfig


@dataclass
class QueryResult:
    """
    Result of a single query execution.

    Attributes:
        query: The query text
        documents: List of document IDs in result order
        scores: Dictionary of document ID to relevance score
        explain_info: Optional dictionary with Solr explain information
    """

    query: str
    documents: List[str]
    scores: Dict[str, float]
    explain_info: Optional[Dict[str, Any]] = None


@dataclass
class MetricResult:
    """
    Metric calculation result.

    Attributes:
        metric_name: Name of the metric (e.g., 'ndcg@10')
        value: Value of the metric
        per_query: Optional dictionary of query to metric value
    """

    metric_name: str
    value: float
    per_query: Dict[str, float] = field(default_factory=dict)


@dataclass
class IterationResult:
    """
    Result of an experiment iteration.

    Attributes:
        iteration_id: Unique identifier for this iteration
        experiment_id: ID of the parent experiment
        query_config: Configuration used for this iteration
        query_results: Dictionary mapping query to QueryResult
        metric_results: List of metric calculation results
        timestamp: When the iteration was run
        compared_to: Optional ID of iteration this was compared to
        metric_deltas: Optional dictionary of metric name to
                       change vs. compared_to
        notes: Optional notes or observations
    """

    iteration_id: str
    experiment_id: str
    query_config: QueryConfig
    query_results: Dict[str, QueryResult]
    metric_results: List[MetricResult]
    timestamp: datetime = field(default_factory=datetime.now)
    compared_to: Optional[str] = None
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    notes: Optional[str] = None

    def get_primary_metric(self) -> Optional[MetricResult]:
        """
        Get the primary metric result.

        Returns:
            The MetricResult for the primary metric, or None if not found
        """
        # By convention, the primary metric is the first one in the list
        return self.metric_results[0] if self.metric_results else None

    def get_metric_by_name(self, name: str) -> Optional[MetricResult]:
        """
        Get a metric result by name.

        Args:
            name: The name of the metric to find

        Returns:
            The MetricResult for the specified metric, or None if not found
        """
        for metric in self.metric_results:
            if metric.metric_name == name:
                return metric
        return None

    def summary_dict(self) -> Dict[str, Any]:
        """
        Generate a summary dictionary of the iteration result.

        Returns:
            Dictionary with key information about the iteration
        """
        return {
            "iteration_id": self.iteration_id,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp.isoformat(),
            "metrics": {metric.metric_name: metric.value for metric in self.metric_results},
            "metric_deltas": self.metric_deltas,
            "query_count": len(self.query_results),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the iteration result to a dictionary for serialization.

        Returns:
            Dictionary representation of the iteration result
        """
        return {
            "iteration_id": self.iteration_id,
            "experiment_id": self.experiment_id,
            "query_config": self.query_config.__dict__,
            "query_results": {
                query: {
                    "query": result.query,
                    "documents": result.documents,
                    "scores": result.scores,
                    "explain_info": result.explain_info
                }
                for query, result in self.query_results.items()
            },
            "metric_results": [
                {
                    "metric_name": metric.metric_name,
                    "value": metric.value,
                    "per_query": metric.per_query
                }
                for metric in self.metric_results
            ],
            "timestamp": self.timestamp.isoformat(),
            "compared_to": self.compared_to,
            "metric_deltas": self.metric_deltas,
            "notes": self.notes
        }
