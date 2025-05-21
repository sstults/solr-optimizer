"""
Experiment Configuration - Model class for experiment setup.

This module defines the ExperimentConfig class that holds the configuration
for a Solr query optimization experiment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """
    Configuration for an experiment, including corpus, queries, judgments,
    and metrics.

    Attributes:
        experiment_id: Unique identifier for the experiment
        corpus: Name of the Solr collection/core to query
        queries: List of test queries to optimize
        judgments: Nested dictionary mapping query -> document_id ->
                   relevance_score
        primary_metric: Main metric to optimize for
                        (e.g., 'ndcg', 'precision', 'recall')
        metric_depth: Depth at which to calculate metrics
                      (e.g., 10 for NDCG@10)
        secondary_metrics: Optional list of additional metrics to track
        description: Optional description of the experiment
    """

    corpus: str
    queries: List[str]
    judgments: Dict[str, Dict[str, int]]
    primary_metric: str
    metric_depth: int
    experiment_id: Optional[str] = None
    secondary_metrics: List[str] = field(default_factory=list)
    description: Optional[str] = None

    def __post_init__(self):
        """Validate the experiment configuration."""
        # Ensure all queries have judgments
        missing_queries = [q for q in self.queries if q not in self.judgments]
        if missing_queries:
            raise ValueError(
                f"Missing judgments for queries: {', '.join(missing_queries)}"
            )

        # Validate primary metric
        valid_metrics = {"ndcg", "precision", "recall", "mrr", "err", "dcg"}
        if self.primary_metric.lower() not in valid_metrics:
            raise ValueError(
                f"Invalid primary_metric: '{self.primary_metric}'. "
                f"Must be one of: {', '.join(valid_metrics)}"
            )

        # Validate metric depth
        if self.metric_depth <= 0:
            raise ValueError(f"metric_depth must be positive, got "
                             f"{self.metric_depth}")
