"""
Dummy Query Tuning Agent - Basic implementation for testing purposes.

This module provides a concrete implementation of the QueryTuningAgent interface
with minimal functionality for testing and development purposes.
"""

from typing import Any, Dict

from solr_optimizer.agents.query.query_tuning_agent import QueryTuningAgent
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.iteration_result import IterationResult
from solr_optimizer.models.query_config import QueryConfig


class DummyQueryTuningAgent(QueryTuningAgent):
    """
    A basic implementation of QueryTuningAgent for testing purposes.

    This agent provides minimal implementations of all required methods
    that allow the system to run without actual query optimization logic.
    """

    def analyze_schema(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Solr schema to identify potentially useful fields.

        Args:
            schema_info: Schema information from Solr

        Returns:
            Empty dictionary (no analysis performed)
        """
        return {}

    def generate_initial_config(self, experiment_config: ExperimentConfig, schema_info: Dict[str, Any]) -> QueryConfig:
        """
        Generate an initial query configuration.

        Args:
            experiment_config: The experiment configuration
            schema_info: Schema information from Solr

        Returns:
            Empty QueryConfig object
        """
        return QueryConfig(additional_params={})

    def suggest_next_config(self, previous_result: IterationResult, schema_info: Dict[str, Any]) -> QueryConfig:
        """
        Suggest a new query configuration based on previous results.

        Args:
            previous_result: The results of the previous iteration
            schema_info: Schema information from Solr

        Returns:
            Empty QueryConfig object
        """
        return QueryConfig(additional_params={})

    def adjust_parameters(self, result: IterationResult, target_metric: str, direction: str) -> QueryConfig:
        """
        Adjust specific parameters to improve a target metric.

        Args:
            result: The results of the previous iteration
            target_metric: The metric to optimize (e.g., 'ndcg@10')
            direction: Either 'increase' or 'decrease'

        Returns:
            Empty QueryConfig object
        """
        return QueryConfig(additional_params={})
