"""
Experiment Manager - Central coordinator for the Solr Optimizer framework.

This module defines the ExperimentManager interface which orchestrates the experiment workflow
and manages communication between all agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.query_config import QueryConfig
from solr_optimizer.models.iteration_result import IterationResult


class ExperimentManager(ABC):
    """
    Central coordinator that orchestrates the experiment workflow and manages communication
    between all agents in the system.
    """

    @abstractmethod
    def setup_experiment(self, config: ExperimentConfig) -> str:
        """
        Set up a new experiment with the provided configuration.

        Args:
            config: The experiment configuration including corpus, queries, and judgments

        Returns:
            The ID of the created experiment
        """
        pass

    @abstractmethod
    def run_iteration(
        self, experiment_id: str, query_config: QueryConfig
    ) -> IterationResult:
        """
        Run a single iteration with the given query configuration.

        Args:
            experiment_id: The ID of the experiment to run the iteration for
            query_config: The query configuration to test

        Returns:
            The result of the iteration
        """
        pass

    @abstractmethod
    def compare_iterations(
        self, experiment_id: str, iteration_id1: str, iteration_id2: str
    ) -> Dict:
        """
        Compare two iterations of the same experiment.

        Args:
            experiment_id: The experiment ID
            iteration_id1: The first iteration ID to compare
            iteration_id2: The second iteration ID to compare

        Returns:
            The comparison result as a dictionary
        """
        pass

    @abstractmethod
    def get_iteration_history(self, experiment_id: str) -> List[IterationResult]:
        """
        Get the history of iterations for an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            A list of iteration results
        """
        pass

    @abstractmethod
    def get_current_state(self, experiment_id: str) -> Optional[IterationResult]:
        """
        Get the current state of an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            The current iteration result or None if experiment does not exist
        """
        pass
