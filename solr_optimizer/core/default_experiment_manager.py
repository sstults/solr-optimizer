"""
Default Experiment Manager - Basic implementation of the ExperimentManager
interface.

This module provides a concrete implementation of the ExperimentManager
interface that coordinates the workflow between different agents.
"""

import logging
import uuid
from typing import Dict, List

from solr_optimizer.agents.comparison.comparison_agent import ComparisonAgent
from solr_optimizer.agents.logging.logging_agent import LoggingAgent
from solr_optimizer.agents.metrics.metrics_agent import MetricsAgent
from solr_optimizer.agents.query.query_tuning_agent import QueryTuningAgent
from solr_optimizer.agents.solr.solr_execution_agent import SolrExecutionAgent
from solr_optimizer.core.experiment_manager import ExperimentManager
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.iteration_result import (
    IterationResult,
    QueryResult,
)
from solr_optimizer.models.query_config import QueryConfig

logger = logging.getLogger(__name__)


class DefaultExperimentManager(ExperimentManager):
    """
    Default implementation of the ExperimentManager interface.
    """

    def __init__(
        self,
        query_tuning_agent: QueryTuningAgent,
        solr_execution_agent: SolrExecutionAgent,
        metrics_agent: MetricsAgent,
        logging_agent: LoggingAgent,
        comparison_agent: ComparisonAgent,
    ):
        """
        Initialize the DefaultExperimentManager with required agents.

        Args:
            query_tuning_agent: Agent for generating query configurations
            solr_execution_agent: Agent for executing Solr queries
            metrics_agent: Agent for calculating relevance metrics
            logging_agent: Agent for logging experiment history
            comparison_agent: Agent for comparing iteration results
        """
        self.query_tuning_agent = query_tuning_agent
        self.solr_execution_agent = solr_execution_agent
        self.metrics_agent = metrics_agent
        self.logging_agent = logging_agent
        self.comparison_agent = comparison_agent

    def setup_experiment(self, config: ExperimentConfig) -> str:
        """
        Set up a new experiment with the provided configuration.

        Args:
            config: The experiment configuration including corpus, queries,
                    and judgments

        Returns:
            The ID of the created experiment
        """
        # Generate ID if not provided
        if not config.experiment_id:
            config.experiment_id = f"exp-{uuid.uuid4().hex[:8]}"

        # Validate configuration
        # (ExperimentConfig's __post_init__ already handles basic validation)

        # Save experiment configuration
        success = self.logging_agent.save_experiment(config)

        if not success:
            raise RuntimeError(f"Failed to save experiment: " f"{config.experiment_id}")

        logger.info(f"Created new experiment: {config.experiment_id}")
        return config.experiment_id

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
        # Get experiment configuration
        experiment = self.logging_agent.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Generate iteration ID if not provided
        iteration_id = query_config.iteration_id or f"iter-{uuid.uuid4().hex[:8]}"
        query_config.iteration_id = iteration_id

        logger.info(
            f"Running iteration {iteration_id} for experiment " f"{experiment_id}"
        )

        # Execute queries
        query_results_dict = self.solr_execution_agent.execute_queries(
            experiment.corpus, experiment.queries, query_config
        )

        # Convert query results to the expected format for metrics calculation
        results_by_query = {}
        query_results = {}

        for query, result in query_results_dict.items():
            documents = result.get("documents", [])
            scores = result.get("scores", {})
            explain_info = result.get("explain_info", {})

            results_by_query[query] = documents
            query_results[query] = QueryResult(
                query=query,
                documents=documents,
                scores=scores,
                explain_info=explain_info,
            )

        # Calculate metrics
        metrics = [experiment.primary_metric] + experiment.secondary_metrics
        metric_results = self.metrics_agent.calculate_metrics(
            metrics, results_by_query, experiment.judgments, experiment.metric_depth
        )

        # Create iteration result
        iteration_result = IterationResult(
            iteration_id=iteration_id,
            experiment_id=experiment_id,
            query_config=query_config,
            query_results=query_results,
            metric_results=metric_results,
        )

        # Compare with previous iteration if available
        previous_iterations = self.logging_agent.list_iterations(experiment_id)
        if previous_iterations:
            # By default, compare with the most recent iteration
            previous_iteration_id = previous_iterations[0].get("iteration_id")
            previous_iteration = self.logging_agent.get_iteration(
                experiment_id, previous_iteration_id
            )

            if previous_iteration:
                iteration_result.compared_to = previous_iteration_id
                # Calculate metric deltas
                deltas = self.comparison_agent.compare_overall_metrics(
                    previous_iteration, iteration_result
                )
                iteration_result.metric_deltas = deltas

        # Log the iteration
        success = self.logging_agent.log_iteration(iteration_result)
        if not success:
            logger.warning(f"Failed to log iteration {iteration_id}")

        return iteration_result

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
        # Retrieve the iterations
        iter1 = self.logging_agent.get_iteration(experiment_id, iteration_id1)
        iter2 = self.logging_agent.get_iteration(experiment_id, iteration_id2)

        if not iter1 or not iter2:
            missing = []
            if not iter1:
                missing.append(iteration_id1)
            if not iter2:
                missing.append(iteration_id2)
            raise ValueError(f"Iterations not found: {', '.join(missing)}")

        # Generate comparison report
        return self.comparison_agent.generate_summary_report(iter1, iter2)

    def get_iteration_history(self, experiment_id: str) -> List[IterationResult]:
        """
        Get the history of iterations for an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            A list of iteration results
        """
        # Get summaries of all iterations
        iteration_summaries = self.logging_agent.list_iterations(experiment_id)

        # Retrieve full details for each iteration
        return [
            self.logging_agent.get_iteration(experiment_id, summary["iteration_id"])
            for summary in iteration_summaries
            if self.logging_agent.get_iteration(experiment_id, summary["iteration_id"])
        ]

    def get_current_state(self, experiment_id: str):
        """
        Get the current state of an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            The current iteration result or None if experiment does not exist
        """
        # Get the most recent iteration
        iterations = self.logging_agent.list_iterations(experiment_id)
        if not iterations:
            # Experiment exists but has no iterations
            return None

        # Return the most recent iteration (assumed to be first in list)
        return self.logging_agent.get_iteration(
            experiment_id, iterations[0]["iteration_id"]
        )
