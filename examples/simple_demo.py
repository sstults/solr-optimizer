"""
Simple demonstration of the Solr Optimizer framework.

This example shows how to set up the framework, configure an experiment,
and run iterations with different query configurations.
"""

import logging
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solr_optimizer.agents.metrics.standard_metrics_agent import StandardMetricsAgent
from solr_optimizer.agents.solr.pysolr_execution_agent import PySolrExecutionAgent
from solr_optimizer.core.default_experiment_manager import DefaultExperimentManager
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.query_config import QueryConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DummyLoggingAgent:
    """Simple in-memory logging agent for demonstration purposes."""

    def __init__(self):
        self.experiments = {}
        self.iterations = {}

    def log_iteration(self, iteration_result):
        exp_id = iteration_result.experiment_id
        iter_id = iteration_result.iteration_id

        if exp_id not in self.iterations:
            self.iterations[exp_id] = {}

        self.iterations[exp_id][iter_id] = iteration_result
        logger.info(f"Logged iteration {iter_id} for experiment {exp_id}")
        return True

    def get_iteration(self, experiment_id, iteration_id):
        return self.iterations.get(experiment_id, {}).get(iteration_id)

    def list_iterations(self, experiment_id):
        if experiment_id not in self.iterations:
            return []

        # Sort by timestamp (newest first)
        sorted_iterations = sorted(
            self.iterations[experiment_id].values(),
            key=lambda x: x.timestamp,
            reverse=True,
        )

        return [i.summary_dict() for i in sorted_iterations]

    def save_experiment(self, experiment):
        self.experiments[experiment.experiment_id] = experiment
        logger.info(f"Saved experiment {experiment.experiment_id}")
        return True

    def get_experiment(self, experiment_id):
        return self.experiments.get(experiment_id)

    def list_experiments(self):
        return [{"id": exp_id, "corpus": exp.corpus} for exp_id, exp in self.experiments.items()]

    def tag_iteration(self, experiment_id, iteration_id, tag):
        if experiment_id in self.iterations and iteration_id in self.iterations[experiment_id]:
            # For a real implementation, we'd store the tag
            logger.info(f"Tagged iteration {iteration_id} as '{tag}'")
            return True
        return False


class DummyComparisonAgent:
    """Simple comparison agent for demonstration purposes."""

    def compare_overall_metrics(self, iter1, iter2):
        """Compare metrics between iterations."""
        deltas = {}

        # Find matching metrics in both iterations
        iter1_metrics = {m.metric_name: m.value for m in iter1.metric_results}
        iter2_metrics = {m.metric_name: m.value for m in iter2.metric_results}

        # Calculate deltas
        for name, value in iter2_metrics.items():
            if name in iter1_metrics:
                deltas[name] = value - iter1_metrics[name]

        return deltas

    def compare_query_results(self, iter1, iter2, query):
        """Compare results for a specific query."""
        return {"status": "not_implemented"}

    def explain_ranking_changes(self, iter1, iter2, query):
        """Explain ranking changes for a query."""
        return [{"status": "not_implemented"}]

    def find_significant_changes(self, iter1, iter2):
        """Find significant changes between iterations."""
        return [{"status": "not_implemented"}]

    def generate_summary_report(self, iter1, iter2):
        """Generate a summary report comparing iterations."""
        metrics_delta = self.compare_overall_metrics(iter1, iter2)

        return {
            "metrics_delta": metrics_delta,
            "config_changes": self.analyze_config_changes(iter1, iter2),
        }

    def analyze_config_changes(self, iter1, iter2):
        """Analyze configuration changes between iterations."""
        changes = {}

        # Compare query fields
        if iter1.query_config.query_fields != iter2.query_config.query_fields:
            changes["query_fields"] = {
                "before": iter1.query_config.query_fields,
                "after": iter2.query_config.query_fields,
            }

        # Compare phrase fields
        if iter1.query_config.phrase_fields != iter2.query_config.phrase_fields:
            changes["phrase_fields"] = {
                "before": iter1.query_config.phrase_fields,
                "after": iter2.query_config.phrase_fields,
            }

        # Compare minimum_match
        if iter1.query_config.minimum_match != iter2.query_config.minimum_match:
            changes["minimum_match"] = {
                "before": iter1.query_config.minimum_match,
                "after": iter2.query_config.minimum_match,
            }

        return changes


class DummyQueryTuningAgent:
    """Simple query tuning agent for demonstration purposes."""

    def analyze_schema(self, schema_info):
        """Analyze Solr schema to identify useful fields."""
        # This would analyze the schema to find text fields, date fields, etc.
        return {"status": "not_implemented"}

    def generate_initial_config(self, experiment_config, schema_info):
        """Generate an initial query configuration."""
        # For demonstration, create a basic eDismax config
        return QueryConfig(
            iteration_id="baseline",
            query_parser="edismax",
            query_fields={"title": 2.0, "content": 1.0},
            phrase_fields={"title": 3.0},
            minimum_match="100%",
            tie_breaker=0.0,
        )

    def suggest_next_config(self, previous_result, schema_info):
        """Suggest a new query configuration based on previous results."""
        # For demonstration, modify the previous config
        old_config = previous_result.query_config

        # Increase title boost
        new_query_fields = dict(old_config.query_fields)
        if "title" in new_query_fields:
            new_query_fields["title"] *= 1.5

        # Relax minimum match requirement
        new_minimum_match = "75%"

        return QueryConfig(
            iteration_id=f"iter_after_{previous_result.iteration_id}",
            query_parser=old_config.query_parser,
            query_fields=new_query_fields,
            phrase_fields=old_config.phrase_fields,
            minimum_match=new_minimum_match,
            tie_breaker=old_config.tie_breaker,
        )

    def adjust_parameters(self, result, target_metric, direction):
        """Adjust specific parameters to improve a target metric."""
        # This would implement more sophisticated parameter adjustment
        return self.suggest_next_config(result, {})


def run_demo():
    """Run a simple demonstration of the framework."""
    # Initialize agents
    solr_url = "http://localhost:8983/solr"  # Change to your Solr URL

    solr_agent = PySolrExecutionAgent(solr_url)
    query_agent = DummyQueryTuningAgent()
    metrics_agent = StandardMetricsAgent()
    logging_agent = DummyLoggingAgent()
    comparison_agent = DummyComparisonAgent()

    # Initialize experiment manager
    manager = DefaultExperimentManager(
        query_tuning_agent=query_agent,
        solr_execution_agent=solr_agent,
        metrics_agent=metrics_agent,
        logging_agent=logging_agent,
        comparison_agent=comparison_agent,
    )

    # Define sample experiment configuration
    # In a real scenario, this would come from file or user input
    experiment = ExperimentConfig(
        corpus="techproducts",  # Use a collection that exists on your Solr
        queries=["laptop", "smartphone", "camera"],
        judgments={
            "laptop": {"doc1": 3, "doc2": 2, "doc3": 1, "doc4": 0},
            "smartphone": {"doc5": 3, "doc6": 2, "doc7": 1},
            "camera": {"doc8": 3, "doc9": 2, "doc10": 1},
        },
        primary_metric="ndcg",
        metric_depth=10,
        experiment_id="demo-experiment",
        description="Demonstration experiment",
    )

    try:
        # Set up experiment
        experiment_id = manager.setup_experiment(experiment)
        logger.info(f"Created experiment: {experiment_id}")

        # Get initial schema info (in a real implementation)
        # schema_info = solr_agent.fetch_schema(experiment.corpus)
        schema_info = {}  # Dummy for demonstration

        # Generate initial configuration
        initial_config = query_agent.generate_initial_config(experiment, schema_info)
        logger.info(f"Generated initial configuration: {initial_config}")

        # Run baseline iteration
        baseline_result = manager.run_iteration(experiment_id, initial_config)
        logger.info(f"Completed baseline iteration: {baseline_result.iteration_id}")
        logger.info(f"Baseline metrics: {[m.metric_name + '=' + str(m.value) for m in baseline_result.metric_results]}")

        # Generate improved configuration
        improved_config = query_agent.suggest_next_config(baseline_result, schema_info)
        logger.info(f"Generated improved configuration: {improved_config}")

        # Run improved iteration
        improved_result = manager.run_iteration(experiment_id, improved_config)
        logger.info(f"Completed improved iteration: {improved_result.iteration_id}")
        logger.info(f"Improved metrics: {[m.metric_name + '=' + str(m.value) for m in improved_result.metric_results]}")

        # Compare iterations
        comparison = manager.compare_iterations(
            experiment_id, baseline_result.iteration_id, improved_result.iteration_id
        )
        logger.info(f"Comparison results: {comparison}")

    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_demo()
