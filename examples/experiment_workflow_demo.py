#!/usr/bin/env python3
"""
Experiment Workflow Demo - Demonstrates the complete experiment workflow with the Solr Optimizer.

This example shows how to:
1. Create an experiment with queries and relevance judgments
2. Run multiple iterations with different query configurations
3. Compare iterations and analyze results
4. Export experiment data and work with tags

Requirements:
- A running Solr instance with a collection containing documents
- Python 3.7+
- solr_optimizer package installed
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path

from solr_optimizer.core.default_experiment_manager import DefaultExperimentManager
from solr_optimizer.agents.solr.pysolr_execution_agent import PySolrExecutionAgent
from solr_optimizer.agents.metrics.standard_metrics_agent import StandardMetricsAgent
from solr_optimizer.agents.logging.file_based_logging_agent import FileBasedLoggingAgent
from solr_optimizer.agents.comparison.standard_comparison_agent import StandardComparisonAgent
from solr_optimizer.agents.query.query_tuning_agent import QueryTuningAgent
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.query_config import QueryConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("experiment_workflow_demo")


def create_sample_data():
    """
    Create sample queries and judgments for demonstration purposes.
    
    In a real scenario, you would load these from files using the CLI
    or prepare them programmatically from your evaluation data.
    
    Returns:
        tuple: (queries, judgments)
    """
    queries = [
        "machine learning",
        "natural language processing",
        "information retrieval",
        "deep learning",
        "neural networks"
    ]
    
    # Sample relevance judgments (query -> {doc_id -> relevance_score})
    judgments = {
        "machine learning": {
            "doc1": 3,  # Highly relevant
            "doc2": 2,  # Relevant
            "doc3": 1,  # Somewhat relevant
            "doc4": 0,  # Not relevant
            "doc5": 2   # Relevant
        },
        "natural language processing": {
            "doc3": 3,
            "doc6": 3,
            "doc7": 2,
            "doc8": 1,
            "doc9": 0
        },
        "information retrieval": {
            "doc10": 3,
            "doc11": 2,
            "doc12": 3,
            "doc4": 1,
            "doc13": 1
        },
        "deep learning": {
            "doc14": 3,
            "doc15": 3,
            "doc2": 2,
            "doc16": 2,
            "doc17": 1
        },
        "neural networks": {
            "doc14": 3,
            "doc18": 2,
            "doc19": 2,
            "doc20": 1,
            "doc21": 0
        }
    }
    
    return queries, judgments


def run_experiment(solr_url, storage_dir, corpus_name):
    """
    Run a complete experiment workflow.
    
    Args:
        solr_url: URL of the Solr server
        storage_dir: Directory to store experiment data
        corpus_name: Name of the Solr collection/core to use
    """
    logger.info(f"Starting experiment with corpus: {corpus_name}")
    
    # Create the experiment manager with all necessary agents
    solr_agent = PySolrExecutionAgent(solr_url)
    metrics_agent = StandardMetricsAgent()
    logging_agent = FileBasedLoggingAgent(storage_dir)
    comparison_agent = StandardComparisonAgent()
    query_tuning_agent = QueryTuningAgent()  # This is a placeholder implementation
    
    experiment_manager = DefaultExperimentManager(
        query_tuning_agent=query_tuning_agent,
        solr_execution_agent=solr_agent,
        metrics_agent=metrics_agent,
        logging_agent=logging_agent,
        comparison_agent=comparison_agent
    )
    
    # Create sample data for the experiment
    queries, judgments = create_sample_data()
    
    # Step 1: Set up the experiment
    experiment_config = ExperimentConfig(
        name="Demo Search Optimization",
        description="Demonstration of the complete optimization workflow",
        corpus=corpus_name,
        queries=queries,
        judgments=judgments,
        primary_metric="ndcg",
        secondary_metrics=["precision", "recall"],
        metric_depth=10
    )
    
    experiment_id = experiment_manager.setup_experiment(experiment_config)
    logger.info(f"Created experiment with ID: {experiment_id}")
    
    # Step 2: Run baseline iteration (default settings)
    baseline_config = QueryConfig(
        iteration_id="baseline",
        description="Default settings",
        query_parser="edismax",
        qf="title^1.0 content^1.0"  # Adjust field names according to your schema
    )
    
    baseline_result = experiment_manager.run_iteration(experiment_id, baseline_config)
    logger.info(f"Completed baseline iteration: {baseline_result.iteration_id}")
    logger.info(f"Baseline NDCG: {baseline_result.metric_results.get('overall', {}).get('ndcg', 0.0):.4f}")
    
    # Step 3: Run iteration with boosted title field
    boosted_title_config = QueryConfig(
        iteration_id="title_boost",
        description="Title field boosted 3x",
        query_parser="edismax",
        qf="title^3.0 content^1.0"  # Adjust field names according to your schema
    )
    
    boosted_title_result = experiment_manager.run_iteration(experiment_id, boosted_title_config)
    logger.info(f"Completed title boost iteration: {boosted_title_result.iteration_id}")
    logger.info(f"Title boost NDCG: {boosted_title_result.metric_results.get('overall', {}).get('ndcg', 0.0):.4f}")
    
    # Step 4: Run iteration with phrase boosting
    phrase_boost_config = QueryConfig(
        iteration_id="phrase_boost",
        description="Phrase fields boosting",
        query_parser="edismax",
        qf="title^2.0 content^1.0",
        pf="title^5.0"  # Adjust field names according to your schema
    )
    
    phrase_boost_result = experiment_manager.run_iteration(experiment_id, phrase_boost_config)
    logger.info(f"Completed phrase boost iteration: {phrase_boost_result.iteration_id}")
    logger.info(f"Phrase boost NDCG: {phrase_boost_result.metric_results.get('overall', {}).get('ndcg', 0.0):.4f}")
    
    # Step 5: Tag the best iteration
    # Find the best iteration based on NDCG score
    iterations = [baseline_result, boosted_title_result, phrase_boost_result]
    best_iteration = max(
        iterations,
        key=lambda it: it.metric_results.get("overall", {}).get("ndcg", 0.0)
    )
    
    logging_agent.tag_iteration(experiment_id, best_iteration.iteration_id, "best")
    logger.info(f"Tagged iteration {best_iteration.iteration_id} as 'best'")
    
    # Step 6: Compare iterations
    comparison = experiment_manager.compare_iterations(
        experiment_id, baseline_result.iteration_id, best_iteration.iteration_id
    )
    
    logger.info("\nComparison results:")
    logger.info(f"NDCG delta: {comparison.get('metrics_comparison', {}).get('ndcg', 0.0):+.4f}")
    
    improved_queries = comparison.get("improved_queries", [])
    degraded_queries = comparison.get("degraded_queries", [])
    
    logger.info(f"Improved queries: {len(improved_queries)}")
    for q in improved_queries:
        delta = q.get("primary_metric", {}).get("delta", 0.0)
        logger.info(f"  - '{q.get('query')}': {delta:+.4f}")
    
    logger.info(f"Degraded queries: {len(degraded_queries)}")
    for q in degraded_queries:
        delta = q.get("primary_metric", {}).get("delta", 0.0)
        logger.info(f"  - '{q.get('query')}': {delta:+.4f}")
    
    # Step 7: Export the experiment
    export_path = os.path.join(storage_dir, f"{experiment_id}_export.json")
    logging_agent.export_experiment(experiment_id, export_path)
    logger.info(f"Exported experiment to {export_path}")
    
    # Step 8: Create a branch of the experiment for further optimization
    branch_id = logging_agent.branch_experiment(
        experiment_id, name="Advanced optimization branch"
    )
    logger.info(f"Created branch experiment with ID: {branch_id}")
    
    return experiment_id


def main():
    """Main entry point for the example."""
    # Default values - adjust these to match your environment
    solr_url = os.environ.get("SOLR_URL", "http://localhost:8983/solr")
    corpus_name = os.environ.get("SOLR_COLLECTION", "techproducts")
    
    # Create a temporary directory for storing experiment data
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory for storage: {temp_dir}")
        
        try:
            experiment_id = run_experiment(solr_url, temp_dir, corpus_name)
            logger.info(f"Successfully completed experiment: {experiment_id}")
        except Exception as e:
            logger.error(f"Error running experiment: {e}", exc_info=True)
            return 1
    
    logger.info("Experiment demo completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
