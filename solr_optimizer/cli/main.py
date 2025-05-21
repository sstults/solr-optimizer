"""
CLI Main Module - Command-line interface for Solr Optimizer.

This module provides the main command-line interface for Solr Optimizer,
including argument parsing and command execution.
"""

import csv
import logging
from typing import Dict, List

from solr_optimizer.agents.comparison.standard_comparison_agent import (
    StandardComparisonAgent,
)
from solr_optimizer.agents.logging.file_based_logging_agent import FileBasedLoggingAgent
from solr_optimizer.agents.metrics.standard_metrics_agent import StandardMetricsAgent
from solr_optimizer.agents.query.query_tuning_agent import QueryTuningAgent
from solr_optimizer.agents.solr.pysolr_execution_agent import PySolrExecutionAgent
from solr_optimizer.core.default_experiment_manager import DefaultExperimentManager
from solr_optimizer.core.experiment_manager import ExperimentManager

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)


def create_experiment_manager(solr_url: str, storage_dir: str) -> ExperimentManager:
    """
    Create and configure an experiment manager with all required agents.

    Args:
        solr_url: URL of the Solr server
        storage_dir: Directory for storing experiment data

    Returns:
        Configured ExperimentManager instance
    """
    # Initialize agents
    solr_agent = PySolrExecutionAgent(solr_url)
    metrics_agent = StandardMetricsAgent()
    logging_agent = FileBasedLoggingAgent(storage_dir)
    comparison_agent = StandardComparisonAgent()

    # We only use a dummy QueryTuningAgent for now - this will be enhanced in
    # future phases
    query_tuning_agent = QueryTuningAgent()

    # Create and return the experiment manager
    return DefaultExperimentManager(
        query_tuning_agent=query_tuning_agent,
        solr_execution_agent=solr_agent,
        metrics_agent=metrics_agent,
        logging_agent=logging_agent,
        comparison_agent=comparison_agent,
    )


def load_queries_from_csv(filepath: str) -> List[str]:
    """
    Load queries from a CSV file.

    The CSV file should have a 'query' column.

    Args:
        filepath: Path to the CSV file

    Returns:
        List of query strings
    """
    queries = []
    try:
        with open(filepath, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if "query" not in reader.fieldnames:
                raise ValueError(f"CSV file must have a 'query' column: " f"{filepath}")

            for row in reader:
                if row["query"] and row["query"].strip():
                    queries.append(row["query"].strip())
    except Exception as e:
        logger.error(f"Error loading queries from CSV: {e}")
        raise

    return queries


def load_judgments_from_csv(filepath: str) -> Dict[str, Dict[str, float]]:
    """
    Load relevance judgments from a CSV file.

    The CSV file should have 'query', 'document_id', and 'relevance' columns.

    Args:
        filepath: Path to the CSV file

    Returns:
        Dictionary mapping query strings to dictionaries of document IDs and
        relevance scores
    """
    judgments = {}
    try:
        with open(filepath, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            required_fields = ["query", "document_id", "relevance"]
            if not all(field in reader.fieldnames for field in required_fields):
                raise ValueError(
                    f"CSV file must have 'query', 'document_id', "
                    f"and 'relevance' columns: {filepath}"
                )

            for row in reader:
                query = row["query"].strip()
                doc_id = row["document_id"].strip()
                try:
                    relevance = float(row["relevance"])
                except ValueError:
                    logger.warning(
                        f"Invalid relevance value for {doc_id}: "
                        f"{row['relevance']}, using 0.0"
                    )
                    relevance = 0.0

                if query not in judgments:
                    judgments[query] = {}

                judgments[query][doc_id] = relevance
    except Exception as e:
        logger.error(f"Error loading judgments from CSV: {e}")
        raise

    return judgments


def load_judgments_from_trec(filepath: str) -> Dict[str, Dict[str, float]]:
    """
    Load relevance judgments from a TREC format file.

    The TREC format is: query_id 0 document_id relevance

    Args:
        filepath: Path to the TREC file

    Returns:
        Dictionary mapping query strings to dictionaries of document IDs and
        relevance scores
    """
    judgments = {}

    try:
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    logger.warning(f"Invalid TREC line: {line}")
                    continue

                query_id = parts[0]
                doc_id = parts[2]
                try:
                    relevance = float(parts[3])
                except ValueError:
                    logger.warning(
                        f"Invalid relevance value for {doc_id}: {parts[3]}, "
                        f"using 0.0"
                    )
                    relevance = 0.0

                # We use the query ID as a placeholder until we can map it to
                # the actual query string
                if query_id not in judgments:
                    judgments[query_id] = {}

                judgments[query_id][doc_id] = relevance
    except Exception as e:
        logger.error(f"Error loading judgments from TREC file: {e}")
        raise

    # Note: You need to provide a mapping from query IDs to query strings
    # separately. This can be done by setting the query_id_to_query parameter
    # when running commands

    return judgments


# Rest of the file remains the same (truncated for brevity)
