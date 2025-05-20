"""
CLI Main Module - Command-line interface for Solr Optimizer.

This module provides the main command-line interface for Solr Optimizer,
including argument parsing and command execution.
"""

import sys
import os
import argparse
import logging
import json
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from solr_optimizer.core.experiment_manager import ExperimentManager
from solr_optimizer.core.default_experiment_manager import DefaultExperimentManager
from solr_optimizer.agents.solr.pysolr_execution_agent import PySolrExecutionAgent
from solr_optimizer.agents.metrics.standard_metrics_agent import StandardMetricsAgent
from solr_optimizer.agents.logging.file_based_logging_agent import FileBasedLoggingAgent
from solr_optimizer.agents.comparison.standard_comparison_agent import StandardComparisonAgent
from solr_optimizer.agents.query.query_tuning_agent import QueryTuningAgent
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.query_config import QueryConfig
from solr_optimizer.models.iteration_result import IterationResult

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
    
    # We only use a dummy QueryTuningAgent for now - this will be enhanced in future phases
    query_tuning_agent = QueryTuningAgent()
    
    # Create and return the experiment manager
    return DefaultExperimentManager(
        query_tuning_agent=query_tuning_agent,
        solr_execution_agent=solr_agent,
        metrics_agent=metrics_agent,
        logging_agent=logging_agent,
        comparison_agent=comparison_agent
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
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'query' not in reader.fieldnames:
                raise ValueError(f"CSV file must have a 'query' column: {filepath}")
                
            for row in reader:
                if row['query'] and row['query'].strip():
                    queries.append(row['query'].strip())
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
        Dictionary mapping query strings to dictionaries of document IDs and relevance scores
    """
    judgments = {}
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            required_fields = ['query', 'document_id', 'relevance']
            if not all(field in reader.fieldnames for field in required_fields):
                raise ValueError(
                    f"CSV file must have 'query', 'document_id', and 'relevance' columns: {filepath}"
                )
                
            for row in reader:
                query = row['query'].strip()
                doc_id = row['document_id'].strip()
                try:
                    relevance = float(row['relevance'])
                except ValueError:
                    logger.warning(f"Invalid relevance value for {doc_id}: {row['relevance']}, using 0.0")
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
        Dictionary mapping query strings to dictionaries of document IDs and relevance scores
    """
    judgments = {}
    query_id_map = {}  # Map from query IDs to actual query strings
    
    try:
        with open(filepath, 'r') as f:
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
                    logger.warning(f"Invalid relevance value for {doc_id}: {parts[3]}, using 0.0")
                    relevance = 0.0
                
                # We use the query ID as a placeholder until we can map it to the actual query string
                if query_id not in judgments:
                    judgments[query_id] = {}
                    
                judgments[query_id][doc_id] = relevance
    except Exception as e:
        logger.error(f"Error loading judgments from TREC file: {e}")
        raise
    
    # Note: You need to provide a mapping from query IDs to query strings separately
    # This can be done by setting the query_id_to_query parameter when running commands
    
    return judgments


def map_query_ids_to_queries(judgments: Dict[str, Dict[str, float]], 
                           mapping: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    Map query IDs in judgments to actual query strings based on provided mapping.
    
    Args:
        judgments: Judgments with query IDs as keys
        mapping: Mapping from query IDs to query strings
        
    Returns:
        Judgments with query strings as keys
    """
    mapped_judgments = {}
    
    for query_id, docs in judgments.items():
        if query_id in mapping:
            mapped_judgments[mapping[query_id]] = docs
        else:
            # If we don't have a mapping, keep the ID (it might be a query string already)
            mapped_judgments[query_id] = docs
            
    return mapped_judgments


def save_json(data: Any, filepath: str):
    """Save data as JSON to the specified file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        return False


def load_json(filepath: str) -> Any:
    """Load JSON data from the specified file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        return None


def format_metric_results(metric_results: Dict) -> str:
    """Format metric results for display."""
    if not metric_results:
        return "No metrics available."
        
    result = []
    
    # Format overall metrics
    if "overall" in metric_results:
        result.append("Overall Metrics:")
        for metric, value in metric_results["overall"].items():
            result.append(f"  {metric}: {value:.4f}")
    
    # Format per-query metrics
    if "per_query" in metric_results:
        result.append("\nPer-Query Metrics:")
        for query, metrics in metric_results["per_query"].items():
            result.append(f"  Query: \"{query}\"")
            for metric, value in metrics.items():
                result.append(f"    {metric}: {value:.4f}")
    
    return "\n".join(result)


def cmd_create_experiment(args):
    """Handle the create-experiment command."""
    # Load queries and judgments if specified
    queries = []
    judgments = {}
    
    if args.queries_csv:
        queries = load_queries_from_csv(args.queries_csv)
        logger.info(f"Loaded {len(queries)} queries from {args.queries_csv}")
    elif args.queries_json:
        data = load_json(args.queries_json)
        if isinstance(data, list):
            queries = data
            logger.info(f"Loaded {len(queries)} queries from {args.queries_json}")
        else:
            logger.error(f"Invalid JSON format in {args.queries_json}: expected a list of query strings")
            return 1
    
    if args.judgments_csv:
        judgments = load_judgments_from_csv(args.judgments_csv)
        logger.info(f"Loaded judgments for {len(judgments)} queries from {args.judgments_csv}")
    elif args.judgments_trec:
        judgments = load_judgments_from_trec(args.judgments_trec)
        if args.query_id_mapping:
            mapping = load_json(args.query_id_mapping)
            judgments = map_query_ids_to_queries(judgments, mapping)
        logger.info(f"Loaded judgments for {len(judgments)} queries from {args.judgments_trec}")
    elif args.judgments_json:
        judgments = load_json(args.judgments_json)
        if not isinstance(judgments, dict):
            logger.error(f"Invalid JSON format in {args.judgments_json}: expected a dictionary mapping queries to relevance judgments")
            return 1
        logger.info(f"Loaded judgments for {len(judgments)} queries from {args.judgments_json}")
    
    # Ensure we have judgments for all queries
    if queries and not all(q in judgments for q in queries):
        missing = [q for q in queries if q not in judgments]
        logger.warning(f"Missing judgments for {len(missing)}/{len(queries)} queries: {missing[:5]}...")
        
    # Create experiment manager
    experiment_manager = create_experiment_manager(args.solr_url, args.storage_dir)
    
    # Create experiment configuration
    experiment_config = ExperimentConfig(
        experiment_id=args.id,
        name=args.name,
        description=args.description,
        corpus=args.corpus,
        queries=queries,
        judgments=judgments,
        primary_metric=args.metric,
        secondary_metrics=args.secondary_metrics,
        metric_depth=args.depth
    )
    
    # Set up the experiment
    try:
        experiment_id = experiment_manager.setup_experiment(experiment_config)
        print(f"Created experiment: {experiment_id}")
        if args.output:
            # Save experiment details to a file
            experiment_summary = {
                "experiment_id": experiment_id,
                "name": args.name,
                "description": args.description,
                "corpus": args.corpus,
                "num_queries": len(queries),
                "primary_metric": args.metric,
                "secondary_metrics": args.secondary_metrics,
                "metric_depth": args.depth,
                "created_at": datetime.datetime.now().isoformat()
            }
            save_json(experiment_summary, args.output)
            print(f"Experiment summary saved to {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        return 1


def cmd_run_iteration(args):
    """Handle the run-iteration command."""
    # Create experiment manager
    experiment_manager = create_experiment_manager(args.solr_url, args.storage_dir)
    
    # Load query configuration from file if specified
    query_config = None
    if args.config:
        config_data = load_json(args.config)
        if not config_data:
            return 1
        try:
            query_config = QueryConfig(**config_data)
        except Exception as e:
            logger.error(f"Error parsing query configuration: {e}")
            return 1
    else:
        # Create query configuration from command-line arguments
        config_dict = {
            "iteration_id": args.id,
            "description": args.description,
            "query_parser": args.query_parser
        }
        
        # Add optional parameters if specified
        if args.qf:
            config_dict["qf"] = args.qf
        if args.pf:
            config_dict["pf"] = args.pf
        if args.mm:
            config_dict["mm"] = args.mm
        if args.bq:
            config_dict["bq"] = args.bq
        if args.bf:
            config_dict["bf"] = args.bf
        if args.boost:
            config_dict["boost"] = args.boost
        
        query_config = QueryConfig(**config_dict)
    
    # Run the iteration
    try:
        result = experiment_manager.run_iteration(args.experiment_id, query_config)
        print(f"Iteration {result.iteration_id} completed successfully.")
        print()
        print(format_metric_results(result.metric_results))
        
        if args.output:
            # Save result summary to a file
            save_json(result.dict(), args.output)
            print(f"\nDetailed results saved to {args.output}")
        
        return 0
    except Exception as e:
        logger.error(f"Error running iteration: {e}")
        return 1


def cmd_compare_iterations(args):
    """Handle the compare-iterations command."""
    # Create experiment manager
    experiment_manager = create_experiment_manager(args.solr_url, args.storage_dir)
    
    # Compare iterations
    try:
        comparison = experiment_manager.compare_iterations(
            args.experiment_id, args.iteration1, args.iteration2
        )
        
        # Print comparison summary
        print(f"Comparison of iterations {args.iteration1} vs {args.iteration2}")
        print()
        
        # Metrics comparison
        print("Metrics Comparison:")
        for metric, delta in comparison.get("metrics_comparison", {}).items():
            print(f"  {metric}: {delta:+.4f}")
        print()
        
        # Configuration changes
        config_changes = comparison.get("config_changes", {})
        if config_changes:
            print("Configuration Changes:")
            for param, change in config_changes.items():
                if change["type"] == "added":
                    print(f"  + {param}: {change['after']}")
                elif change["type"] == "removed":
                    print(f"  - {param}: {change['before']}")
                else:
                    print(f"  ~ {param}: {change['before']} -> {change['after']}")
            print()
        
        # Improved and degraded queries
        improved = comparison.get("improved_queries", [])
        degraded = comparison.get("degraded_queries", [])
        
        if improved:
            print(f"Improved Queries ({len(improved)}):")
            for item in improved[:5]:  # Show top 5
                query = item["query"]
                delta = item["primary_metric"]["delta"]
                metric = item["primary_metric"]["name"]
                print(f"  {query}: {metric} {delta:+.4f}")
            if len(improved) > 5:
                print(f"  ... and {len(improved) - 5} more")
            print()
        
        if degraded:
            print(f"Degraded Queries ({len(degraded)}):")
            for item in degraded[:5]:  # Show top 5
                query = item["query"]
                delta = item["primary_metric"]["delta"]
                metric = item["primary_metric"]["name"]
                print(f"  {query}: {metric} {delta:+.4f}")
            if len(degraded) > 5:
                print(f"  ... and {len(degraded) - 5} more")
            print()
        
        # Save detailed comparison if requested
        if args.output:
            save_json(comparison, args.output)
            print(f"Detailed comparison saved to {args.output}")
        
        return 0
    except Exception as e:
        logger.error(f"Error comparing iterations: {e}")
        return 1


def cmd_list_experiments(args):
    """Handle the list-experiments command."""
    # Create experiment manager with just the logging agent
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    
    try:
        experiments = logging_agent.list_experiments()
        
        if not experiments:
            print("No experiments found.")
            return 0
        
        print(f"Found {len(experiments)} experiments:")
        for exp in experiments:
            exp_id = exp["experiment_id"]
            name = exp.get("name", exp_id)
            last_modified = exp.get("last_modified", "unknown")
            
            corpus = exp.get("metadata", {}).get("corpus", "unknown")
            metric = exp.get("metadata", {}).get("primary_metric", "unknown")
            
            print(f"- {exp_id}: {name}")
            print(f"  Corpus: {corpus}, Metric: {metric}")
            print(f"  Last modified: {last_modified}")
            print()
        
        return 0
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        return 1


def cmd_list_iterations(args):
    """Handle the list-iterations command."""
    # Create experiment manager with just the logging agent
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    
    try:
        iterations = logging_agent.list_iterations(args.experiment_id)
        
        if not iterations:
            print(f"No iterations found for experiment {args.experiment_id}.")
            return 0
        
        print(f"Found {len(iterations)} iterations for experiment {args.experiment_id}:")
        for iter_info in iterations:
            iter_id = iter_info["iteration_id"]
            timestamp = iter_info.get("timestamp", "unknown")
            
            # Extract query config summary
            config_summary = ""
            if "query_config" in iter_info:
                qc = iter_info["query_config"]
                parser = qc.get("query_parser", "unknown")
                desc = qc.get("description", "")
                config_summary = f"{parser}" + (f" - {desc}" if desc else "")
            
            # Extract metric
            metric_summary = ""
            if "metric" in iter_info:
                for metric, value in iter_info["metric"].items():
                    metric_summary = f"{metric}: {value:.4f}"
            
            # Extract tags
            tags = iter_info.get("tags", [])
            tags_summary = f"Tags: {', '.join(tags)}" if tags else ""
            
            print(f"- {iter_id}: {timestamp}")
            if config_summary:
                print(f"  Config: {config_summary}")
            if metric_summary:
                print(f"  {metric_summary}")
            if tags_summary:
                print(f"  {tags_summary}")
            print()
        
        return 0
    except Exception as e:
        logger.error(f"Error listing iterations: {e}")
        return 1


def cmd_export_experiment(args):
    """Handle the export-experiment command."""
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    
    try:
        success = logging_agent.export_experiment(args.experiment_id, args.output)
        
        if success:
            print(f"Experiment {args.experiment_id} exported to {args.output}")
            return 0
        else:
            print(f"Failed to export experiment {args.experiment_id}")
            return 1
    except Exception as e:
        logger.error(f"Error exporting experiment: {e}")
        return 1


def cmd_import_experiment(args):
    """Handle the import-experiment command."""
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    
    try:
        experiment_id = logging_agent.import_experiment(args.input)
        
        if experiment_id:
            print(f"Experiment imported with ID: {experiment_id}")
            return 0
        else:
            print("Failed to import experiment")
            return 1
    except Exception as e:
        logger.error(f"Error importing experiment: {e}")
        return 1


def cmd_tag_iteration(args):
    """Handle the tag-iteration command."""
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    
    try:
        success = logging_agent.tag_iteration(
            args.experiment_id, args.iteration_id, args.tag
        )
        
        if success:
            print(f"Tagged iteration {args.iteration_id} with '{args.tag}'")
            return 0
        else:
            print(f"Failed to tag iteration {args.iteration_id}")
            return 1
    except Exception as e:
        logger.error(f"Error tagging iteration: {e}")
        return 1


def cmd_branch_experiment(args):
    """Handle the branch-experiment command."""
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    
    try:
        new_id = logging_agent.branch_experiment(
            args.experiment_id, args.new_id, args.name
        )
        
        if new_id:
            print(f"Created branch of experiment {args.experiment_id} with ID: {new_id}")
            return 0
        else:
            print(f"Failed to branch experiment {args.experiment_id}")
            return 1
    except Exception as e:
        logger.error(f"Error branching experiment: {e}")
        return 1


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Solr Optimizer - A tool for optimizing Apache Solr queries."
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--storage-dir", 
        default="experiment_storage",
        help="Directory for storing experiment data"
    )
    parser.add_argument(
        "--solr-url", 
        default="http://localhost:8983/solr/",
        help="URL of the Solr server"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to execute",
        required=True
    )
    
    # create-experiment command
    create_parser = subparsers.add_parser(
        "create-experiment", 
        help="Create a new experiment"
    )
    create_parser.add_argument(
        "--id", 
        help="Experiment ID (will be generated if not provided)"
    )
    create_parser.add_argument(
        "--name", 
        help="Human-readable name for the experiment"
    )
    create_parser.add_argument(
        "--description", 
        help="Description of the experiment"
    )
    create_parser.add_argument(
        "--corpus", 
        required=True,
        help="Solr collection or core name"
    )
    create_parser.add_argument(
        "--metric", 
        default="ndcg",
        help="Primary evaluation metric (default: ndcg)"
    )
    create_parser.add_argument(
        "--secondary-metrics", 
        nargs="+",
        default=["precision", "recall"],
        help="Secondary evaluation metrics (default: precision recall)"
    )
    create_parser.add_argument(
        "--depth", 
        type=int,
        default=10,
        help="Evaluation depth for metrics (default: 10)"
    )
    create_parser.add_argument(
        "--queries-csv", 
        help="CSV file with queries (must have a 'query' column)"
    )
    create_parser.add_argument(
        "--queries-json", 
        help="JSON file with an array of query strings"
    )
    create_parser.add_argument(
        "--judgments-csv", 
        help="CSV file with relevance judgments (must have 'query', 'document_id', and 'relevance' columns)"
    )
    create_parser.add_argument(
        "--judgments-trec", 
        help="TREC format file with relevance judgments"
    )
    create_parser.add_argument(
        "--query-id-mapping", 
        help="JSON file mapping query IDs to query strings (for use with TREC judgments)"
    )
    create_parser.add_argument(
        "--judgments-json", 
        help="JSON file with relevance judgments"
    )
    create_parser.add_argument(
        "--output", 
        help="Output file for experiment details (JSON)"
    )
    create_parser.set_defaults(func=cmd_create_experiment)
    
    # run-iteration command
    run_parser = subparsers.add_parser(
        "run-iteration", 
        help="Run an experiment iteration"
    )
    run_parser.add_argument(
        "--experiment-id", 
        required=True,
        help="ID of the experiment to run"
    )
    run_parser.add_argument(
        "--id", 
        help="Iteration ID (will be generated if not provided)"
    )
    run_parser.add_argument(
        "--description", 
        help="Description of the iteration"
    )
    run_parser.add_argument(
        "--config", 
        help="JSON file with query configuration parameters"
    )
    run_parser.add_argument(
        "--query-parser", 
        default="edismax",
        help="Query parser to use (default: edismax)"
    )
    run_parser.add_argument(
        "--qf", 
        help="Query fields parameter (e.g., 'title^2.0 body^1.0')"
    )
    run_parser.add_argument(
        "--pf", 
        help="Phrase fields parameter"
    )
    run_parser.add_argument(
        "--mm", 
        help="Minimum match parameter"
    )
    run_parser.add_argument(
        "--bq", 
        help="Boost query parameter"
    )
    run_parser.add_argument(
        "--bf", 
        help="Boost function parameter"
    )
    run_parser.add_argument(
        "--boost", 
        help="Boost parameter"
    )
    run_parser.add_argument(
        "--output", 
        help="Output file for iteration results (JSON)"
    )
    run_parser.set_defaults(func=cmd_run_iteration)
    
    # compare-iterations command
    compare_parser = subparsers.add_parser(
        "compare-iterations", 
        help="Compare two iterations of an experiment"
    )
    compare_parser.add_argument(
        "--experiment-id", 
        required=True,
        help="ID of the experiment"
    )
    compare_parser.add_argument(
        "--iteration1", 
        required=True,
        help="ID of the first iteration to compare"
    )
    compare_parser.add_argument(
        "--iteration2", 
        required=True,
        help="ID of the second iteration to compare"
    )
    compare_parser.add_argument(
        "--output", 
        help="Output file for comparison results (JSON)"
    )
    compare_parser.set_defaults(func=cmd_compare_iterations)
    
    # list-experiments command
    list_exp_parser = subparsers.add_parser(
        "list-experiments", 
        help="List all experiments"
    )
    list_exp_parser.set_defaults(func=cmd_list_experiments)
    
    # list-iterations command
    list_iter_parser = subparsers.add_parser(
        "list-iterations", 
        help="List iterations for an experiment"
    )
    list_iter_parser.add_argument(
        "--experiment-id", 
        required=True,
        help="ID of the experiment"
    )
    list_iter_parser.set_defaults(func=cmd_list_iterations)
    
    # export-experiment command
    export_parser = subparsers.add_parser(
        "export-experiment", 
        help="Export an experiment to a JSON file"
    )
    export_parser.add_argument(
        "--experiment-id", 
        required=True,
        help="ID of the experiment to export"
    )
    export_parser.add_argument(
        "--output", 
        required=True,
        help="Output file path"
    )
    export_parser.set_defaults(func=cmd_export_experiment)
    
    # import-experiment command
    import_parser = subparsers.add_parser(
        "import-experiment", 
        help="Import an experiment from a JSON file"
    )
    import_parser.add_argument(
        "--input", 
        required=True,
        help="Input file path"
    )
    import_parser.set_defaults(func=cmd_import_experiment)
    
    # tag-iteration command
    tag_parser = subparsers.add_parser(
        "tag-iteration", 
        help="Tag an iteration with a user-friendly name"
    )
    tag_parser.add_argument(
        "--experiment-id", 
        required=True,
        help="ID of the experiment"
    )
    tag_parser.add_argument(
        "--iteration-id", 
        required=True,
        help="ID of the iteration to tag"
    )
    tag_parser.add_argument(
        "--tag", 
        required=True,
        help="Tag to apply to the iteration"
    )
    tag_parser.set_defaults(func=cmd_tag_iteration)
    
    # branch-experiment command
    branch_parser = subparsers.add_parser(
        "branch-experiment", 
        help="Create a branch of an existing experiment"
    )
    branch_parser.add_argument(
        "--experiment-id", 
        required=True,
        help="ID of the experiment to branch"
    )
    branch_parser.add_argument(
        "--new-id", 
        help="ID for the new experiment (will be generated if not provided)"
    )
    branch_parser.add_argument(
        "--name", 
        help="Name for the new experiment"
    )
    branch_parser.set_defaults(func=cmd_branch_experiment)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Execute the command
    if hasattr(args, 'func'):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
