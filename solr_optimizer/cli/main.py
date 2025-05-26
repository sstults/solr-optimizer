"""
CLI Main Module - Command-line interface for Solr Optimizer.

This module provides the main command-line interface for Solr Optimizer,
including argument parsing and command execution.
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Optional

from solr_optimizer.agents.comparison.standard_comparison_agent import (
    StandardComparisonAgent,
)
from solr_optimizer.agents.logging.file_based_logging_agent import FileBasedLoggingAgent
from solr_optimizer.agents.metrics.standard_metrics_agent import StandardMetricsAgent
from solr_optimizer.agents.query.dummy_query_tuning_agent import DummyQueryTuningAgent
from solr_optimizer.agents.solr.pysolr_execution_agent import PySolrExecutionAgent
from solr_optimizer.core.default_experiment_manager import DefaultExperimentManager
from solr_optimizer.core.ai_experiment_manager import AIExperimentManager
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.query_config import QueryConfig

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)


def create_experiment_manager(solr_url: str, storage_dir: str, enable_ai: bool = False, 
                             ai_model: str = "openai:gpt-4", ai_config: Optional[Dict] = None) -> DefaultExperimentManager:
    """
    Create and configure an experiment manager with all required agents.

    Args:
        solr_url: URL of the Solr server
        storage_dir: Directory for storing experiment data
        enable_ai: Whether to enable AI-powered optimization
        ai_model: AI model to use for optimization
        ai_config: Additional AI configuration parameters

    Returns:
        Configured DefaultExperimentManager or AIExperimentManager instance
    """
    # Initialize agents
    solr_agent = PySolrExecutionAgent(solr_url)
    metrics_agent = StandardMetricsAgent()
    logging_agent = FileBasedLoggingAgent(storage_dir)
    comparison_agent = StandardComparisonAgent()
    query_tuning_agent = DummyQueryTuningAgent()

    # Create appropriate experiment manager based on AI settings
    if enable_ai:
        return AIExperimentManager(
            query_tuning_agent=query_tuning_agent,
            solr_execution_agent=solr_agent,
            metrics_agent=metrics_agent,
            logging_agent=logging_agent,
            comparison_agent=comparison_agent,
            ai_model=ai_model,
            ai_config=ai_config or {},
            enable_ai=True,
        )
    else:
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
            if reader.fieldnames is None or "query" not in reader.fieldnames:
                raise ValueError(f"CSV file must have a 'query' column: {filepath}")

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
    judgments: Dict[str, Dict[str, float]] = {}
    try:
        with open(filepath, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            required_fields = ["query", "document_id", "relevance"]
            if reader.fieldnames is None or not all(field in reader.fieldnames for field in required_fields):
                raise ValueError(f"CSV file must have 'query', 'document_id', and 'relevance' columns: {filepath}")

            for row in reader:
                query = row["query"].strip()
                doc_id = row["document_id"].strip()
                try:
                    relevance = float(row["relevance"])
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
        Dictionary mapping query IDs to dictionaries of document IDs and
        relevance scores
    """
    judgments: Dict[str, Dict[str, float]] = {}

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
                    logger.warning(f"Invalid relevance value for {doc_id}: {parts[3]}, using 0.0")
                    relevance = 0.0

                if query_id not in judgments:
                    judgments[query_id] = {}

                judgments[query_id][doc_id] = relevance
    except Exception as e:
        logger.error(f"Error loading judgments from TREC file: {e}")
        raise

    return judgments


def cmd_create_experiment(args) -> None:
    """Create a new experiment with queries and judgments."""
    setup_logging(args.verbose)
    
    # Load queries
    queries = []
    if args.queries_csv:
        queries = load_queries_from_csv(args.queries_csv)
    elif args.queries_json:
        with open(args.queries_json, 'r') as f:
            data = json.load(f)
            queries = data if isinstance(data, list) else data.get('queries', [])
    
    # Load judgments
    judgments = {}
    if args.judgments_csv:
        judgments = load_judgments_from_csv(args.judgments_csv)
    elif args.judgments_trec:
        judgments = load_judgments_from_trec(args.judgments_trec)
    elif args.judgments_json:
        with open(args.judgments_json, 'r') as f:
            judgments = json.load(f)
    
    # Create experiment manager
    experiment_manager = create_experiment_manager(args.solr_url, args.storage_dir)
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        name=args.name,
        description=args.description or "",
        corpus=args.corpus,
        queries=queries,
        judgments=judgments,
        primary_metric=args.metric,
        secondary_metrics=args.secondary_metrics or [],
        metric_depth=args.depth,
    )
    
    # Setup experiment
    experiment_id = experiment_manager.setup_experiment(experiment_config)
    print(f"Created experiment: {experiment_id}")


def cmd_run_iteration(args) -> None:
    """Run an iteration with specified query configuration."""
    setup_logging(args.verbose)
    
    # Create experiment manager
    experiment_manager = create_experiment_manager(args.solr_url, args.storage_dir)
    
    # Build query config from arguments
    query_config = QueryConfig(
        iteration_id=args.iteration_id,
        description=args.description or "",
        query_parser=args.parser or "edismax",
    )
    
    # Add query parameters
    if args.qf:
        query_config.qf = args.qf
    if args.pf:
        query_config.pf = args.pf
    if args.mm:
        query_config.mm = args.mm
    if args.boost:
        query_config.boost = args.boost
    
    # Add any additional parameters
    additional_params = {}
    if args.additional_params:
        for param in args.additional_params:
            key, value = param.split('=', 1)
            additional_params[key] = value
    query_config.additional_params = additional_params
    
    # Run iteration
    result = experiment_manager.run_iteration(args.experiment_id, query_config)
    
    print(f"Iteration completed: {result.iteration_id}")
    if hasattr(result.metric_results, 'get') and result.metric_results.get('overall'):
        for metric, value in result.metric_results['overall'].items():
            print(f"  {metric}: {value:.4f}")


def cmd_compare_iterations(args) -> None:
    """Compare two iterations and show differences."""
    setup_logging(args.verbose)
    
    # Create experiment manager
    experiment_manager = create_experiment_manager(args.solr_url, args.storage_dir)
    
    # Compare iterations
    comparison = experiment_manager.compare_iterations(
        args.experiment_id, args.iteration1, args.iteration2
    )
    
    print(f"Comparison: {args.iteration1} vs {args.iteration2}")
    print("=" * 50)
    
    # Overall metrics comparison
    metrics_comparison = comparison.get('metrics_comparison', {})
    if metrics_comparison:
        print("Overall Metrics:")
        for metric, delta in metrics_comparison.items():
            print(f"  {metric}: {delta:+.4f}")
        print()
    
    # Improved and degraded queries
    improved = comparison.get('improved_queries', [])
    degraded = comparison.get('degraded_queries', [])
    unchanged = comparison.get('unchanged_queries', [])
    
    print(f"Query Summary:")
    print(f"  Improved: {len(improved)}")
    print(f"  Degraded: {len(degraded)}")
    print(f"  Unchanged: {len(unchanged)}")
    print()
    
    if improved and args.show_details:
        print("Top Improved Queries:")
        for q in improved[:5]:  # Show top 5
            delta = q.get('primary_metric', {}).get('delta', 0.0)
            print(f"  '{q.get('query')}': {delta:+.4f}")
        print()
    
    if degraded and args.show_details:
        print("Top Degraded Queries:")
        for q in degraded[:5]:  # Show top 5
            delta = q.get('primary_metric', {}).get('delta', 0.0)
            print(f"  '{q.get('query')}': {delta:+.4f}")


def cmd_list_experiments(args) -> None:
    """List all experiments."""
    setup_logging(args.verbose)
    
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    experiments = logging_agent.list_experiments()
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp_id in experiments:
        try:
            exp_config = logging_agent.load_experiment_config(exp_id)
            print(f"  {exp_id}: {exp_config.name} ({exp_config.corpus})")
        except Exception as e:
            print(f"  {exp_id}: <error loading config: {e}>")


def cmd_list_iterations(args) -> None:
    """List iterations for an experiment."""
    setup_logging(args.verbose)
    
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    iterations = logging_agent.list_iterations(args.experiment_id)
    
    if not iterations:
        print(f"No iterations found for experiment {args.experiment_id}")
        return
    
    print(f"Found {len(iterations)} iterations for {args.experiment_id}:")
    for iter_id in iterations:
        try:
            result = logging_agent.load_iteration_result(args.experiment_id, iter_id)
            description = getattr(result.query_config, 'description', '') if result.query_config else ''
            print(f"  {iter_id}: {description}")
        except Exception as e:
            print(f"  {iter_id}: <error loading: {e}>")


def cmd_export_experiment(args) -> None:
    """Export experiment data to a file."""
    setup_logging(args.verbose)
    
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    logging_agent.export_experiment(args.experiment_id, args.output_file)
    print(f"Exported experiment {args.experiment_id} to {args.output_file}")


def cmd_import_experiment(args) -> None:
    """Import experiment data from a file."""
    setup_logging(args.verbose)
    
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    experiment_id = logging_agent.import_experiment(args.input_file)
    print(f"Imported experiment: {experiment_id}")


def cmd_tag_iteration(args) -> None:
    """Tag an iteration with a label."""
    setup_logging(args.verbose)
    
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    logging_agent.tag_iteration(args.experiment_id, args.iteration_id, args.tag)
    print(f"Tagged iteration {args.iteration_id} with '{args.tag}'")


def cmd_branch_experiment(args) -> None:
    """Create a branch of an experiment."""
    setup_logging(args.verbose)
    
    logging_agent = FileBasedLoggingAgent(args.storage_dir)
    branch_id = logging_agent.branch_experiment(args.experiment_id, name=args.name)
    print(f"Created branch: {branch_id}")


def cmd_ai_recommend(args) -> None:
    """Get AI-powered optimization recommendation for an experiment."""
    setup_logging(args.verbose)
    
    # Parse constraints if provided
    constraints = {}
    if args.constraints:
        for constraint in args.constraints:
            key, value = constraint.split('=', 1)
            constraints[key] = value
    
    # Create AI-enabled experiment manager
    experiment_manager = create_experiment_manager(
        args.solr_url, args.storage_dir, enable_ai=True, 
        ai_model=args.ai_model, ai_config={}
    )
    
    if not isinstance(experiment_manager, AIExperimentManager):
        print("Error: AI functionality not available")
        return
    
    # Get AI recommendation
    recommendation = experiment_manager.get_ai_recommendation(args.experiment_id, constraints)
    
    if not recommendation:
        print("No AI recommendation available. Check that:")
        print("  - The experiment exists")
        print("  - AI model is properly configured")
        print("  - There is sufficient experiment history")
        return
    
    print("AI Optimization Recommendation")
    print("=" * 40)
    print(f"Confidence: {recommendation.confidence:.2f}")
    print(f"Risk Level: {recommendation.risk_level}")
    print(f"Priority: {recommendation.priority}/10")
    print(f"Expected Impact: {recommendation.expected_impact}")
    print()
    print("Reasoning:")
    print(f"  {recommendation.reasoning}")
    print()
    print("Suggested Changes:")
    for key, value in recommendation.suggested_changes.items():
        print(f"  {key}: {value}")


def cmd_ai_preview(args) -> None:
    """Preview AI optimization recommendation without executing it."""
    setup_logging(args.verbose)
    
    # Parse constraints if provided
    constraints = {}
    if args.constraints:
        for constraint in args.constraints:
            key, value = constraint.split('=', 1)
            constraints[key] = value
    
    # Create AI-enabled experiment manager
    experiment_manager = create_experiment_manager(
        args.solr_url, args.storage_dir, enable_ai=True, 
        ai_model=args.ai_model, ai_config={}
    )
    
    if not isinstance(experiment_manager, AIExperimentManager):
        print("Error: AI functionality not available")
        return
    
    # Get AI recommendation preview
    preview = experiment_manager.preview_ai_recommendation(args.experiment_id, constraints)
    
    if not preview:
        print("No AI recommendation preview available.")
        return
    
    print("AI Optimization Preview")
    print("=" * 30)
    print(f"Confidence: {preview['confidence']:.2f}")
    print(f"Risk Level: {preview['risk_level']}")
    print(f"Priority: {preview['priority']}/10")
    print(f"Expected Impact: {preview['expected_impact']}")
    print()
    print("Reasoning:")
    print(f"  {preview['reasoning']}")
    print()
    
    if preview['preview_query_config']:
        config = preview['preview_query_config']
        print("Generated Query Configuration:")
        print(f"  Iteration ID: {config.iteration_id}")
        print(f"  Description: {config.description}")
        print(f"  Query Parser: {config.query_parser}")
        if config.qf:
            print(f"  Query Fields (qf): {config.qf}")
        if config.pf:
            print(f"  Phrase Fields (pf): {config.pf}")
        if config.mm:
            print(f"  Minimum Match (mm): {config.mm}")
        if config.boost:
            print(f"  Boost: {config.boost}")
        if config.additional_params:
            print("  Additional Parameters:")
            for key, value in config.additional_params.items():
                print(f"    {key}: {value}")


def cmd_ai_optimize(args) -> None:
    """Run AI-optimized iteration for an experiment."""
    setup_logging(args.verbose)
    
    # Parse constraints if provided
    constraints = {}
    if args.constraints:
        for constraint in args.constraints:
            key, value = constraint.split('=', 1)
            constraints[key] = value
    
    # Create AI-enabled experiment manager
    experiment_manager = create_experiment_manager(
        args.solr_url, args.storage_dir, enable_ai=True, 
        ai_model=args.ai_model, ai_config={}
    )
    
    if not isinstance(experiment_manager, AIExperimentManager):
        print("Error: AI functionality not available")
        return
    
    # Run AI-optimized iteration
    result = experiment_manager.run_ai_optimized_iteration(args.experiment_id, constraints)
    
    if not result:
        print("AI optimization failed. Check logs for details.")
        return
    
    print(f"AI-optimized iteration completed: {result.iteration_id}")
    
    # Show AI metadata if available
    if result.metadata and result.metadata.get('ai_generated'):
        print("AI Optimization Details:")
        print(f"  Model: {result.metadata.get('ai_model', 'Unknown')}")
        print(f"  Confidence: {result.metadata.get('ai_confidence', 0.0):.2f}")
        print(f"  Risk Level: {result.metadata.get('ai_risk_level', 'Unknown')}")
        print(f"  Reasoning: {result.metadata.get('ai_reasoning', 'No reasoning available')}")
        print()
    
    # Show metric results
    if hasattr(result.metric_results, 'get') and result.metric_results.get('overall'):
        print("Metric Results:")
        for metric, value in result.metric_results['overall'].items():
            print(f"  {metric}: {value:.4f}")
        
        # Show improvement if available
        if result.metric_deltas:
            print("Improvements vs Previous:")
            for metric, delta in result.metric_deltas.items():
                print(f"  {metric}: {delta:+.4f}")


def cmd_ai_status(args) -> None:
    """Show AI system status and configuration."""
    setup_logging(args.verbose)
    
    # Create AI-enabled experiment manager
    experiment_manager = create_experiment_manager(
        args.solr_url, args.storage_dir, enable_ai=True, 
        ai_model=args.ai_model, ai_config={}
    )
    
    if not isinstance(experiment_manager, AIExperimentManager):
        print("AI functionality is not available")
        return
    
    status = experiment_manager.get_ai_status()
    
    print("AI System Status")
    print("=" * 20)
    print(f"AI Enabled: {status['ai_enabled']}")
    print(f"AI Model: {status['ai_model']}")
    print(f"Orchestrator Available: {status['orchestrator_available']}")
    
    if status['ai_config']:
        print("AI Configuration:")
        for key, value in status['ai_config'].items():
            print(f"  {key}: {value}")
    else:
        print("AI Configuration: Default settings")


def main() -> None:
    """Main CLI entry point for Solr Optimizer."""
    parser = argparse.ArgumentParser(
        description="Solr Optimizer - Systematic query optimization framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Global arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--solr-url", default="http://localhost:8983/solr", help="Solr server URL")
    parser.add_argument("--storage-dir", default="./experiments", help="Directory for experiment storage")
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # create-experiment command
    create_parser = subparsers.add_parser("create-experiment", help="Create a new experiment")
    create_parser.add_argument("--name", required=True, help="Experiment name")
    create_parser.add_argument("--description", help="Experiment description")
    create_parser.add_argument("--corpus", required=True, help="Solr collection/corpus name")
    create_parser.add_argument("--queries-csv", help="CSV file with queries")
    create_parser.add_argument("--queries-json", help="JSON file with queries")
    create_parser.add_argument("--judgments-csv", help="CSV file with relevance judgments")
    create_parser.add_argument("--judgments-trec", help="TREC format judgments file")
    create_parser.add_argument("--judgments-json", help="JSON file with judgments")
    create_parser.add_argument("--metric", default="ndcg", help="Primary metric (default: ndcg)")
    create_parser.add_argument("--secondary-metrics", nargs="*", help="Secondary metrics")
    create_parser.add_argument("--depth", type=int, default=10, help="Metric depth (default: 10)")
    create_parser.set_defaults(func=cmd_create_experiment)
    
    # run-iteration command
    run_parser = subparsers.add_parser("run-iteration", help="Run an iteration")
    run_parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    run_parser.add_argument("--iteration-id", help="Iteration ID (auto-generated if not provided)")
    run_parser.add_argument("--description", help="Iteration description")
    run_parser.add_argument("--parser", help="Query parser (default: edismax)")
    run_parser.add_argument("--qf", help="Query fields (qf parameter)")
    run_parser.add_argument("--pf", help="Phrase fields (pf parameter)")
    run_parser.add_argument("--mm", help="Minimum should match (mm parameter)")
    run_parser.add_argument("--boost", help="Boost functions")
    run_parser.add_argument("--additional-params", nargs="*", 
                          help="Additional parameters as key=value pairs")
    run_parser.set_defaults(func=cmd_run_iteration)
    
    # compare-iterations command
    compare_parser = subparsers.add_parser("compare-iterations", help="Compare two iterations")
    compare_parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    compare_parser.add_argument("--iteration1", required=True, help="First iteration ID")
    compare_parser.add_argument("--iteration2", required=True, help="Second iteration ID")
    compare_parser.add_argument("--show-details", action="store_true", 
                               help="Show detailed query-level changes")
    compare_parser.set_defaults(func=cmd_compare_iterations)
    
    # list-experiments command
    list_exp_parser = subparsers.add_parser("list-experiments", help="List all experiments")
    list_exp_parser.set_defaults(func=cmd_list_experiments)
    
    # list-iterations command
    list_iter_parser = subparsers.add_parser("list-iterations", help="List iterations for an experiment")
    list_iter_parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    list_iter_parser.set_defaults(func=cmd_list_iterations)
    
    # export-experiment command
    export_parser = subparsers.add_parser("export-experiment", help="Export experiment data")
    export_parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    export_parser.add_argument("--output-file", required=True, help="Output file path")
    export_parser.set_defaults(func=cmd_export_experiment)
    
    # import-experiment command
    import_parser = subparsers.add_parser("import-experiment", help="Import experiment data")
    import_parser.add_argument("--input-file", required=True, help="Input file path")
    import_parser.set_defaults(func=cmd_import_experiment)
    
    # tag-iteration command
    tag_parser = subparsers.add_parser("tag-iteration", help="Tag an iteration")
    tag_parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    tag_parser.add_argument("--iteration-id", required=True, help="Iteration ID")
    tag_parser.add_argument("--tag", required=True, help="Tag name")
    tag_parser.set_defaults(func=cmd_tag_iteration)
    
    # branch-experiment command
    branch_parser = subparsers.add_parser("branch-experiment", help="Create experiment branch")
    branch_parser.add_argument("--experiment-id", required=True, help="Source experiment ID")
    branch_parser.add_argument("--name", required=True, help="Branch name")
    branch_parser.set_defaults(func=cmd_branch_experiment)
    
    # AI commands
    # ai-recommend command
    ai_recommend_parser = subparsers.add_parser("ai-recommend", help="Get AI optimization recommendations")
    ai_recommend_parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    ai_recommend_parser.add_argument("--ai-model", default="openai:gpt-4", help="AI model to use")
    ai_recommend_parser.add_argument("--constraints", nargs="*", help="Optimization constraints as key=value pairs")
    ai_recommend_parser.set_defaults(func=cmd_ai_recommend)
    
    # ai-preview command
    ai_preview_parser = subparsers.add_parser("ai-preview", help="Preview AI optimization recommendation")
    ai_preview_parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    ai_preview_parser.add_argument("--ai-model", default="openai:gpt-4", help="AI model to use")
    ai_preview_parser.add_argument("--constraints", nargs="*", help="Optimization constraints as key=value pairs")
    ai_preview_parser.set_defaults(func=cmd_ai_preview)
    
    # ai-optimize command
    ai_optimize_parser = subparsers.add_parser("ai-optimize", help="Run AI-optimized iteration")
    ai_optimize_parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    ai_optimize_parser.add_argument("--ai-model", default="openai:gpt-4", help="AI model to use")
    ai_optimize_parser.add_argument("--constraints", nargs="*", help="Optimization constraints as key=value pairs")
    ai_optimize_parser.set_defaults(func=cmd_ai_optimize)
    
    # ai-status command
    ai_status_parser = subparsers.add_parser("ai-status", help="Show AI system status")
    ai_status_parser.add_argument("--ai-model", default="openai:gpt-4", help="AI model to use")
    ai_status_parser.set_defaults(func=cmd_ai_status)
    
    # Parse arguments and execute command
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
