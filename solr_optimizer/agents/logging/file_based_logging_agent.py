"""
File-Based Logging Agent - Implementation of the LoggingAgent.

This module provides a concrete implementation of the LoggingAgent interface
that stores experiment history and configuration as JSON files in a structured
directory.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from solr_optimizer.agents.logging.logging_agent import LoggingAgent
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.iteration_result import IterationResult

logger = logging.getLogger(__name__)


class FileBasedLoggingAgent(LoggingAgent):
    """
    File-based implementation of the LoggingAgent interface.

    This agent stores experiment data in a structured directory hierarchy:

    storage_dir/
    ├── experiments/
    │   ├── experiment_id_1/
    │   │   ├── config.json
    │   │   ├── iterations/
    │   │   │   ├── iteration_id_1.json
    │   │   │   ├── iteration_id_2.json
    │   │   │   └── ...
    │   │   └── tags.json
    │   ├── experiment_id_2/
    │   │   └── ...
    │   └── ...
    └── index.json
    """

    def __init__(self, storage_dir: str = "experiment_storage"):
        """
        Initialize the FileBasedLoggingAgent.

        Args:
            storage_dir: Directory where experiment data will be stored
        """
        self.storage_dir = Path(storage_dir)
        self.experiments_dir = self.storage_dir / "experiments"
        self.index_path = self.storage_dir / "index.json"

        # Create directory structure if it doesn't exist
        self._init_storage()

    def _init_storage(self):
        """Initialize storage directory structure."""
        # Create main storage directory
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Create index file if it doesn't exist
        if not self.index_path.exists():
            self._write_json(self.index_path, {"experiments": {}})

    def _get_experiment_dir(self, experiment_id: str) -> Path:
        """Get the directory for a specific experiment."""
        return self.experiments_dir / experiment_id

    def _get_iterations_dir(self, experiment_id: str) -> Path:
        """Get the iterations directory for a specific experiment."""
        return self._get_experiment_dir(experiment_id) / "iterations"

    def _get_experiment_config_path(self, experiment_id: str) -> Path:
        """Get the path to the experiment config file."""
        return self._get_experiment_dir(experiment_id) / "config.json"

    def _get_iteration_path(self, experiment_id: str,
                            iteration_id: str) -> Path:
        """Get the path to a specific iteration file."""
        return self._get_iterations_dir(experiment_id) / f"{iteration_id}.json"

    def _get_tags_path(self, experiment_id: str) -> Path:
        """Get the path to the tags file for an experiment."""
        return self._get_experiment_dir(experiment_id) / "tags.json"

    def _read_json(self, path: Path) -> Dict:
        """Read JSON from a file."""
        try:
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error reading JSON from {path}: {e}")
            return {}

    def _write_json(self, path: Path, data: Dict) -> bool:
        """Write JSON to a file."""
        try:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error writing JSON to {path}: {e}")
            return False

    def _update_index(
        self, experiment_id: str, name: str = None, metadata: Dict = None
    ) -> bool:
        """Update the index file with experiment information."""
        index = self._read_json(self.index_path)

        if "experiments" not in index:
            index["experiments"] = {}

        if experiment_id not in index["experiments"]:
            index["experiments"][experiment_id] = {
                "name": name or experiment_id,
                "created_at": datetime.datetime.now().isoformat(),
                "metadata": metadata or {},
            }
        elif name is not None or metadata is not None:
            if name is not None:
                index["experiments"][experiment_id]["name"] = name
            if metadata is not None:
                index["experiments"][experiment_id]["metadata"].update(
                    metadata)

        # Update last modified timestamp
        index["experiments"][experiment_id][
            "last_modified"
        ] = datetime.datetime.now().isoformat()

        return self._write_json(self.index_path, index)

    def log_iteration(self, iteration_result: IterationResult) -> bool:
        """
        Log an iteration result to storage.

        Args:
            iteration_result: The iteration result to log

        Returns:
            True if logging was successful, False otherwise
        """
        experiment_id = iteration_result.experiment_id
        iteration_id = iteration_result.iteration_id

        # Create iterations directory if it doesn't exist
        iterations_dir = self._get_iterations_dir(experiment_id)
        iterations_dir.mkdir(parents=True, exist_ok=True)

        # Update the experiment index
        self._update_index(
            experiment_id, metadata={
                "last_iteration": iteration_id})

        # Prepare iteration data for storage
        iteration_data = iteration_result.dict()

        # Record timestamp if not already present
        if "timestamp" not in iteration_data:
            iteration_data["timestamp"] = datetime.datetime.now().isoformat()

        # Write iteration data
        iteration_path = self._get_iteration_path(experiment_id, iteration_id)
        return self._write_json(iteration_path, iteration_data)

    def get_iteration(
        self, experiment_id: str, iteration_id: str
    ) -> Optional[IterationResult]:
        """
        Retrieve a specific iteration result from storage.

        Args:
            experiment_id: The experiment ID
            iteration_id: The iteration ID

        Returns:
            The iteration result, or None if not found
        """
        iteration_path = self._get_iteration_path(experiment_id, iteration_id)

        if not iteration_path.exists():
            return None

        iteration_data = self._read_json(iteration_path)

        try:
            return IterationResult(**iteration_data)
        except Exception as e:
            logger.error(f"Error deserializing iteration {iteration_id}: {e}")
            return None

    def list_iterations(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        List all iterations for an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            List of iteration summary dictionaries sorted by timestamp
            (most recent first)
        """
        iterations_dir = self._get_iterations_dir(experiment_id)

        if not iterations_dir.exists():
            return []

        iteration_summaries = []

        # Collect all iteration files
        for iteration_file in iterations_dir.glob("*.json"):
            iteration_id = iteration_file.stem
            iteration_data = self._read_json(iteration_file)

            # Create a summary with essential information
            summary = {
                "iteration_id": iteration_id,
                "timestamp": iteration_data.get("timestamp", "unknown"),
            }

            # Add query config summary
            if "query_config" in iteration_data:
                summary["query_config"] = {
                    k: iteration_data["query_config"].get(k)
                    for k in ["description", "query_parser", "iteration_id"]
                    if k in iteration_data["query_config"]
                }

            # Add primary metric if available
            if (
                "metric_results" in iteration_data
                and "overall" in iteration_data["metric_results"]
            ):
                metrics = iteration_data["metric_results"]["overall"]
                if metrics:
                    # Just include the first metric for the summary
                    first_metric = next(iter(metrics.items()))
                    summary["metric"] = {first_metric[0]: first_metric[1]}

            # Add tags if available
            tags_data = self._read_json(self._get_tags_path(experiment_id))
            if iteration_id in tags_data:
                summary["tags"] = tags_data[iteration_id]

            iteration_summaries.append(summary)

        # Sort by timestamp (most recent first)
        iteration_summaries.sort(
            key=lambda x: x.get(
                "timestamp", ""), reverse=True)

        return iteration_summaries

    def save_experiment(self, experiment: ExperimentConfig) -> bool:
        """
        Save an experiment configuration to storage.

        Args:
            experiment: The experiment configuration

        Returns:
            True if saving was successful, False otherwise
        """
        experiment_id = experiment.experiment_id

        # Create experiment directory if it doesn't exist
        experiment_dir = self._get_experiment_dir(experiment_id)
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create iterations directory
        iterations_dir = self._get_iterations_dir(experiment_id)
        iterations_dir.mkdir(exist_ok=True)

        # Create empty tags file if it doesn't exist
        tags_path = self._get_tags_path(experiment_id)
        if not tags_path.exists():
            self._write_json(tags_path, {})

        # Update the experiment index
        self._update_index(
            experiment_id,
            name=experiment.name or experiment_id,
            metadata={
                "corpus": experiment.corpus,
                "primary_metric": experiment.primary_metric,
                "num_queries": len(experiment.queries),
                "description": experiment.description,
            },
        )

        # Write experiment configuration
        config_path = self._get_experiment_config_path(experiment_id)
        return self._write_json(config_path, experiment.dict())

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """
        Retrieve an experiment configuration from storage.

        Args:
            experiment_id: The experiment ID

        Returns:
            The experiment configuration, or None if not found
        """
        config_path = self._get_experiment_config_path(experiment_id)

        if not config_path.exists():
            return None

        config_data = self._read_json(config_path)

        try:
            return ExperimentConfig(**config_data)
        except Exception as e:
            logger.error(
                f"Error deserializing experiment {experiment_id}: {e}")
            return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.

        Returns:
            List of experiment summary dictionaries
        """
        index = self._read_json(self.index_path)

        experiments = []
        for experiment_id, info in index.get("experiments", {}).items():
            experiment_summary = {
                "experiment_id": experiment_id,
                "name": info.get("name", experiment_id),
                "created_at": info.get("created_at", "unknown"),
                "last_modified": info.get("last_modified", "unknown"),
                "metadata": info.get("metadata", {}),
            }
            experiments.append(experiment_summary)

        # Sort by last_modified (most recent first)
        experiments.sort(
            key=lambda x: x.get(
                "last_modified",
                ""),
            reverse=True)

        return experiments

    def tag_iteration(self, experiment_id: str,
                      iteration_id: str, tag: str) -> bool:
        """
        Tag an iteration with a user-friendly name or category.

        Args:
            experiment_id: The experiment ID
            iteration_id: The iteration ID
            tag: The tag to apply

        Returns:
            True if tagging was successful, False otherwise
        """
        tags_path = self._get_tags_path(experiment_id)
        tags_data = self._read_json(tags_path)

        # Ensure the iteration exists
        iteration_path = self._get_iteration_path(experiment_id, iteration_id)
        if not iteration_path.exists():
            logger.error(f"Cannot tag non-existent iteration: {iteration_id}")
            return False

        # Add or update tags
        if iteration_id not in tags_data:
            tags_data[iteration_id] = []

        if tag not in tags_data[iteration_id]:
            tags_data[iteration_id].append(tag)

        return self._write_json(tags_path, tags_data)

    def branch_experiment(
        self,
        source_experiment_id: str,
        new_experiment_id: str = None,
        name: str = None
    ) -> Optional[str]:
        """
        Branch an experiment to create a new experiment with the same
        configuration.

        Args:
            source_experiment_id: The ID of the source experiment
            new_experiment_id: Optional ID for the new experiment
            name: Optional name for the new experiment

        Returns:
            The ID of the new experiment, or None if branching failed
        """
        # Get the source experiment
        source_experiment = self.get_experiment(source_experiment_id)
        if not source_experiment:
            logger.error(
                f"Source experiment not found: {source_experiment_id}")
            return None

        # Create a copy of the experiment with a new ID
        new_id = (
            new_experiment_id
            or f"{source_experiment_id}-branch-"
            f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        branched_experiment = ExperimentConfig(
            **{
                **source_experiment.dict(),
                "experiment_id": new_id,
                "name": name or f"Branch of {source_experiment_id}",
            }
        )

        # Add branch metadata
        branched_experiment.metadata = {
            **(branched_experiment.metadata or {}),
            "branched_from": source_experiment_id,
            "branched_at": datetime.datetime.now().isoformat(),
        }

        # Save the new experiment
        success = self.save_experiment(branched_experiment)

        return new_id if success else None

    def archive_experiment(self, experiment_id: str) -> bool:
        """
        Archive an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            True if archiving was successful, False otherwise
        """
        index = self._read_json(self.index_path)

        if ("experiments" not in index or
                experiment_id not in index["experiments"]):
            logger.error(f"Experiment not found: {experiment_id}")
            return False

        # Mark as archived in the index
        index["experiments"][experiment_id]["archived"] = True
        index["experiments"][experiment_id][
            "archive_date"
        ] = datetime.datetime.now().isoformat()

        return self._write_json(self.index_path, index)

    def export_experiment(self, experiment_id: str, target_path: str) -> bool:
        """
        Export an experiment to a single JSON file.

        Args:
            experiment_id: The experiment ID
            target_path: Path where to export the experiment

        Returns:
            True if export was successful, False otherwise
        """
        export_data = {
            "experiment_id": experiment_id,
            "config": (
                self.get_experiment(experiment_id).dict()
                if self.get_experiment(experiment_id)
                else None
            ),
            "iterations": {},
            "tags": self._read_json(self._get_tags_path(experiment_id)),
            "export_date": datetime.datetime.now().isoformat(),
        }

        # Add all iterations
        iterations_dir = self._get_iterations_dir(experiment_id)
        if iterations_dir.exists():
            for iteration_file in iterations_dir.glob("*.json"):
                iteration_id = iteration_file.stem
                export_data["iterations"][iteration_id] = self._read_json(
                    iteration_file
                )

        # Write the export file
        try:
            with open(target_path, "w") as f:
                json.dump(export_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error exporting experiment to {target_path}: {e}")
            return False

    def import_experiment(self, source_path: str) -> Optional[str]:
        """
        Import an experiment from a JSON file.

        Args:
            source_path: Path to the export file

        Returns:
            The ID of the imported experiment, or None if import failed
        """
        try:
            with open(source_path, "r") as f:
                import_data = json.load(f)

            experiment_id = import_data.get("experiment_id")
            if not experiment_id:
                logger.error("Import file does not contain an experiment_id")
                return None

            # Check if experiment already exists
            existing_experiment = self._get_experiment_config_path(
                experiment_id)
            if existing_experiment.exists():
                # Generate a new ID
                import_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                experiment_id = f"{experiment_id}-import-{import_time}"

            # Create experiment config
            config_data = import_data.get("config")
            if not config_data:
                logger.error(
                    "Import file does not contain experiment configuration")
                return None

            # Update experiment ID in config
            config_data["experiment_id"] = experiment_id

            # Create experiment
            experiment = ExperimentConfig(**config_data)
            self.save_experiment(experiment)

            # Import iterations
            iterations_dir = self._get_iterations_dir(experiment_id)
            iterations_dir.mkdir(parents=True, exist_ok=True)

            for iteration_id, iteration_data in import_data.get(
                "iterations", {}
            ).items():
                iteration_path = self._get_iteration_path(
                    experiment_id, iteration_id)
                self._write_json(iteration_path, iteration_data)

            # Import tags
            tags_path = self._get_tags_path(experiment_id)
            self._write_json(tags_path, import_data.get("tags", {}))

            return experiment_id
        except Exception as e:
            logger.error(
                f"Error importing experiment from {source_path}: "
                f"{e}"
            )
            return None
