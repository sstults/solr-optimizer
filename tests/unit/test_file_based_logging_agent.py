"""
Unit tests for the FileBasedLoggingAgent class.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, Mock

from solr_optimizer.agents.logging.file_based_logging_agent import FileBasedLoggingAgent
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.iteration_result import IterationResult, MetricResult, QueryResult
from solr_optimizer.models.query_config import QueryConfig


class TestFileBasedLoggingAgent:
    """Test cases for the FileBasedLoggingAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.agent = FileBasedLoggingAgent(storage_dir=self.temp_dir)
        
        # Sample experiment config
        self.experiment_config = ExperimentConfig(
            corpus="test_collection",
            queries=["query1", "query2"],
            judgments={
                "query1": {"doc1": 3, "doc2": 1},
                "query2": {"doc3": 2, "doc4": 3}
            },
            primary_metric="ndcg",
            metric_depth=10,
            description="Test experiment"
        )
        
        # Sample iteration result
        self.iteration_result = IterationResult(
            experiment_id="test_exp_123",
            iteration_id="iter_456",
            query_config=QueryConfig(query_parser="edismax"),
            query_results={
                "query1": QueryResult(query="query1", documents=["doc1", "doc2"], scores={"doc1": 0.8, "doc2": 0.6})
            },
            metric_results=[
                MetricResult(metric_name="ndcg@10", value=0.85, per_query={"query1": 0.85})
            ],
            timestamp=datetime.now(timezone.utc)
        )

    def teardown_method(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_default_directory(self):
        """Test initialization with default storage directory."""
        agent = FileBasedLoggingAgent()
        assert agent.storage_dir is not None
        assert agent.storage_dir.exists()

    def test_initialization_custom_directory(self):
        """Test initialization with custom storage directory."""
        assert self.agent.storage_dir == Path(self.temp_dir)
        assert self.agent.storage_dir.exists()
        
        # Check that index file is created
        index_file = self.agent.storage_dir / "index.json"
        assert index_file.exists()

    def test_get_experiment_dir(self):
        """Test experiment directory path generation."""
        experiment_id = "test_exp_123"
        exp_dir = self.agent._get_experiment_dir(experiment_id)
        
        expected_path = Path(self.temp_dir) / "experiments" / experiment_id
        assert exp_dir == expected_path

    def test_get_iterations_dir(self):
        """Test iterations directory path generation."""
        experiment_id = "test_exp_123"
        iter_dir = self.agent._get_iterations_dir(experiment_id)
        
        expected_path = Path(self.temp_dir) / "experiments" / experiment_id / "iterations"
        assert iter_dir == expected_path

    def test_save_experiment(self):
        """Test saving an experiment."""
        # Add experiment ID to config
        config_with_id = ExperimentConfig(
            corpus=self.experiment_config.corpus,
            queries=self.experiment_config.queries,
            judgments=self.experiment_config.judgments,
            primary_metric=self.experiment_config.primary_metric,
            metric_depth=self.experiment_config.metric_depth,
            description=self.experiment_config.description,
            experiment_id="test_exp_123"
        )
        
        result = self.agent.save_experiment(config_with_id)
        
        assert result is True
        
        # Check that experiment directory was created
        exp_dir = self.agent._get_experiment_dir("test_exp_123")
        assert exp_dir.exists()
        
        # Check that experiment config file was created
        config_file = self.agent._get_experiment_config_path("test_exp_123")
        assert config_file.exists()
        
        # Verify config content
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        assert saved_config["corpus"] == "test_collection"
        assert saved_config["experiment_id"] == "test_exp_123"

    def test_get_experiment(self):
        """Test retrieving an experiment."""
        # First save an experiment
        config_with_id = ExperimentConfig(
            corpus=self.experiment_config.corpus,
            queries=self.experiment_config.queries,
            judgments=self.experiment_config.judgments,
            primary_metric=self.experiment_config.primary_metric,
            metric_depth=self.experiment_config.metric_depth,
            description=self.experiment_config.description,
            experiment_id="test_exp_123"
        )
        self.agent.save_experiment(config_with_id)
        
        # Retrieve the experiment
        retrieved_config = self.agent.get_experiment("test_exp_123")
        
        assert retrieved_config is not None
        assert retrieved_config.corpus == "test_collection"
        assert retrieved_config.experiment_id == "test_exp_123"
        assert retrieved_config.queries == ["query1", "query2"]

    def test_get_experiment_not_found(self):
        """Test retrieving a non-existent experiment."""
        result = self.agent.get_experiment("nonexistent_exp")
        assert result is None

    def test_log_iteration(self):
        """Test logging an iteration result."""
        result = self.agent.log_iteration(self.iteration_result)
        
        assert result is True
        
        # Check that iteration file was created
        iter_file = self.agent._get_iteration_path("test_exp_123", "iter_456")
        assert iter_file.exists()
        
        # Verify iteration content
        with open(iter_file, 'r') as f:
            saved_iteration = json.load(f)
        assert saved_iteration["experiment_id"] == "test_exp_123"
        assert saved_iteration["iteration_id"] == "iter_456"

    def test_get_iteration(self):
        """Test retrieving an iteration result."""
        # First log an iteration
        self.agent.log_iteration(self.iteration_result)
        
        # Retrieve the iteration
        retrieved_iteration = self.agent.get_iteration("test_exp_123", "iter_456")
        
        assert retrieved_iteration is not None
        assert retrieved_iteration.experiment_id == "test_exp_123"
        assert retrieved_iteration.iteration_id == "iter_456"
        assert len(retrieved_iteration.query_results) == 1
        assert len(retrieved_iteration.metric_results) == 1

    def test_get_iteration_not_found(self):
        """Test retrieving a non-existent iteration."""
        result = self.agent.get_iteration("test_exp_123", "nonexistent_iter")
        assert result is None

    def test_list_iterations(self):
        """Test listing iterations for an experiment."""
        # Log multiple iterations
        iteration1 = IterationResult(
            experiment_id="test_exp_123",
            iteration_id="iter_001",
            query_config=QueryConfig(),
            query_results={},
            metric_results=[],
            timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        
        iteration2 = IterationResult(
            experiment_id="test_exp_123",
            iteration_id="iter_002",
            query_config=QueryConfig(),
            query_results={},
            metric_results=[],
            timestamp=datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        )
        
        self.agent.log_iteration(iteration1)
        self.agent.log_iteration(iteration2)
        
        # List iterations
        iterations = self.agent.list_iterations("test_exp_123")
        
        assert len(iterations) == 2
        # Should be sorted by timestamp (most recent first)
        assert iterations[0]["iteration_id"] == "iter_002"  # More recent
        assert iterations[1]["iteration_id"] == "iter_001"  # Earlier

    def test_list_experiments(self):
        """Test listing all experiments."""
        # Save multiple experiments
        config1 = ExperimentConfig(
            corpus="collection1",
            queries=["q1"],
            judgments={"q1": {"doc1": 1}},
            primary_metric="ndcg",
            metric_depth=10,
            experiment_id="exp_001"
        )
        
        config2 = ExperimentConfig(
            corpus="collection2",
            queries=["q2"],
            judgments={"q2": {"doc2": 2}},
            primary_metric="precision",
            metric_depth=5,
            experiment_id="exp_002"
        )
        
        self.agent.save_experiment(config1)
        self.agent.save_experiment(config2)
        
        # List experiments
        experiments = self.agent.list_experiments()
        
        assert len(experiments) == 2
        exp_ids = [exp["experiment_id"] for exp in experiments]
        assert "exp_001" in exp_ids
        assert "exp_002" in exp_ids

    def test_tag_iteration(self):
        """Test tagging an iteration."""
        # First log an iteration
        self.agent.log_iteration(self.iteration_result)
        
        # Tag the iteration
        result = self.agent.tag_iteration("test_exp_123", "iter_456", "baseline")
        
        assert result is True
        
        # Check that tags file was created/updated
        tags_file = self.agent._get_tags_path("test_exp_123")
        assert tags_file.exists()
        
        # Verify tag content
        with open(tags_file, 'r') as f:
            tags_data = json.load(f)
        assert "iter_456" in tags_data
        assert "baseline" in tags_data["iter_456"]

    def test_tag_iteration_nonexistent(self):
        """Test tagging a non-existent iteration."""
        result = self.agent.tag_iteration("test_exp_123", "nonexistent_iter", "tag")
        assert result is False

    def test_branch_experiment(self):
        """Test branching an experiment."""
        # First save the original experiment
        config_with_id = ExperimentConfig(
            corpus=self.experiment_config.corpus,
            queries=self.experiment_config.queries,
            judgments=self.experiment_config.judgments,
            primary_metric=self.experiment_config.primary_metric,
            metric_depth=self.experiment_config.metric_depth,
            description=self.experiment_config.description,
            experiment_id="test_exp_123"
        )
        self.agent.save_experiment(config_with_id)
        
        # Branch the experiment
        new_exp_id = self.agent.branch_experiment("test_exp_123", "branched_exp_456", "Branched experiment")
        
        assert new_exp_id is not None
        assert new_exp_id == "branched_exp_456"
        
        # Check that new experiment was created
        new_config = self.agent.get_experiment(new_exp_id)
        assert new_config is not None
        assert new_config.corpus == "test_collection"  # Same as original

    def test_archive_experiment(self):
        """Test archiving an experiment."""
        # Save an experiment first
        config_with_id = ExperimentConfig(
            corpus=self.experiment_config.corpus,
            queries=self.experiment_config.queries,
            judgments=self.experiment_config.judgments,
            primary_metric=self.experiment_config.primary_metric,
            metric_depth=self.experiment_config.metric_depth,
            description=self.experiment_config.description,
            experiment_id="test_exp_123"
        )
        self.agent.save_experiment(config_with_id)
        
        # Archive the experiment
        result = self.agent.archive_experiment("test_exp_123")
        
        assert result is True
        
        # Check that experiment is marked as archived in index
        index_file = self.agent.storage_dir / "index.json"
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        assert "test_exp_123" in index_data["experiments"]
        exp_entry = index_data["experiments"]["test_exp_123"]
        assert exp_entry["archived"] is True
        assert "archive_date" in exp_entry

    def test_export_experiment(self):
        """Test exporting an experiment."""
        # Save an experiment and log an iteration
        config_with_id = ExperimentConfig(
            corpus=self.experiment_config.corpus,
            queries=self.experiment_config.queries,
            judgments=self.experiment_config.judgments,
            primary_metric=self.experiment_config.primary_metric,
            metric_depth=self.experiment_config.metric_depth,
            description=self.experiment_config.description,
            experiment_id="test_exp_123"
        )
        self.agent.save_experiment(config_with_id)
        self.agent.log_iteration(self.iteration_result)
        
        # Export to a temporary JSON file
        export_file = Path(self.temp_dir) / "export.json"
        result = self.agent.export_experiment("test_exp_123", str(export_file))
        
        assert result is True
        assert export_file.exists()
        assert export_file.stat().st_size > 0
        
        # Verify export content
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        assert export_data["experiment_id"] == "test_exp_123"
        assert "config" in export_data
        assert "iterations" in export_data

    def test_import_experiment(self):
        """Test importing an experiment."""
        # First export an experiment
        config_with_id = ExperimentConfig(
            corpus=self.experiment_config.corpus,
            queries=self.experiment_config.queries,
            judgments=self.experiment_config.judgments,
            primary_metric=self.experiment_config.primary_metric,
            metric_depth=self.experiment_config.metric_depth,
            description=self.experiment_config.description,
            experiment_id="test_exp_123"
        )
        self.agent.save_experiment(config_with_id)
        self.agent.log_iteration(self.iteration_result)
        
        export_file = Path(self.temp_dir) / "export.json"
        self.agent.export_experiment("test_exp_123", str(export_file))
        
        # Remove the original experiment
        exp_dir = self.agent._get_experiment_dir("test_exp_123")
        shutil.rmtree(exp_dir)
        
        # Import the experiment
        imported_exp_id = self.agent.import_experiment(str(export_file))
        
        assert imported_exp_id is not None
        
        # Check that experiment was restored
        imported_config = self.agent.get_experiment(imported_exp_id)
        assert imported_config is not None
        assert imported_config.corpus == "test_collection"

    def test_file_write_error_handling(self):
        """Test handling of file write errors."""
        # Use mocking to simulate write errors instead of filesystem permissions
        with patch.object(Path, 'mkdir', side_effect=PermissionError("Permission denied")):
            config_with_id = ExperimentConfig(
                corpus="test",
                queries=["q1"],
                judgments={"q1": {"doc1": 1}},
                primary_metric="ndcg",
                metric_depth=10,
                experiment_id="test_exp"
            )
            
            result = self.agent.save_experiment(config_with_id)
            # Should return False due to permission error
            assert result is False

    def test_corrupted_json_handling(self):
        """Test handling of corrupted JSON files."""
        # Create a corrupted JSON file
        exp_dir = self.agent._get_experiment_dir("test_exp_123")
        exp_dir.mkdir(parents=True)
        
        config_file = self.agent._get_experiment_config_path("test_exp_123")
        with open(config_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle corrupted file gracefully
        result = self.agent.get_experiment("test_exp_123")
        assert result is None

    def test_update_index(self):
        """Test index file updates."""
        # Save an experiment
        config_with_id = ExperimentConfig(
            corpus=self.experiment_config.corpus,
            queries=self.experiment_config.queries,
            judgments=self.experiment_config.judgments,
            primary_metric=self.experiment_config.primary_metric,
            metric_depth=self.experiment_config.metric_depth,
            description=self.experiment_config.description,
            experiment_id="test_exp_123"
        )
        self.agent.save_experiment(config_with_id)
        
        # Check that index was updated
        index_file = self.agent.storage_dir / "index.json"
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        assert "test_exp_123" in index_data["experiments"]
        exp_entry = index_data["experiments"]["test_exp_123"]
        assert exp_entry["metadata"]["corpus"] == "test_collection"
        assert "created_at" in exp_entry
