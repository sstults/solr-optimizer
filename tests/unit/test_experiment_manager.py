"""
Unit tests for the DefaultExperimentManager class.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import tempfile
import shutil
from pathlib import Path

from solr_optimizer.core.default_experiment_manager import DefaultExperimentManager
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.query_config import QueryConfig
from solr_optimizer.models.iteration_result import IterationResult, MetricResult, QueryResult


class TestDefaultExperimentManager:
    """Test cases for the DefaultExperimentManager class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock agents
        self.mock_query_tuning_agent = Mock()
        self.mock_solr_agent = Mock()
        self.mock_metrics_agent = Mock()
        self.mock_comparison_agent = Mock()
        self.mock_logging_agent = Mock()
        
        # Create experiment manager with mocked agents
        self.manager = DefaultExperimentManager(
            query_tuning_agent=self.mock_query_tuning_agent,
            solr_execution_agent=self.mock_solr_agent,
            metrics_agent=self.mock_metrics_agent,
            comparison_agent=self.mock_comparison_agent,
            logging_agent=self.mock_logging_agent
        )
        
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
        
        # Sample query config
        self.query_config = QueryConfig(
            query_parser="edismax",
            query_fields={"title": 2.0, "content": 1.0}
        )

    def teardown_method(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test DefaultExperimentManager initialization."""
        assert self.manager.query_tuning_agent == self.mock_query_tuning_agent
        assert self.manager.solr_execution_agent == self.mock_solr_agent
        assert self.manager.metrics_agent == self.mock_metrics_agent
        assert self.manager.comparison_agent == self.mock_comparison_agent
        assert self.manager.logging_agent == self.mock_logging_agent

    def test_setup_experiment(self):
        """Test experiment setup."""
        self.mock_logging_agent.save_experiment.return_value = True
        
        experiment_id = self.manager.setup_experiment(self.experiment_config)
        
        assert experiment_id is not None
        assert len(experiment_id) > 0
        self.mock_logging_agent.save_experiment.assert_called_once()
        
        # Verify the experiment config was saved with the generated ID
        saved_config = self.mock_logging_agent.save_experiment.call_args[0][0]
        assert saved_config.corpus == "test_collection"
        assert saved_config.queries == ["query1", "query2"]

    def test_setup_experiment_logging_failure(self):
        """Test experiment setup when logging fails."""
        self.mock_logging_agent.save_experiment.return_value = False
        
        with pytest.raises(RuntimeError, match="Failed to save experiment"):
            self.manager.setup_experiment(self.experiment_config)

    def test_run_iteration_success(self):
        """Test successful iteration run."""
        experiment_id = "test_exp_123"
        
        # Mock getting experiment
        self.mock_logging_agent.get_experiment.return_value = self.experiment_config
        
        # Mock Solr execution response
        mock_query_results = {
            "query1": {"documents": ["doc1", "doc2"], "scores": {"doc1": 1.0, "doc2": 0.8}},
            "query2": {"documents": ["doc3", "doc4"], "scores": {"doc3": 0.9, "doc4": 0.7}}
        }
        self.mock_solr_agent.execute_queries.return_value = mock_query_results
        
        # Mock metrics calculation
        mock_metric_results = [
            MetricResult(metric_name="ndcg@10", value=0.85, per_query={"query1": 0.9, "query2": 0.8})
        ]
        self.mock_metrics_agent.calculate_metrics.return_value = mock_metric_results
        
        # Mock logging
        self.mock_logging_agent.log_iteration.return_value = True
        self.mock_logging_agent.list_iterations.return_value = []  # No previous iterations
        
        result = self.manager.run_iteration(experiment_id, self.query_config)
        
        assert result is not None
        assert result.experiment_id == experiment_id
        assert result.query_config == self.query_config
        assert len(result.query_results) == 2
        assert len(result.metric_results) == 1
        
        # Verify agent calls
        self.mock_solr_agent.execute_queries.assert_called_once()
        self.mock_metrics_agent.calculate_metrics.assert_called_once()
        self.mock_logging_agent.log_iteration.assert_called_once()

    def test_run_iteration_solr_failure(self):
        """Test iteration run when Solr execution fails."""
        # Mock getting experiment
        self.mock_logging_agent.get_experiment.return_value = self.experiment_config
        self.mock_solr_agent.execute_queries.side_effect = Exception("Solr connection failed")
        
        with pytest.raises(Exception, match="Solr connection failed"):
            self.manager.run_iteration("test_exp", self.query_config)

    def test_run_iteration_logging_failure(self):
        """Test iteration run when logging fails."""
        experiment_id = "test_exp_123"
        
        # Mock getting experiment
        self.mock_logging_agent.get_experiment.return_value = self.experiment_config
        
        # Mock successful Solr and metrics
        self.mock_solr_agent.execute_queries.return_value = {
            "query1": {"documents": ["doc1"], "scores": {"doc1": 1.0}}
        }
        self.mock_metrics_agent.calculate_metrics.return_value = [
            MetricResult(metric_name="ndcg@10", value=0.5, per_query={"query1": 0.5})
        ]
        self.mock_logging_agent.list_iterations.return_value = []
        
        # Mock logging failure
        self.mock_logging_agent.log_iteration.return_value = False
        
        # Should not raise an exception, just log a warning
        result = self.manager.run_iteration(experiment_id, self.query_config)
        assert result is not None

    def test_compare_iterations(self):
        """Test iteration comparison."""
        experiment_id = "test_exp"
        iteration_id1 = "iter1"
        iteration_id2 = "iter2"
        
        # Mock iteration results
        mock_iteration1 = IterationResult(
            experiment_id=experiment_id,
            iteration_id=iteration_id1,
            query_config=self.query_config,
            query_results={},
            metric_results=[MetricResult("ndcg@10", 0.8, {})],
            timestamp=datetime.now(timezone.utc)
        )
        
        mock_iteration2 = IterationResult(
            experiment_id=experiment_id,
            iteration_id=iteration_id2,
            query_config=self.query_config,
            query_results={},
            metric_results=[MetricResult("ndcg@10", 0.9, {})],
            timestamp=datetime.now(timezone.utc)
        )
        
        self.mock_logging_agent.get_iteration.side_effect = [mock_iteration1, mock_iteration2]
        self.mock_comparison_agent.generate_summary_report.return_value = {"improvement": 0.1}
        
        result = self.manager.compare_iterations(experiment_id, iteration_id1, iteration_id2)
        
        assert result is not None
        assert "improvement" in result
        self.mock_comparison_agent.generate_summary_report.assert_called_once()

    def test_compare_iterations_missing_iteration(self):
        """Test comparison when one iteration is missing."""
        self.mock_logging_agent.get_iteration.side_effect = [None, Mock()]
        
        with pytest.raises(ValueError, match="Iterations not found"):
            self.manager.compare_iterations("exp", "iter1", "iter2")

    def test_get_iteration_history(self):
        """Test getting iteration history."""
        experiment_id = "test_exp"
        mock_history_summaries = [
            {"iteration_id": "iter1", "timestamp": "2023-01-01T00:00:00Z"},
            {"iteration_id": "iter2", "timestamp": "2023-01-02T00:00:00Z"}
        ]
        
        mock_iterations = [
            IterationResult(
                experiment_id=experiment_id,
                iteration_id="iter1",
                query_config=self.query_config,
                query_results={},
                metric_results=[],
                timestamp=datetime.now(timezone.utc)
            ),
            IterationResult(
                experiment_id=experiment_id,
                iteration_id="iter2",
                query_config=self.query_config,
                query_results={},
                metric_results=[],
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        self.mock_logging_agent.list_iterations.return_value = mock_history_summaries
        self.mock_logging_agent.get_iteration.side_effect = mock_iterations
        
        history = self.manager.get_iteration_history(experiment_id)
        
        assert len(history) == 2
        assert history[0].iteration_id == "iter1"
        assert history[1].iteration_id == "iter2"
        self.mock_logging_agent.list_iterations.assert_called_once_with(experiment_id)

    def test_get_current_state_with_iterations(self):
        """Test getting current state when iterations exist."""
        experiment_id = "test_exp"
        mock_iteration = IterationResult(
            experiment_id=experiment_id,
            iteration_id="latest_iter",
            query_config=self.query_config,
            query_results={},
            metric_results=[],
            timestamp=datetime.now(timezone.utc)
        )
        
        self.mock_logging_agent.list_iterations.return_value = [
            {"iteration_id": "latest_iter", "timestamp": "2023-01-02T00:00:00Z"}
        ]
        self.mock_logging_agent.get_iteration.return_value = mock_iteration
        
        current_state = self.manager.get_current_state(experiment_id)
        
        assert current_state == mock_iteration

    def test_get_current_state_no_iterations(self):
        """Test getting current state when no iterations exist."""
        self.mock_logging_agent.list_iterations.return_value = []
        
        current_state = self.manager.get_current_state("test_exp")
        
        assert current_state is None

    def test_set_queries(self):
        """Test setting queries."""
        new_queries = ["new_query1", "new_query2"]
        
        self.manager.set_queries(new_queries)
        
        assert self.manager.current_queries == new_queries

    def test_unique_experiment_id_generation(self):
        """Test that experiment IDs are unique."""
        self.mock_logging_agent.save_experiment.return_value = True
        
        exp_id1 = self.manager.setup_experiment(self.experiment_config)
        exp_id2 = self.manager.setup_experiment(self.experiment_config)
        
        assert exp_id1 != exp_id2

    def test_iteration_id_generation(self):
        """Test iteration ID generation."""
        experiment_id = "test_exp"
        
        # Mock getting experiment
        self.mock_logging_agent.get_experiment.return_value = self.experiment_config
        
        # Mock successful execution
        self.mock_solr_agent.execute_queries.return_value = {
            "query1": {"documents": [], "scores": {}}
        }
        self.mock_metrics_agent.calculate_metrics.return_value = []
        self.mock_logging_agent.log_iteration.return_value = True
        self.mock_logging_agent.list_iterations.return_value = []
        
        result = self.manager.run_iteration(experiment_id, self.query_config)
        
        # Should have generated an iteration ID
        assert result.iteration_id is not None
        assert len(result.iteration_id) > 0

    def test_primary_metric_extraction(self):
        """Test that primary metric is correctly extracted during iteration."""
        experiment_id = "test_exp"
        
        # Mock getting experiment
        self.mock_logging_agent.get_experiment.return_value = self.experiment_config
        
        # Set up the manager with queries that match the experiment config
        self.manager.set_queries(["query1", "query2"])
        
        # Mock Solr execution
        self.mock_solr_agent.execute_queries.return_value = {
            "query1": {"documents": ["doc1"], "scores": {"doc1": 1.0}},
            "query2": {"documents": ["doc2"], "scores": {"doc2": 0.9}}
        }
        
        # Mock metrics with specific primary metric
        mock_metrics = [
            MetricResult(metric_name="ndcg@10", value=0.85, per_query={"query1": 0.9, "query2": 0.8}),
            MetricResult(metric_name="precision@5", value=0.7, per_query={"query1": 0.75, "query2": 0.65})
        ]
        self.mock_metrics_agent.calculate_metrics.return_value = mock_metrics
        self.mock_logging_agent.log_iteration.return_value = True
        self.mock_logging_agent.list_iterations.return_value = []
        
        result = self.manager.run_iteration(experiment_id, self.query_config)
        
        # Should have metric results
        assert len(result.metric_results) == 2
        assert result.metric_results[0].metric_name == "ndcg@10"
