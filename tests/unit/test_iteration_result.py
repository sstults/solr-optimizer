"""
Unit tests for the IterationResult and related model classes.
"""

import pytest
from datetime import datetime, timezone

from solr_optimizer.models.iteration_result import IterationResult, MetricResult, QueryResult
from solr_optimizer.models.query_config import QueryConfig


class TestQueryResult:
    """Test cases for the QueryResult class."""

    def test_query_result_creation(self):
        """Test basic QueryResult creation."""
        result = QueryResult(
            query="test query",
            documents=["doc1", "doc2", "doc3"],
            scores={"doc1": 1.0, "doc2": 0.8, "doc3": 0.5}
        )
        
        assert result.query == "test query"
        assert result.documents == ["doc1", "doc2", "doc3"]
        assert result.scores == {"doc1": 1.0, "doc2": 0.8, "doc3": 0.5}
        assert result.explain_info is None

    def test_query_result_optional_fields(self):
        """Test QueryResult with optional fields."""
        result = QueryResult(
            query="test query",
            documents=["doc1"],
            scores={"doc1": 1.0},
            explain_info={"doc1": "explanation"}
        )
        
        assert result.query == "test query"
        assert result.documents == ["doc1"]
        assert result.scores == {"doc1": 1.0}
        assert result.explain_info == {"doc1": "explanation"}


class TestMetricResult:
    """Test cases for the MetricResult class."""

    def test_metric_result_creation(self):
        """Test basic MetricResult creation."""
        result = MetricResult(
            metric_name="ndcg@10",
            value=0.85,
            per_query={"q1": 0.9, "q2": 0.8}
        )
        
        assert result.metric_name == "ndcg@10"
        assert result.value == 0.85
        assert result.per_query == {"q1": 0.9, "q2": 0.8}

    def test_metric_result_without_per_query(self):
        """Test MetricResult without per_query data."""
        result = MetricResult(
            metric_name="precision@5",
            value=0.7
        )
        
        assert result.metric_name == "precision@5"
        assert result.value == 0.7
        assert result.per_query == {}


class TestIterationResult:
    """Test cases for the IterationResult class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.query_config = QueryConfig(
            query_parser="edismax",
            query_fields={"title": 2.0, "content": 1.0}
        )
        
        self.query_results = {
            "query1": QueryResult(
                query="query1",
                documents=["doc1", "doc2"],
                scores={"doc1": 1.0, "doc2": 0.8}
            ),
            "query2": QueryResult(
                query="query2", 
                documents=["doc3", "doc4"],
                scores={"doc3": 0.9, "doc4": 0.7}
            )
        }
        
        self.metric_results = [
            MetricResult(
                metric_name="ndcg@10",
                value=0.85,
                per_query={"query1": 0.9, "query2": 0.8}
            ),
            MetricResult(
                metric_name="precision@5",
                value=0.7,
                per_query={"query1": 0.75, "query2": 0.65}
            )
        ]

    def test_iteration_result_creation(self):
        """Test basic IterationResult creation."""
        timestamp = datetime.now(timezone.utc)
        
        result = IterationResult(
            experiment_id="exp123",
            iteration_id="iter456",
            query_config=self.query_config,
            query_results=self.query_results,
            metric_results=self.metric_results,
            timestamp=timestamp
        )
        
        assert result.experiment_id == "exp123"
        assert result.iteration_id == "iter456"
        assert result.query_config == self.query_config
        assert result.query_results == self.query_results
        assert result.metric_results == self.metric_results
        assert result.timestamp == timestamp

    def test_get_primary_metric(self):
        """Test getting the primary metric."""
        result = IterationResult(
            experiment_id="exp123",
            iteration_id="iter456",
            query_config=self.query_config,
            query_results=self.query_results,
            metric_results=self.metric_results,
            timestamp=datetime.now(timezone.utc)
        )
        
        primary = result.get_primary_metric()
        assert primary is not None
        assert primary.metric_name == "ndcg@10"
        assert primary.value == 0.85

    def test_get_primary_metric_not_found(self):
        """Test getting primary metric when no metrics exist."""
        result = IterationResult(
            experiment_id="exp123",
            iteration_id="iter456",
            query_config=self.query_config,
            query_results=self.query_results,
            metric_results=[],  # No metrics
            timestamp=datetime.now(timezone.utc)
        )
        
        primary = result.get_primary_metric()
        assert primary is None

    def test_get_metric_by_name(self):
        """Test getting a metric by name."""
        result = IterationResult(
            experiment_id="exp123",
            iteration_id="iter456",
            query_config=self.query_config,
            query_results=self.query_results,
            metric_results=self.metric_results,
            timestamp=datetime.now(timezone.utc)
        )
        
        precision = result.get_metric_by_name("precision@5")
        assert precision is not None
        assert precision.metric_name == "precision@5"
        assert precision.value == 0.7

    def test_get_metric_by_name_not_found(self):
        """Test getting a metric by name when it doesn't exist."""
        result = IterationResult(
            experiment_id="exp123",
            iteration_id="iter456",
            query_config=self.query_config,
            query_results=self.query_results,
            metric_results=self.metric_results,
            timestamp=datetime.now(timezone.utc)
        )
        
        nonexistent = result.get_metric_by_name("nonexistent@5")
        assert nonexistent is None

    def test_summary_dict(self):
        """Test generating a summary dictionary."""
        timestamp = datetime.now(timezone.utc)
        
        result = IterationResult(
            experiment_id="exp123",
            iteration_id="iter456",
            query_config=self.query_config,
            query_results=self.query_results,
            metric_results=self.metric_results,
            timestamp=timestamp
        )
        
        summary = result.summary_dict()
        
        assert summary["experiment_id"] == "exp123"
        assert summary["iteration_id"] == "iter456"
        assert summary["timestamp"] == timestamp.isoformat()
        assert summary["query_count"] == 2
        assert len(summary["metrics"]) == 2
        
        # Check metric summary
        assert "ndcg@10" in summary["metrics"]
        assert summary["metrics"]["ndcg@10"] == 0.85

    def test_summary_dict_no_metrics(self):
        """Test summary dict when no metrics exist."""
        result = IterationResult(
            experiment_id="exp123",
            iteration_id="iter456",
            query_config=self.query_config,
            query_results=self.query_results,
            metric_results=[],
            timestamp=datetime.now(timezone.utc)
        )
        
        summary = result.summary_dict()
        
        assert summary["experiment_id"] == "exp123"
        assert summary["query_count"] == 2
        assert len(summary["metrics"]) == 0

    def test_empty_results(self):
        """Test IterationResult with empty results."""
        result = IterationResult(
            experiment_id="exp123",
            iteration_id="iter456",
            query_config=self.query_config,
            query_results={},
            metric_results=[],
            timestamp=datetime.now(timezone.utc)
        )
        
        assert len(result.query_results) == 0
        assert len(result.metric_results) == 0
        assert result.get_primary_metric() is None
        
        summary = result.summary_dict()
        assert summary["query_count"] == 0
        assert len(summary["metrics"]) == 0

    def test_metric_deltas(self):
        """Test iteration result with metric deltas."""
        result = IterationResult(
            experiment_id="exp123",
            iteration_id="iter456",
            query_config=self.query_config,
            query_results=self.query_results,
            metric_results=self.metric_results,
            timestamp=datetime.now(timezone.utc),
            compared_to="iter455",
            metric_deltas={"ndcg@10": 0.05, "precision@5": -0.02}
        )
        
        assert result.compared_to == "iter455"
        assert result.metric_deltas["ndcg@10"] == 0.05
        assert result.metric_deltas["precision@5"] == -0.02
        
        summary = result.summary_dict()
        assert summary["metric_deltas"]["ndcg@10"] == 0.05
        assert summary["metric_deltas"]["precision@5"] == -0.02

    def test_notes(self):
        """Test iteration result with notes."""
        result = IterationResult(
            experiment_id="exp123",
            iteration_id="iter456",
            query_config=self.query_config,
            query_results=self.query_results,
            metric_results=self.metric_results,
            timestamp=datetime.now(timezone.utc),
            notes="This iteration showed significant improvement"
        )
        
        assert result.notes == "This iteration showed significant improvement"
