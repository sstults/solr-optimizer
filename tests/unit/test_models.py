"""
Unit tests for the model classes in the Solr Optimizer framework.
"""

import pytest

from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.query_config import QueryConfig


class TestExperimentConfig:
    """Test cases for the ExperimentConfig class."""

    def test_valid_config_creation(self):
        """Test that a valid configuration can be created."""
        config = ExperimentConfig(
            corpus="test_collection",
            queries=["query1", "query2"],
            judgments={
                "query1": {"doc1": 3, "doc2": 1, "doc3": 0},
                "query2": {"doc4": 2, "doc5": 3, "doc6": 1},
            },
            primary_metric="ndcg",
            metric_depth=10,
            description="Test experiment",
        )

        assert config.corpus == "test_collection"
        assert len(config.queries) == 2
        assert config.primary_metric == "ndcg"
        assert config.metric_depth == 10
        assert config.description == "Test experiment"

    def test_missing_judgments_validation(self):
        """Test validation for missing judgments."""
        with pytest.raises(ValueError, match="Missing judgments"):
            ExperimentConfig(
                corpus="test_collection",
                queries=["query1", "query2", "query3"],
                judgments={
                    "query1": {"doc1": 3, "doc2": 1},
                    "query2": {"doc4": 2, "doc5": 3},
                },
                primary_metric="ndcg",
                metric_depth=10,
            )

    def test_invalid_metric_validation(self):
        """Test validation for invalid metrics."""
        with pytest.raises(ValueError, match="Invalid primary_metric"):
            ExperimentConfig(
                corpus="test_collection",
                queries=["query1"],
                judgments={"query1": {"doc1": 3}},
                primary_metric="invalid_metric",
                metric_depth=10,
            )

    def test_invalid_depth_validation(self):
        """Test validation for invalid metric depth."""
        with pytest.raises(ValueError, match="metric_depth must be positive"):
            ExperimentConfig(
                corpus="test_collection",
                queries=["query1"],
                judgments={"query1": {"doc1": 3}},
                primary_metric="ndcg",
                metric_depth=0,
            )


class TestQueryConfig:
    """Test cases for the QueryConfig class."""

    def test_default_values(self):
        """Test default values for QueryConfig."""
        config = QueryConfig()

        assert config.query_parser == "edismax"
        assert config.query_fields == {}
        assert config.phrase_fields == {}
        assert config.boost_queries == []
        assert config.boost_functions == []
        assert config.minimum_match is None
        assert config.tie_breaker == 0.0
        assert config.iteration_id is None

    def test_to_solr_params(self):
        """Test conversion to Solr parameters."""
        config = QueryConfig(
            query_parser="edismax",
            query_fields={"title": 2.0, "content": 1.0},
            phrase_fields={"title": 3.0},
            boost_queries=["category:electronics^1.5"],
            boost_functions=["recip(rord(date),1,1000,1000)"],
            minimum_match="75%",
            tie_breaker=0.3,
            additional_params={"facet": "true", "facet.field": "category"},
        )

        params = config.to_solr_params()

        assert params["defType"] == "edismax"
        assert params["qf"] == "title^2.0 content^1.0"
        assert params["pf"] == "title^3.0"
        assert params["bq"] == ["category:electronics^1.5"]
        assert params["bf"] == ["recip(rord(date),1,1000,1000)"]
        assert params["mm"] == "75%"
        assert params["tie"] == 0.3
        assert params["facet"] == "true"
        assert params["facet.field"] == "category"
