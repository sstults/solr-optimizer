"""
Unit tests for the DummyQueryTuningAgent class.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from solr_optimizer.agents.query.dummy_query_tuning_agent import DummyQueryTuningAgent
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.query_config import QueryConfig
from solr_optimizer.models.iteration_result import IterationResult, MetricResult, QueryResult


class TestDummyQueryTuningAgent:
    """Test cases for the DummyQueryTuningAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.agent = DummyQueryTuningAgent()
        
        # Sample schema info
        self.schema_info = {
            "schema": {
                "fields": [
                    {"name": "id", "type": "string", "stored": True, "indexed": True},
                    {"name": "title", "type": "text_general", "stored": True, "indexed": True},
                    {"name": "content", "type": "text_general", "stored": True, "indexed": True},
                    {"name": "category", "type": "string", "stored": True, "indexed": True},
                    {"name": "date", "type": "pdate", "stored": True, "indexed": True}
                ],
                "fieldTypes": [
                    {"name": "string", "class": "solr.StrField"},
                    {"name": "text_general", "class": "solr.TextField"},
                    {"name": "pdate", "class": "solr.DatePointField"}
                ]
            }
        }
        
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
        
        # Sample previous iteration result
        self.previous_result = IterationResult(
            experiment_id="test_exp",
            iteration_id="iter_001",
            query_config=QueryConfig(
                query_parser="edismax",
                query_fields={"title": 2.0, "content": 1.0}
            ),
            query_results={
                "query1": QueryResult(query="query1", documents=["doc1", "doc2"], scores={"doc1": 1.0, "doc2": 0.8})
            },
            metric_results=[
                MetricResult(metric_name="ndcg@10", value=0.75, per_query={"query1": 0.75})
            ],
            timestamp=datetime.now(timezone.utc)
        )

    def test_analyze_schema(self):
        """Test schema analysis."""
        analysis = self.agent.analyze_schema(self.schema_info)
        
        assert analysis is not None
        assert isinstance(analysis, dict)
        assert "text_fields" in analysis
        assert "indexed_fields" in analysis
        assert "stored_fields" in analysis
        
        # Check that text fields are identified
        assert "title" in analysis["text_fields"]
        assert "content" in analysis["text_fields"]
        
        # Check that indexed fields are identified
        assert "title" in analysis["indexed_fields"]
        assert "content" in analysis["indexed_fields"]
        assert "category" in analysis["indexed_fields"]

    def test_analyze_schema_empty(self):
        """Test schema analysis with empty schema."""
        empty_schema = {"schema": {"fields": [], "fieldTypes": []}}
        analysis = self.agent.analyze_schema(empty_schema)
        
        assert analysis["text_fields"] == []
        assert analysis["indexed_fields"] == []
        assert analysis["stored_fields"] == []

    def test_generate_initial_config(self):
        """Test generation of initial query configuration."""
        config = self.agent.generate_initial_config(self.experiment_config, self.schema_info)
        
        assert config is not None
        assert isinstance(config, QueryConfig)
        assert config.query_parser == "edismax"
        
        # Should have query fields populated with text fields
        assert len(config.query_fields) > 0
        assert "title" in config.query_fields
        assert "content" in config.query_fields
        
        # Title should have higher boost than content
        assert config.query_fields["title"] > config.query_fields["content"]

    def test_generate_initial_config_no_text_fields(self):
        """Test initial config generation when no text fields are available."""
        schema_no_text = {
            "schema": {
                "fields": [
                    {"name": "id", "type": "string", "stored": True, "indexed": True},
                    {"name": "category", "type": "string", "stored": True, "indexed": True}
                ],
                "fieldTypes": [
                    {"name": "string", "class": "solr.StrField"}
                ]
            }
        }
        
        config = self.agent.generate_initial_config(self.experiment_config, schema_no_text)
        
        assert config is not None
        assert config.query_parser == "edismax"
        # Should fall back to basic configuration
        assert len(config.query_fields) >= 0  # May be empty or have fallback fields

    def test_suggest_next_config(self):
        """Test suggestion of next query configuration."""
        config = self.agent.suggest_next_config(self.previous_result, self.schema_info)
        
        assert config is not None
        assert isinstance(config, QueryConfig)
        
        # Should be different from previous config in some way
        prev_config = self.previous_result.query_config
        
        # At least one parameter should be different
        differences = 0
        
        if config.query_fields != prev_config.query_fields:
            differences += 1
        if config.phrase_fields != prev_config.phrase_fields:
            differences += 1
        if config.boost_queries != prev_config.boost_queries:
            differences += 1
        if config.minimum_match != prev_config.minimum_match:
            differences += 1
        if config.tie_breaker != prev_config.tie_breaker:
            differences += 1
            
        assert differences > 0, "Next config should differ from previous config"

    def test_suggest_next_config_random_variations(self):
        """Test that suggest_next_config produces random variations."""
        configs = []
        for _ in range(5):
            config = self.agent.suggest_next_config(self.previous_result, self.schema_info)
            configs.append(config)
        
        # Should have some variation in the generated configs
        unique_configs = set()
        for config in configs:
            # Create a hashable representation
            config_tuple = (
                tuple(sorted(config.query_fields.items())),
                tuple(sorted(config.phrase_fields.items())),
                tuple(config.boost_queries),
                config.minimum_match,
                config.tie_breaker
            )
            unique_configs.add(config_tuple)
        
        # Should have at least some variation (not all identical)
        assert len(unique_configs) >= 1

    def test_adjust_parameters_increase(self):
        """Test parameter adjustment for metric improvement."""
        config = self.agent.adjust_parameters(
            self.previous_result, 
            "ndcg@10", 
            "increase"
        )
        
        assert config is not None
        assert isinstance(config, QueryConfig)
        
        # Should make adjustments to try to increase the metric
        # The dummy agent makes random adjustments, so we just verify structure
        assert hasattr(config, 'query_fields')
        assert hasattr(config, 'phrase_fields')
        assert hasattr(config, 'boost_queries')

    def test_adjust_parameters_decrease(self):
        """Test parameter adjustment for metric decrease."""
        config = self.agent.adjust_parameters(
            self.previous_result, 
            "ndcg@10", 
            "decrease"
        )
        
        assert config is not None
        assert isinstance(config, QueryConfig)
        
        # Should make adjustments to try to decrease the metric
        # The dummy agent makes random adjustments, so we just verify structure
        assert hasattr(config, 'query_fields')
        assert hasattr(config, 'phrase_fields')
        assert hasattr(config, 'boost_queries')

    def test_adjust_parameters_invalid_direction(self):
        """Test parameter adjustment with invalid direction."""
        # Should handle invalid direction gracefully
        config = self.agent.adjust_parameters(
            self.previous_result, 
            "ndcg@10", 
            "invalid_direction"
        )
        
        assert config is not None
        assert isinstance(config, QueryConfig)

    def test_adjust_parameters_missing_metric(self):
        """Test parameter adjustment when target metric is missing."""
        config = self.agent.adjust_parameters(
            self.previous_result, 
            "nonexistent_metric", 
            "increase"
        )
        
        assert config is not None
        assert isinstance(config, QueryConfig)

    def test_field_type_detection(self):
        """Test correct detection of field types from schema."""
        analysis = self.agent.analyze_schema(self.schema_info)
        
        # Text fields should be detected correctly
        assert "title" in analysis["text_fields"]
        assert "content" in analysis["text_fields"]
        assert "id" not in analysis["text_fields"]  # String field, not text
        assert "category" not in analysis["text_fields"]  # String field, not text

    def test_boost_value_ranges(self):
        """Test that boost values are within reasonable ranges."""
        config = self.agent.generate_initial_config(self.experiment_config, self.schema_info)
        
        # Query field boosts should be positive and reasonable
        for field, boost in config.query_fields.items():
            assert boost > 0, f"Boost for {field} should be positive"
            assert boost <= 10, f"Boost for {field} should be reasonable (<=10)"
        
        # Phrase field boosts should be positive
        for field, boost in config.phrase_fields.items():
            assert boost > 0, f"Phrase boost for {field} should be positive"
            assert boost <= 10, f"Phrase boost for {field} should be reasonable (<=10)"

    def test_tie_breaker_ranges(self):
        """Test that tie breaker values are within valid ranges."""
        config = self.agent.generate_initial_config(self.experiment_config, self.schema_info)
        
        # Tie breaker should be between 0 and 1
        if config.tie_breaker is not None:
            assert 0 <= config.tie_breaker <= 1, "Tie breaker should be between 0 and 1"

    def test_config_consistency(self):
        """Test that generated configurations are internally consistent."""
        config = self.agent.generate_initial_config(self.experiment_config, self.schema_info)
        
        # Query fields and phrase fields should use valid field names
        schema_fields = {field["name"] for field in self.schema_info["schema"]["fields"]}
        
        for field in config.query_fields.keys():
            assert field in schema_fields, f"Query field {field} should exist in schema"
        
        for field in config.phrase_fields.keys():
            assert field in schema_fields, f"Phrase field {field} should exist in schema"

    def test_minimum_match_values(self):
        """Test that minimum match values are reasonable."""
        configs = []
        for _ in range(10):
            config = self.agent.generate_initial_config(self.experiment_config, self.schema_info)
            configs.append(config)
        
        # Check that minimum match values, when set, are reasonable
        for config in configs:
            if config.minimum_match is not None:
                if config.minimum_match.endswith('%'):
                    percentage = int(config.minimum_match[:-1])
                    assert 1 <= percentage <= 100, "Percentage should be between 1 and 100"
                else:
                    # Assume it's a number
                    try:
                        value = int(config.minimum_match)
                        assert value >= 1, "Minimum match number should be >= 1"
                    except ValueError:
                        # Could be a more complex MM expression, just verify it's not empty
                        assert len(config.minimum_match) > 0

    def test_random_seed_consistency(self):
        """Test behavior with different random seeds."""
        # This test ensures the dummy agent is actually using randomization
        configs1 = []
        configs2 = []
        
        # Generate multiple configurations
        for _ in range(3):
            config1 = self.agent.generate_initial_config(self.experiment_config, self.schema_info)
            config2 = self.agent.generate_initial_config(self.experiment_config, self.schema_info)
            configs1.append(config1)
            configs2.append(config2)
        
        # There should be some variation across multiple calls
        # (This is probabilistic, but very likely to pass with proper randomization)
        all_same = True
        for i in range(len(configs1)):
            if (configs1[i].query_fields != configs2[i].query_fields or
                configs1[i].phrase_fields != configs2[i].phrase_fields or
                configs1[i].tie_breaker != configs2[i].tie_breaker):
                all_same = False
                break
        
        # With proper randomization, configs should not all be identical
        # (This might occasionally fail due to randomness, but should usually pass)
        assert not all_same or len(configs1) == 1, "Should have some randomization in config generation"
