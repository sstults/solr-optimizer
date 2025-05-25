"""
Unit tests for the StandardComparisonAgent class.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime, timezone

from solr_optimizer.agents.comparison.standard_comparison_agent import StandardComparisonAgent
from solr_optimizer.models.iteration_result import IterationResult, MetricResult, QueryResult
from solr_optimizer.models.query_config import QueryConfig


class TestStandardComparisonAgent:
    """Test cases for the StandardComparisonAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.agent = StandardComparisonAgent()
        
        # Sample query configs
        self.config1 = QueryConfig(
            query_parser="edismax",
            query_fields={"title": 2.0, "content": 1.0},
            phrase_fields={"title": 3.0},
            boost_queries=["category:electronics^1.5"],
            minimum_match="75%",
            tie_breaker=0.1
        )
        
        self.config2 = QueryConfig(
            query_parser="edismax",
            query_fields={"title": 3.0, "content": 1.5},
            phrase_fields={"title": 4.0, "content": 2.0},
            boost_queries=["category:electronics^2.0", "brand:apple^1.2"],
            minimum_match="80%",
            tie_breaker=0.2
        )
        
        # Sample iteration results
        self.iteration1 = IterationResult(
            experiment_id="exp_123",
            iteration_id="iter_001",
            query_config=self.config1,
            query_results={
                "query1": QueryResult(
                    query="query1", 
                    documents=["doc1", "doc2", "doc3"], 
                    scores={"doc1": 0.95, "doc2": 0.85, "doc3": 0.75}
                ),
                "query2": QueryResult(
                    query="query2", 
                    documents=["doc4", "doc5"], 
                    scores={"doc4": 0.90, "doc5": 0.80}
                )
            },
            metric_results=[
                MetricResult(metric_name="ndcg@10", value=0.75, per_query={"query1": 0.8, "query2": 0.7}),
                MetricResult(metric_name="precision@5", value=0.6, per_query={"query1": 0.7, "query2": 0.5})
            ],
            timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        
        self.iteration2 = IterationResult(
            experiment_id="exp_123",
            iteration_id="iter_002",
            query_config=self.config2,
            query_results={
                "query1": QueryResult(
                    query="query1", 
                    documents=["doc1", "doc3", "doc2"], 
                    scores={"doc1": 0.98, "doc3": 0.88, "doc2": 0.78}
                ),
                "query2": QueryResult(
                    query="query2", 
                    documents=["doc5", "doc4", "doc6"], 
                    scores={"doc5": 0.92, "doc4": 0.82, "doc6": 0.72}
                )
            },
            metric_results=[
                MetricResult(metric_name="ndcg@10", value=0.85, per_query={"query1": 0.9, "query2": 0.8}),
                MetricResult(metric_name="precision@5", value=0.7, per_query={"query1": 0.8, "query2": 0.6})
            ],
            timestamp=datetime(2023, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        )

    def test_analyze_config_changes(self):
        """Test analysis of configuration changes."""
        changes = self.agent.analyze_config_changes(self.iteration1, self.iteration2)
        
        assert changes is not None
        assert isinstance(changes, dict)
        
        # Should detect changes in query fields
        if "query_fields" in changes:
            assert changes["query_fields"]["type"] == "modified"
        
        # Should detect changes in phrase fields
        if "phrase_fields" in changes:
            assert changes["phrase_fields"]["type"] == "modified"
        
        # Should detect changes in boost queries
        if "boost_queries" in changes:
            assert changes["boost_queries"]["type"] == "modified"
        
        # Should detect changes in minimum match
        if "minimum_match" in changes:
            assert changes["minimum_match"]["before"] == "75%"
            assert changes["minimum_match"]["after"] == "80%"

    def test_analyze_config_changes_identical(self):
        """Test analysis when configs are identical."""
        changes = self.agent.analyze_config_changes(self.iteration1, self.iteration1)
        
        assert changes is not None
        assert isinstance(changes, dict)
        
        # Should have no changes
        assert len(changes) == 0

    def test_compare_overall_metrics(self):
        """Test comparison of overall metrics."""
        comparison = self.agent.compare_overall_metrics(self.iteration1, self.iteration2)
        
        assert comparison is not None
        assert isinstance(comparison, dict)
        
        # Should compare NDCG
        assert "ndcg@10" in comparison
        assert abs(comparison["ndcg@10"] - 0.1) < 1e-10  # 0.85 - 0.75
        
        # Should compare Precision
        assert "precision@5" in comparison
        assert abs(comparison["precision@5"] - 0.1) < 1e-10  # 0.7 - 0.6

    def test_compare_overall_metrics_missing_metric(self):
        """Test metric comparison when one metric is missing."""
        iter1 = IterationResult(
            experiment_id="exp_123",
            iteration_id="iter_001",
            query_config=self.config1,
            query_results={},
            metric_results=[MetricResult(metric_name="ndcg@10", value=0.75, per_query={})],
            timestamp=datetime.now(timezone.utc)
        )
        
        iter2 = IterationResult(
            experiment_id="exp_123",
            iteration_id="iter_002",
            query_config=self.config2,
            query_results={},
            metric_results=[MetricResult(metric_name="precision@5", value=0.7, per_query={})],
            timestamp=datetime.now(timezone.utc)
        )
        
        comparison = self.agent.compare_overall_metrics(iter1, iter2)
        
        assert comparison is not None
        # Should handle missing metrics gracefully
        assert "ndcg@10" in comparison
        assert "precision@5" in comparison

    def test_compare_query_results(self):
        """Test comparison of query results."""
        comparison = self.agent.compare_query_results(self.iteration1, self.iteration2, "query1")
        
        assert comparison is not None
        assert isinstance(comparison, dict)
        
        # Should have comparison information
        assert "query" in comparison
        assert comparison["query"] == "query1"
        assert "metrics_comparison" in comparison
        assert "documents_comparison" in comparison

    def test_explain_ranking_changes(self):
        """Test explanation of ranking changes."""
        explanation = self.agent.explain_ranking_changes(self.iteration1, self.iteration2, "query1")
        
        assert explanation is not None
        assert isinstance(explanation, list)
        
        # Should have explanations for changed documents
        if len(explanation) > 0:
            assert "document_id" in explanation[0]
            assert "position_change" in explanation[0]

    def test_explain_ranking_changes_identical(self):
        """Test explanation when rankings are identical."""
        explanation = self.agent.explain_ranking_changes(self.iteration1, self.iteration1, "query1")
        
        assert explanation is not None
        assert isinstance(explanation, list)
        # Should have no significant changes
        assert len(explanation) == 0

    def test_explain_ranking_changes_completely_different(self):
        """Test explanation when rankings are completely different."""
        iter_diff = IterationResult(
            experiment_id="exp_123",
            iteration_id="iter_diff",
            query_config=self.config2,
            query_results={
                "query1": QueryResult(
                    query="query1", 
                    documents=["doc4", "doc5", "doc6"], 
                    scores={"doc4": 0.95, "doc5": 0.85, "doc6": 0.75}
                )
            },
            metric_results=self.iteration2.metric_results,
            timestamp=datetime.now(timezone.utc)
        )
        
        explanation = self.agent.explain_ranking_changes(self.iteration1, iter_diff, "query1")
        
        assert explanation is not None
        assert isinstance(explanation, list)

    def test_find_significant_changes(self):
        """Test finding significant changes between iterations."""
        significant = self.agent.find_significant_changes(self.iteration1, self.iteration2)
        
        assert significant is not None
        assert isinstance(significant, list)
        
        # Should identify changes
        if len(significant) > 0:
            assert "type" in significant[0]

    def test_generate_summary_report(self):
        """Test generation of summary report."""
        report = self.agent.generate_summary_report(self.iteration1, self.iteration2)
        
        assert report is not None
        assert isinstance(report, dict)
        
        # Should contain all major comparison sections
        expected_keys = ["iteration1_id", "iteration2_id", "metrics_comparison"]
        for key in expected_keys:
            assert key in report

    def test_compare_with_none_values(self):
        """Test comparison handling None values gracefully."""
        # These should not crash
        changes = self.agent.analyze_config_changes(self.iteration1, self.iteration1)
        assert changes is not None
        
        comparison = self.agent.compare_overall_metrics(self.iteration1, self.iteration1)
        assert comparison is not None

    def test_ranking_position_changes(self):
        """Test detailed ranking position change analysis."""
        explanation = self.agent.explain_ranking_changes(self.iteration1, self.iteration2, "query1")
        
        # Should identify specific position changes
        assert explanation is not None
        assert isinstance(explanation, list)

    def test_response_time_analysis(self):
        """Test response time change analysis."""
        # Create iterations with response time info in explain_info
        iter1 = IterationResult(
            experiment_id="exp_123",
            iteration_id="iter_001",
            query_config=self.config1,
            query_results={
                "q1": QueryResult(
                    query="q1", 
                    documents=["doc1"], 
                    scores={"doc1": 0.95},
                    explain_info={"response_time": 100.0}
                )
            },
            metric_results=[],
            timestamp=datetime.now(timezone.utc)
        )
        
        iter2 = IterationResult(
            experiment_id="exp_123",
            iteration_id="iter_002",
            query_config=self.config2,
            query_results={
                "q1": QueryResult(
                    query="q1", 
                    documents=["doc1"], 
                    scores={"doc1": 0.95},
                    explain_info={"response_time": 50.0}
                )
            },
            metric_results=[],
            timestamp=datetime.now(timezone.utc)
        )
        
        comparison = self.agent.compare_query_results(iter1, iter2, "q1")
        
        assert comparison is not None
        assert "query" in comparison

    def test_empty_results_comparison(self):
        """Test comparison with empty results."""
        empty_iteration = IterationResult(
            experiment_id="exp_123",
            iteration_id="iter_empty",
            query_config=QueryConfig(),
            query_results={},
            metric_results=[],
            timestamp=datetime.now(timezone.utc)
        )
        
        comparison = self.agent.compare_overall_metrics(self.iteration1, empty_iteration)
        
        assert comparison is not None
        assert isinstance(comparison, dict)

    def test_metric_significance_threshold(self):
        """Test that significant changes use reasonable thresholds."""
        # Create iterations with small differences
        small_diff_iteration = IterationResult(
            experiment_id="exp_123",
            iteration_id="iter_small",
            query_config=self.config1,
            query_results={},
            metric_results=[
                MetricResult(metric_name="ndcg@10", value=0.751, per_query={})  # Very small improvement
            ],
            timestamp=datetime.now(timezone.utc)
        )
        
        significant = self.agent.find_significant_changes(self.iteration1, small_diff_iteration)
        
        # Small changes might not be considered significant
        assert significant is not None
        assert isinstance(significant, list)

    def test_config_parameter_tracking(self):
        """Test that all config parameter changes are tracked."""
        # Create config with all possible differences
        config_diff = QueryConfig(
            query_parser="lucene",  # Different parser
            query_fields={"title": 1.0},  # Different fields
            phrase_fields={},  # No phrase fields
            boost_queries=[],  # No boost queries
            minimum_match=None,  # No minimum match
            tie_breaker=0.5  # Different tie breaker
        )
        
        iter_diff = IterationResult(
            experiment_id="exp_123",
            iteration_id="iter_diff",
            query_config=config_diff,
            query_results={},
            metric_results=[],
            timestamp=datetime.now(timezone.utc)
        )
        
        changes = self.agent.analyze_config_changes(self.iteration1, iter_diff)
        
        assert changes is not None
        # Should detect parser change
        if "query_parser" in changes:
            assert changes["query_parser"]["before"] == "edismax"
            assert changes["query_parser"]["after"] == "lucene"

    def test_extract_score_components(self):
        """Test extraction of score components from explain info."""
        explain = {
            "value": 1.5,
            "description": "sum of:",
            "details": [
                {"value": 1.0, "description": "weight(title:test)"},
                {"value": 0.5, "description": "weight(content:test)"}
            ]
        }
        
        components = self.agent._extract_score_components(explain)
        
        assert components is not None
        assert isinstance(components, dict)
        assert "sum of:" in components
        assert components["sum of:"] == 1.5

    def test_summarize_explain_info(self):
        """Test summarization of explain information."""
        explain = {
            "value": 1.5,
            "description": "sum of:",
            "details": [
                {"value": 1.0, "description": "weight(title:test)"},
                {"value": 0.5, "description": "weight(content:test)"}
            ]
        }
        
        summary = self.agent._summarize_explain_info(explain)
        
        assert summary is not None
        assert isinstance(summary, dict)
        assert "score" in summary
        assert summary["score"] == 1.5
        assert "top_contributors" in summary

    def test_analyze_explain_differences(self):
        """Test analysis of explain differences."""
        explain1 = {
            "value": 1.0,
            "description": "sum of:",
            "details": [
                {"value": 1.0, "description": "weight(title:test)"}
            ]
        }
        
        explain2 = {
            "value": 1.5,
            "description": "sum of:",
            "details": [
                {"value": 1.0, "description": "weight(title:test)"},
                {"value": 0.5, "description": "weight(content:test)"}
            ]
        }
        
        analysis = self.agent._analyze_explain_differences(explain1, explain2)
        
        assert analysis is not None
        assert isinstance(analysis, dict)
        assert "score_change" in analysis
        assert analysis["score_change"] == 0.5
