"""
Unit tests for the StandardMetricsAgent implementation.
"""

import pytest

from solr_optimizer.agents.metrics.standard_metrics_agent import StandardMetricsAgent
from solr_optimizer.models.iteration_result import MetricResult


class TestStandardMetricsAgent:
    """Test cases for the StandardMetricsAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.agent = StandardMetricsAgent()

        # Sample judgments for a query (scale 0-3)
        self.judgments = {
            "doc1": 3,  # Highly relevant
            "doc2": 2,  # Relevant
            "doc3": 1,  # Somewhat relevant
            "doc4": 0,  # Not relevant
            "doc5": 3,  # Highly relevant
            "doc6": 2,  # Relevant
            "doc7": 0,  # Not relevant
        }

        # Ideal ranking (for reference)
        self.ideal_ranking = ["doc1", "doc5", "doc2", "doc6", "doc3"]

        # Actual result rankings to test
        self.perfect_ranking = ["doc1", "doc5", "doc2", "doc6", "doc3"]
        self.good_ranking = ["doc1", "doc2", "doc5", "doc3", "doc6", "doc4"]
        self.poor_ranking = ["doc4", "doc7", "doc3", "doc2", "doc1"]

    def test_supported_metrics(self):
        """Test that the agent reports its supported metrics."""
        supported = self.agent.get_supported_metrics()
        assert "ndcg" in supported
        assert "precision" in supported
        assert "recall" in supported
        assert "mrr" in supported
        assert "dcg" in supported
        assert "err" in supported

    def test_ndcg_perfect_ranking(self):
        """Test NDCG calculation with perfect ranking."""
        ndcg = self.agent.calculate_metric(
            "ndcg", self.perfect_ranking, self.judgments, 5
        )
        # Perfect ranking should give NDCG of 1.0
        assert ndcg == 1.0

    def test_ndcg_good_ranking(self):
        """Test NDCG calculation with good but imperfect ranking."""
        ndcg = self.agent.calculate_metric("ndcg", self.good_ranking, self.judgments, 5)
        # Good ranking should give high but not perfect NDCG
        assert 0.8 < ndcg < 1.0

    def test_ndcg_poor_ranking(self):
        """Test NDCG calculation with poor ranking."""
        ndcg = self.agent.calculate_metric("ndcg", self.poor_ranking, self.judgments, 5)
        # Poor ranking should give low NDCG
        assert ndcg < 0.8

    def test_precision(self):
        """Test precision calculation."""
        # Perfect ranking has all 5 documents relevant
        precision_perfect = self.agent.calculate_metric(
            "precision", self.perfect_ranking, self.judgments, 5
        )
        assert precision_perfect == 1.0

        # Good ranking has 5 relevant documents out of 6
        precision_good = self.agent.calculate_metric(
            "precision", self.good_ranking, self.judgments, 6
        )
        assert precision_good == 5 / 6

        # Poor ranking has 3 relevant documents out of 5
        precision_poor = self.agent.calculate_metric(
            "precision", self.poor_ranking, self.judgments, 5
        )
        assert precision_poor == 3 / 5

    def test_recall(self):
        """Test recall calculation."""
        # Perfect ranking has all relevant documents in top 5
        recall_perfect = self.agent.calculate_metric(
            "recall", self.perfect_ranking, self.judgments, 5
        )
        assert recall_perfect == 1.0

        # Good ranking includes only 5 out of 5 relevant documents in top 6
        recall_good = self.agent.calculate_metric(
            "recall", self.good_ranking, self.judgments, 6
        )
        assert recall_good == 5 / 5

        # Poor ranking includes 3 out of 5 relevant documents
        recall_poor = self.agent.calculate_metric(
            "recall", self.poor_ranking, self.judgments, 5
        )
        assert recall_poor == 3 / 5

    def test_mrr(self):
        """Test MRR calculation."""
        # Perfect ranking has first relevant document at position 1
        mrr_perfect = self.agent.calculate_metric(
            "mrr", self.perfect_ranking, self.judgments, 5
        )
        assert mrr_perfect == 1.0

        # Good ranking has first relevant document at position 1
        mrr_good = self.agent.calculate_metric(
            "mrr", self.good_ranking, self.judgments, 6
        )
        assert mrr_good == 1.0

        # Poor ranking has first relevant document at position 3
        mrr_poor = self.agent.calculate_metric(
            "mrr", self.poor_ranking, self.judgments, 5
        )
        assert mrr_poor == 1 / 3

    def test_invalid_metric(self):
        """Test handling of invalid metrics."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            self.agent.calculate_metric(
                "invalid_metric", self.perfect_ranking, self.judgments, 5
            )

    def test_empty_results(self):
        """Test handling of empty result lists."""
        empty_results = []
        ndcg = self.agent.calculate_metric("ndcg", empty_results, self.judgments, 5)
        assert ndcg == 0.0

        precision = self.agent.calculate_metric(
            "precision", empty_results, self.judgments, 5
        )
        assert precision == 0.0

    def test_calculate_multiple_metrics(self):
        """Test calculation of multiple metrics at once."""
        results_by_query = {"query1": self.perfect_ranking, "query2": self.good_ranking}

        judgments_by_query = {"query1": self.judgments, "query2": self.judgments}

        # Calculate multiple metrics
        metric_results = self.agent.calculate_metrics(
            ["ndcg", "precision"], results_by_query, judgments_by_query, 5
        )

        # Check results
        assert len(metric_results) == 2
        assert isinstance(metric_results[0], MetricResult)
        assert metric_results[0].metric_name in ["ndcg@5", "precision@5"]

        # Check per-query values
        for result in metric_results:
            assert "query1" in result.per_query
            assert "query2" in result.per_query

            # Mean value should be average of query values
            query1_val = result.per_query["query1"]
            query2_val = result.per_query["query2"]
            expected_avg = (query1_val + query2_val) / 2
            assert result.value == expected_avg
