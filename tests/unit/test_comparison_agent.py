from solr_optimizer.agents.comparison.comparison_agent import ComparisonAgent


class MockComparisonAgent(ComparisonAgent):
    """Concrete implementation of ComparisonAgent for testing"""

    def analyze_config_changes(self, config1, config2):
        return {"change": "config"}

    def compare_overall_metrics(self, metrics1, metrics2):
        return {"metric_diff": 0.5}

    def compare_query_results(self, results1, results2):
        return {"query_diff": "some_diff"}

    def explain_ranking_changes(self, rankings1, rankings2):
        return "Ranking explanation"

    def find_significant_changes(self, data1, data2):
        return {"significant": "changes"}

    def generate_summary_report(self, comparison_data):
        return "Summary report"

    def compare(self, method1, method2):
        return {"method_comparison": f"{method1} vs {method2}", "results": "mocked"}


def test_comparison_agent_initialization():
    """Test basic initialization of ComparisonAgent"""
    agent = MockComparisonAgent()
    assert agent is not None, "Agent should initialize successfully"


def test_compare_methods():
    """Test comparison method functionality"""
    agent = MockComparisonAgent()
    result = agent.compare("method1", "method2")
    assert result is not None, "Comparison should return results"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "method_comparison" in result, "Result should contain comparison details"
