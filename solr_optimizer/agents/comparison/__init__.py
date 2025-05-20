"""
Comparison Agent Module - Agents for analyzing differences between iterations.

This package contains implementations of the ComparisonAgent interface which is responsible for
comparing iterations and explaining changes in rankings.
"""

from solr_optimizer.agents.comparison.comparison_agent import ComparisonAgent
from solr_optimizer.agents.comparison.standard_comparison_agent import StandardComparisonAgent

__all__ = [
    "ComparisonAgent",
    "StandardComparisonAgent"
]
