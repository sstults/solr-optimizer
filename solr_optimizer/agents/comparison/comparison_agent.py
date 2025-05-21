"""
Comparison Agent - Interface for analyzing differences between iterations.

This module defines the ComparisonAgent interface which is responsible for
comparing iterations and explaining changes in rankings.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from solr_optimizer.models.iteration_result import IterationResult


class ComparisonAgent(ABC):
    """
    Agent responsible for comparing iterations and explaining changes in
    rankings.
    """

    @abstractmethod
    def compare_overall_metrics(
        self, iter1: IterationResult, iter2: IterationResult
    ) -> Dict[str, float]:
        """
        Compare the overall metrics between two iterations.

        Args:
            iter1: The first iteration result
            iter2: The second iteration result

        Returns:
            Dictionary of metric name to delta value (iter2 - iter1)
        """
        pass

    @abstractmethod
    def compare_query_results(
        self, iter1: IterationResult, iter2: IterationResult, query: str
    ) -> Dict[str, Any]:
        """
        Compare the results for a specific query between two iterations.

        Args:
            iter1: The first iteration result
            iter2: The second iteration result
            query: The query to compare

        Returns:
            Dictionary with comparison information
        """
        pass

    @abstractmethod
    def explain_ranking_changes(
        self, iter1: IterationResult, iter2: IterationResult, query: str
    ) -> List[Dict[str, Any]]:
        """
        Explain why document rankings changed between iterations for a specific
        query.

        Args:
            iter1: The first iteration result
            iter2: The second iteration result
            query: The query to explain

        Returns:
            List of dictionaries with explanation for each document that
            changed position
        """
        pass

    @abstractmethod
    def find_significant_changes(
        self, iter1: IterationResult, iter2: IterationResult
    ) -> List[Dict[str, Any]]:
        """
        Identify the most significant changes between iterations.

        Args:
            iter1: The first iteration result
            iter2: The second iteration result

        Returns:
            List of dictionaries describing significant changes
        """
        pass

    @abstractmethod
    def generate_summary_report(
        self, iter1: IterationResult, iter2: IterationResult
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report comparing two iterations.

        Args:
            iter1: The first iteration result
            iter2: The second iteration result

        Returns:
            Dictionary with summary report information
        """
        pass

    @abstractmethod
    def analyze_config_changes(
        self, iter1: IterationResult, iter2: IterationResult
    ) -> Dict[str, Any]:
        """
        Analyze what configuration changes were made between iterations.

        Args:
            iter1: The first iteration result
            iter2: The second iteration result

        Returns:
            Dictionary of configuration parameter to change description
        """
        pass
