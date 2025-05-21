"""
Standard Comparison Agent - Implementation of the ComparisonAgent interface.

This module provides a concrete implementation of the ComparisonAgent interface
that analyzes differences between iterations and explains ranking changes.
"""

import logging
from typing import Any, Dict, List, TypedDict, cast

from solr_optimizer.agents.comparison.comparison_agent import ComparisonAgent
from solr_optimizer.models.iteration_result import IterationResult

logger = logging.getLogger(__name__)


class StandardComparisonAgent(ComparisonAgent):
    """
    Standard implementation of the ComparisonAgent interface.

    This agent provides detailed analysis of differences between experiment
    iterations including metric comparisons, ranking changes, and configuration
    differences.
    """

    def __init__(
        self,
        significant_metric_threshold: float = 0.05,
        significant_rank_change: int = 3,
        analyze_top_n: int = 10,
    ):
        """
        Initialize the StandardComparisonAgent.

        Args:
            significant_metric_threshold: Threshold for considering a metric
                change significant
            significant_rank_change: Number of rank positions change to be
                considered significant
            analyze_top_n: Number of top documents to analyze in detail
        """
        self.significant_metric_threshold = significant_metric_threshold
        self.significant_rank_change = significant_rank_change
        self.analyze_top_n = analyze_top_n

    def _get_metric_values_by_name(self, metric_results: List[Any]) -> Dict[str, float]:
        """Convert a list of MetricResult objects to a dict of metric_name -> value"""
        if isinstance(metric_results, list):
            return {metric.metric_name: metric.value for metric in metric_results if hasattr(metric, "metric_name")}
        elif isinstance(metric_results, dict):
            # Legacy support for when metric_results was a dict
            return dict(metric_results.get("overall", {}))
        return {}

    def _get_metric_values_by_query(self, metric_results: List[Any], query: str) -> Dict[str, float]:
        """Get metric values for a specific query from a list of MetricResult objects"""
        result = {}
        if isinstance(metric_results, list):
            for metric in metric_results:
                if hasattr(metric, "per_query") and isinstance(metric.per_query, dict):
                    if query in metric.per_query:
                        result[metric.metric_name] = metric.per_query[query]
        elif isinstance(metric_results, dict):
            per_query = metric_results.get("per_query", {})
            if isinstance(per_query, dict) and query in per_query:
                return dict(per_query[query])
        return result

    def compare_overall_metrics(self, iter1: IterationResult, iter2: IterationResult) -> Dict[str, float]:
        """
        Compare the overall metrics between two iterations.

        Args:
            iter1: The first iteration result
            iter2: The second iteration result

        Returns:
            Dictionary of metric name to delta value (iter2 - iter1)
        """
        deltas = {}

        # Extract overall metrics from both iterations
        metrics1 = self._get_metric_values_by_name(iter1.metric_results)
        metrics2 = self._get_metric_values_by_name(iter2.metric_results)

        # Calculate deltas for all metrics present in both iterations
        for metric_name in set(metrics1.keys()).union(metrics2.keys()):
            value1 = metrics1.get(metric_name, 0.0)
            value2 = metrics2.get(metric_name, 0.0)
            deltas[metric_name] = value2 - value1

        return deltas

    def _get_query_result(self, iteration: IterationResult, query: str) -> Dict[str, Any]:
        """Get a dictionary representation of a QueryResult object"""
        if query not in iteration.query_results:
            return {"documents": [], "scores": {}, "explain_info": {}}

        query_result = iteration.query_results[query]

        # If query_result is already a dict, return it
        if isinstance(query_result, dict):
            return dict(query_result)

        # If it's a QueryResult object, extract its attributes
        return {
            "documents": list(getattr(query_result, "documents", [])),
            "scores": dict(getattr(query_result, "scores", {})),
            "explain_info": dict(getattr(query_result, "explain_info", {})),
        }

    def compare_query_results(self, iter1: IterationResult, iter2: IterationResult, query: str) -> Dict[str, Any]:
        """
        Compare the results for a specific query between two iterations.

        Args:
            iter1: The first iteration result
            iter2: The second iteration result
            query: The query to compare

        Returns:
            Dictionary with comparison information
        """
        # Initialize with proper typing for mypy
        result: Dict[str, Any] = {
            "query": query,
            "metrics_comparison": {},
            "documents_comparison": {},
            "new_documents": [],
            "removed_documents": [],
        }

        # Compare metrics for this query
        metrics1 = self._get_metric_values_by_query(iter1.metric_results, query)
        metrics2 = self._get_metric_values_by_query(iter2.metric_results, query)

        for metric_name in set(metrics1.keys()).union(metrics2.keys()):
            value1 = metrics1.get(metric_name, 0.0)
            value2 = metrics2.get(metric_name, 0.0)
            result["metrics_comparison"][metric_name] = {
                "before": value1,
                "after": value2,
                "delta": value2 - value1,
            }

        # Extract document lists
        query_result1 = self._get_query_result(iter1, query)
        query_result2 = self._get_query_result(iter2, query)

        docs1 = query_result1.get("documents", [])
        docs2 = query_result2.get("documents", [])

        # Create position maps for both iterations
        pos_map1 = {doc_id: pos for pos, doc_id in enumerate(docs1)}
        pos_map2 = {doc_id: pos for pos, doc_id in enumerate(docs2)}

        # Find documents in both result sets
        common_docs = set(pos_map1.keys()).intersection(set(pos_map2.keys()))

        # Analyze position changes
        for doc_id in common_docs:
            pos1 = pos_map1[doc_id]
            pos2 = pos_map2[doc_id]
            result["documents_comparison"][doc_id] = {
                "position_before": pos1,
                "position_after": pos2,
                "position_change": pos1 - pos2,  # Positive means improved rank
            }

        # Find new documents
        # Use a temporary variable with proper typing to avoid mypy errors
        new_docs: List[Dict[str, Any]] = [
            {"document_id": doc_id, "position": pos_map2[doc_id]}
            for doc_id in set(pos_map2.keys()).difference(set(pos_map1.keys()))
        ]
        result["new_documents"] = new_docs

        # Find removed documents
        # Use a temporary variable with proper typing to avoid mypy errors
        removed_docs: List[Dict[str, Any]] = [
            {"document_id": doc_id, "previous_position": pos_map1[doc_id]}
            for doc_id in set(pos_map1.keys()).difference(set(pos_map2.keys()))
        ]
        result["removed_documents"] = removed_docs

        return result

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
        explanations = []

        # Extract results and explain info
        query_result1 = self._get_query_result(iter1, query)
        query_result2 = self._get_query_result(iter2, query)

        docs1 = query_result1.get("documents", [])
        docs2 = query_result2.get("documents", [])

        scores1 = query_result1.get("scores", {})
        scores2 = query_result2.get("scores", {})

        explain1 = query_result1.get("explain_info", {})
        explain2 = query_result2.get("explain_info", {})

        # Create position maps
        pos_map1 = {doc_id: pos for pos, doc_id in enumerate(docs1)}
        pos_map2 = {doc_id: pos for pos, doc_id in enumerate(docs2)}

        # Find documents in both result sets
        common_docs = set(pos_map1.keys()).intersection(set(pos_map2.keys()))

        # Analyze significant position changes
        for doc_id in common_docs:
            pos1 = pos_map1[doc_id]
            pos2 = pos_map2[doc_id]
            pos_change = pos1 - pos2

            # Skip if position change is not significant
            if abs(pos_change) < self.significant_rank_change:
                continue

            # Collect explanation data
            explanation = {
                "document_id": doc_id,
                "position_before": pos1,
                "position_after": pos2,
                "position_change": pos_change,
                "score_before": scores1.get(doc_id, 0.0),
                "score_after": scores2.get(doc_id, 0.0),
                "score_change": (scores2.get(doc_id, 0.0) - scores1.get(doc_id, 0.0)),
            }

            # Add explain info if available - use get() instead of 'in' operator
            if explain1.get(doc_id) is not None and explain2.get(doc_id) is not None:
                explanation["explanation"] = self._analyze_explain_differences(
                    explain1.get(doc_id), explain2.get(doc_id)
                )

            explanations.append(explanation)

        # Add explanations for new documents in top positions
        for doc_id in set(pos_map2.keys()).difference(set(pos_map1.keys())):
            pos2 = pos_map2[doc_id]

            # Only explain new documents in top positions
            if pos2 < self.analyze_top_n:
                explanation = {
                    "document_id": doc_id,
                    "position_before": None,
                    "position_after": pos2,
                    "position_change": "new",
                    "score_after": scores2.get(doc_id, 0.0),
                }

                # Add explain info if available - use get() instead of 'in' operator
                if explain2.get(doc_id) is not None:
                    explanation["explanation"] = self._summarize_explain_info(explain2.get(doc_id))

                explanations.append(explanation)

        # Sort explanations by absolute position change (most significant
        # first)
        explanations.sort(
            key=lambda x: (
                float("inf")
                if x["position_change"] == "new"  # New docs first
                else (abs(x["position_change"]) if isinstance(x["position_change"], (int, float)) else 0)
            ),
            reverse=True,
        )

        return explanations

    def find_significant_changes(self, iter1: IterationResult, iter2: IterationResult) -> List[Dict[str, Any]]:
        """
        Identify the most significant changes between iterations.

        Args:
            iter1: The first iteration result
            iter2: The second iteration result

        Returns:
            List of dictionaries describing significant changes
        """
        significant_changes = []

        # Compare overall metrics
        metric_deltas = self.compare_overall_metrics(iter1, iter2)

        # Find significant metric changes
        for metric, delta in metric_deltas.items():
            if abs(delta) >= self.significant_metric_threshold:
                # Handle metric_results safely
                overall1 = iter1.metric_results.get("overall", {}) if hasattr(iter1.metric_results, "get") else {}
                overall2 = iter2.metric_results.get("overall", {}) if hasattr(iter2.metric_results, "get") else {}

                significant_changes.append(
                    {
                        "type": "metric",
                        "metric": metric,
                        "delta": delta,
                        "before": overall1.get(metric),
                        "after": overall2.get(metric),
                    }
                )

        # Get queries from both iterations
        queries1 = set(iter1.query_results.keys())
        queries2 = set(iter2.query_results.keys())
        all_queries = queries1.union(queries2)

        # Check each query for significant changes
        for query in all_queries:
            # Skip if query doesn't exist in both iterations
            if query not in queries1 or query not in queries2:
                continue

            # Compare per-query metrics - handle list or dict type
            per_query1 = iter1.metric_results.get("per_query", {}) if hasattr(iter1.metric_results, "get") else {}
            per_query2 = iter2.metric_results.get("per_query", {}) if hasattr(iter2.metric_results, "get") else {}

            metrics1 = per_query1.get(query, {}) if hasattr(per_query1, "get") else {}
            metrics2 = per_query2.get(query, {}) if hasattr(per_query2, "get") else {}

            # Find queries with significant metric improvements or degradation
            for metric_name in set(metrics1.keys()).intersection(set(metrics2.keys())):
                value1 = metrics1.get(metric_name, 0.0)
                value2 = metrics2.get(metric_name, 0.0)
                delta = value2 - value1

                if abs(delta) >= self.significant_metric_threshold:
                    significant_changes.append(
                        {
                            "type": "query_metric",
                            "query": query,
                            "metric": metric_name,
                            "delta": delta,
                            "before": value1,
                            "after": value2,
                        }
                    )

            # Check for ranking changes in top results
            query_result1 = self._get_query_result(iter1, query)
            query_result2 = self._get_query_result(iter2, query)

            docs1 = query_result1.get("documents", [])[: self.analyze_top_n]
            docs2 = query_result2.get("documents", [])[: self.analyze_top_n]

            # Calculate Jaccard similarity of top results
            jaccard = (
                len(set(docs1).intersection(set(docs2))) / len(set(docs1).union(set(docs2))) if docs1 or docs2 else 1.0
            )

            # If top results are significantly different, report it
            if jaccard < 0.7:  # Less than 70% similarity
                significant_changes.append(
                    {
                        "type": "ranking_change",
                        "query": query,
                        "similarity": jaccard,
                        "before": docs1,
                        "after": docs2,
                    }
                )

        return significant_changes

    def generate_summary_report(self, iter1: IterationResult, iter2: IterationResult) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report comparing two iterations.

        Args:
            iter1: The first iteration result
            iter2: The second iteration result

        Returns:
            Dictionary with summary report information
        """
        # Initialize with proper typing for MyPy
        report: Dict[str, Any] = {
            "iteration1_id": iter1.iteration_id,
            "iteration2_id": iter2.iteration_id,
            "metrics_comparison": self.compare_overall_metrics(iter1, iter2),
            "config_changes": self.analyze_config_changes(iter1, iter2),
            "significant_changes": self.find_significant_changes(iter1, iter2),
            "improved_queries": [],
            "degraded_queries": [],
            "unchanged_queries": [],
            "query_level_details": {},
        }

        # Get primary metric if available
        primary_metric = None
        # Handle metric_results that could be a list or dict
        if hasattr(iter1.metric_results, "get"):  # Check if it's a dict-like object
            overall_metrics = iter1.metric_results.get("overall", {})
            if overall_metrics:
                # Convert to dict explicitly for type checking
                overall_metrics_dict = dict(overall_metrics)
                if overall_metrics_dict:
                    primary_metric = next(iter(overall_metrics_dict.keys()))

        # If we don't have a primary metric, we can't categorize queries
        if not primary_metric:
            return report

        # Categorize queries into improved, degraded, and unchanged
        queries1 = set(iter1.query_results.keys())
        queries2 = set(iter2.query_results.keys())
        common_queries = queries1.intersection(queries2)

        for query in common_queries:
            # Get query metrics - handle list or dict type
            per_query1 = iter1.metric_results.get("per_query", {}) if hasattr(iter1.metric_results, "get") else {}
            per_query2 = iter2.metric_results.get("per_query", {}) if hasattr(iter2.metric_results, "get") else {}

            query_metrics1 = per_query1.get(query, {}) if hasattr(per_query1, "get") else {}
            query_metrics2 = per_query2.get(query, {}) if hasattr(per_query2, "get") else {}

            # Convert to dict explicitly for type checking
            metrics1 = dict(query_metrics1) if query_metrics1 else {}
            metrics2 = dict(query_metrics2) if query_metrics2 else {}

            # Check if primary_metric exists in metrics using get() instead of 'in' operator
            if metrics1.get(primary_metric) is not None and metrics2.get(primary_metric) is not None:
                value1 = metrics1.get(primary_metric, 0.0)
                value2 = metrics2.get(primary_metric, 0.0)
                delta = value2 - value1

                query_summary = {
                    "query": query,
                    "primary_metric": {
                        "name": primary_metric,
                        "before": value1,
                        "after": value2,
                        "delta": delta,
                    },
                }

                # Add details to the right category - with proper typing
                query_summary_typed = cast(QuerySummary, query_summary)

                if delta > self.significant_metric_threshold:
                    report["improved_queries"] = report["improved_queries"] + [query_summary_typed]
                elif delta < -self.significant_metric_threshold:
                    report["degraded_queries"] = report["degraded_queries"] + [query_summary_typed]
                else:
                    report["unchanged_queries"] = report["unchanged_queries"] + [query_summary_typed]

                # Add detailed comparison for this query
                report["query_level_details"][query] = self.compare_query_results(iter1, iter2, query)

        # Sort the query lists by absolute delta
        improved_queries = report["improved_queries"]
        degraded_queries = report["degraded_queries"]

        # Sort as list and then reassign
        improved_queries_sorted = sorted(improved_queries, key=lambda x: x["primary_metric"]["delta"], reverse=True)
        degraded_queries_sorted = sorted(degraded_queries, key=lambda x: x["primary_metric"]["delta"])

        # Use explicit casting to fix the type error
        from typing import cast

        report["improved_queries"] = cast(List[QuerySummary], improved_queries_sorted)
        report["degraded_queries"] = cast(List[QuerySummary], degraded_queries_sorted)

        return report

    def analyze_config_changes(self, iter1: IterationResult, iter2: IterationResult) -> Dict[str, Any]:
        """
        Analyze what configuration changes were made between iterations.

        Args:
            iter1: The first iteration result
            iter2: The second iteration result

        Returns:
            Dictionary of configuration parameter to change description
        """
        changes = {}

        # Convert query_config to dict in a type-safe way
        config1 = {}
        if iter1.query_config:
            try:
                config1 = dict(iter1.query_config.__dict__)
            except Exception:
                config1 = {}

        config2 = {}
        if iter2.query_config:
            try:
                config2 = dict(iter2.query_config.__dict__)
            except Exception:
                config2 = {}

        # Remove iteration ID and non-relevant fields
        for config in [config1, config2]:
            for field in ["iteration_id", "timestamp"]:
                if field in config:
                    del config[field]

        # Find all parameter keys from both configurations
        all_params = set(config1.keys()).union(set(config2.keys()))

        for param in all_params:
            val1 = config1.get(param)
            val2 = config2.get(param)

            # Parameter exists in both configs but with different values
            if param in config1 and param in config2 and val1 != val2:
                changes[param] = {"before": val1, "after": val2, "type": "modified"}
            # Parameter exists only in the first config
            elif param in config1 and param not in config2:
                changes[param] = {"before": val1, "after": None, "type": "removed"}
            # Parameter exists only in the second config
            elif param not in config1 and param in config2:
                changes[param] = {"before": None, "after": val2, "type": "added"}

        return changes

    def _analyze_explain_differences(self, explain1: Dict, explain2: Dict) -> Dict[str, Any]:
        """
        Analyze the differences between two explain information dictionaries.

        Args:
            explain1: Explain information from first iteration
            explain2: Explain information from second iteration

        Returns:
            Dictionary with explain differences analysis
        """
        if not explain1 or not explain2:
            return {"error": "Missing explain information"}

        result = {
            "score_change": (explain2.get("value", 0) - explain1.get("value", 0)),
            "main_contributors": [],
            "new_components": [],
            "removed_components": [],
            "changed_components": [],
        }

        # Extract score components
        components1 = self._extract_score_components(explain1)
        components2 = self._extract_score_components(explain2)

        # Find new, removed, and changed components
        all_components = set(components1.keys()).union(set(components2.keys()))

        for component in all_components:
            val1 = components1.get(component, 0)
            val2 = components2.get(component, 0)

            # Use 'in' operator since we're checking dict keys directly
            if component in components1 and component in components2:
                delta = val2 - val1
                if abs(delta) > 0.001:  # Only include significant changes
                    result["changed_components"].append(
                        {
                            "component": component,
                            "before": val1,
                            "after": val2,
                            "delta": delta,
                        }
                    )
            elif component in components2:  # New component
                result["new_components"].append({"component": component, "value": val2})
            else:  # Removed component
                result["removed_components"].append({"component": component, "value": val1})

        # Sort by abs delta
        r_comp = result["changed_components"]
        r_comp.sort(key=lambda x: abs(x["delta"]), reverse=True)

        # Identify main contributors to score change
        if result["changed_components"]:
            result["main_contributors"] = result["changed_components"][:3]

        return result

    def _extract_score_components(self, explain: Dict) -> Dict[str, float]:
        """
        Extract score components from an explain information dictionary.

        Args:
            explain: Explain information dictionary

        Returns:
            Dictionary of component description to contribution value
        """
        components: Dict[str, float] = {}

        # Basic case - just use value if it exists
        if "value" in explain:
            value = explain["value"]
        else:
            return components

        # Extract description if it exists
        description = explain.get("description", "")
        if description:
            components[description] = value

        # Recursively process details
        details = explain.get("details", [])
        if details:
            for detail in details:
                sub_components = self._extract_score_components(detail)
                components.update(sub_components)

        return components

    def _summarize_explain_info(self, explain: Dict) -> Dict[str, Any]:
        """
        Create a simplified summary of explain information.

        Args:
            explain: Explain information dictionary

        Returns:
            Dictionary with explain summary
        """
        if not explain:
            return {"error": "No explain information available"}

        # Extract score components
        components = self._extract_score_components(explain)

        # Sort components by absolute contribution
        sorted_components = sorted(components.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "score": explain.get("value", 0),
            "top_contributors": [
                {"component": comp, "value": val} for comp, val in sorted_components[:5]  # Top 5 contributors
            ],
        }


class QuerySummary(TypedDict):
    """Type definition for query summary dictionaries."""

    query: str
    primary_metric: Dict[str, Any]


class SummaryReport(TypedDict):
    """Type definition for the summary report dictionary."""

    iteration1_id: str
    iteration2_id: str
    metrics_comparison: Dict[str, float]
    config_changes: Dict[str, Any]
    significant_changes: List[Dict[str, Any]]
    improved_queries: List[QuerySummary]
    degraded_queries: List[QuerySummary]
    unchanged_queries: List[QuerySummary]
    query_level_details: Dict[str, Any]
