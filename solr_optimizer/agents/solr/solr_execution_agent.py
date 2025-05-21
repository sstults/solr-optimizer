"""
Solr Execution Agent - Interface for SolrCloud interactions.

This module defines the SolrExecutionAgent interface which is responsible for
executing queries against a Solr cluster and retrieving results.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from solr_optimizer.models.query_config import QueryConfig


class SolrExecutionAgent(ABC):
    """
    Agent responsible for executing queries against a Solr cluster and
    retrieving results.
    """

    @abstractmethod
    def execute_queries(
        self, corpus: str, queries: List[str], query_config: QueryConfig
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute a set of queries against the specified Solr collection.

        Args:
            corpus: The Solr collection/core name
            queries: List of query strings to execute
            query_config: Configuration for the queries

        Returns:
            Dictionary mapping query string to query results
        """
        pass

    @abstractmethod
    def fetch_schema(self, corpus: str) -> Dict[str, Any]:
        """
        Retrieve the schema information for a Solr collection.

        Args:
            corpus: The Solr collection/core name

        Returns:
            Schema information as a dictionary
        """
        pass

    @abstractmethod
    def get_explain_info(
        self, corpus: str, query: str, doc_id: str, query_config: QueryConfig
    ) -> Dict[str, Any]:
        """
        Get the Solr explain information for a specific document in a query.

        Args:
            corpus: The Solr collection/core name
            query: The query string
            doc_id: The document ID to explain
            query_config: Query configuration

        Returns:
            The explain information as a dictionary
        """
        pass

    @abstractmethod
    def execute_streaming_expression(
        self, corpus: str, expression: str
    ) -> Dict[str, Any]:
        """
        Execute a Solr streaming expression.

        Args:
            corpus: The Solr collection/core name
            expression: The streaming expression to execute

        Returns:
            The results of the streaming expression
        """
        pass

    @abstractmethod
    def ping(self) -> bool:
        """
        Check if the Solr server is reachable.

        Returns:
            True if the server is reachable, False otherwise
        """
        pass
