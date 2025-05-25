"""
PySolr Execution Agent - Implementation of SolrExecutionAgent using PySolr.

This module provides a concrete implementation of the SolrExecutionAgent
interface using the PySolr library to communicate with Apache Solr.
"""

import json
import logging
from typing import Any, Dict, List

import pysolr  # type: ignore[import-untyped]
import requests

from solr_optimizer.agents.solr.solr_execution_agent import SolrExecutionAgent
from solr_optimizer.models.query_config import QueryConfig

logger = logging.getLogger(__name__)


class PySolrExecutionAgent(SolrExecutionAgent):
    """
    Implementation of SolrExecutionAgent using the PySolr library.
    """

    def __init__(self, solr_url: str, timeout: int = 10, always_commit: bool = False):
        """
        Initialize the PySolr Execution Agent.

        Args:
            solr_url: Base URL for the Solr instance
                      (e.g., 'http://localhost:8983/solr')
            timeout: Connection timeout in seconds
            always_commit: Whether to always commit after write operations
        """
        self.solr_url = solr_url.rstrip("/")
        self.timeout = timeout
        self.always_commit = always_commit
        self.solr_clients: Dict[str, pysolr.Solr] = {}  # Cache for Solr clients by collection
        logger.info(f"Initialized PySolrExecutionAgent with base URL: " f"{self.solr_url}")

    def _get_client(self, collection: str) -> pysolr.Solr:
        """
        Get or create a PySolr client for the specified collection.

        Args:
            collection: Solr collection name

        Returns:
            A PySolr client for the collection
        """
        if collection not in self.solr_clients:
            collection_url = f"{self.solr_url}/{collection}"
            self.solr_clients[collection] = pysolr.Solr(
                collection_url, timeout=self.timeout, always_commit=self.always_commit
            )
        return self.solr_clients[collection]

    def execute_queries(self, corpus: str, queries: List[str], query_config: QueryConfig) -> Dict[str, Dict[str, Any]]:
        """
        Execute a set of queries against the specified Solr collection.

        Args:
            corpus: The Solr collection/core name
            queries: List of query strings to execute
            query_config: Configuration for the queries

        Returns:
            Dictionary mapping query string to query results
        """
        if not queries:
            return {}
            
        client = self._get_client(corpus)
        results = {}

        # Handle None query_config
        if query_config is None:
            params = {}
        else:
            # Convert query_config to Solr params
            params = query_config.to_solr_params()

        # Add debug info if needed for explain
        params["debug"] = "true"
        params["debugQuery"] = "true"

        for query in queries:
            logger.debug(f"Executing query: {query} with params: {params}")

            response = client.search(query, **params)

            results[query] = {
                "docs": response.docs,
                "numFound": response.hits,
                "responseTime": response.qtime if hasattr(response, "qtime") else None,
            }

        return results

    def fetch_schema(self, corpus: str) -> Dict[str, Any]:
        """
        Retrieve the schema information for a Solr collection.

        Args:
            corpus: The Solr collection/core name

        Returns:
            Schema information as a dictionary
        """
        client = self._get_client(corpus)
        
        return client._send_request('get', 'schema')

    def get_explain_info(self, corpus: str, query: str, doc_id: str, query_config: QueryConfig) -> Dict[str, Any]:
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
        client = self._get_client(corpus)
        params = query_config.to_solr_params()
        params["debug"] = "true"
        params["debugQuery"] = "true"

        response = client.search(query, **params)

        if hasattr(response, "debug"):
            return response.debug

        logger.warning(f"No explain info found for document {doc_id}")
        return {}

    def execute_streaming_expression(self, corpus: str, expression: str) -> Dict[str, Any]:
        """
        Execute a Solr streaming expression.

        Args:
            corpus: The Solr collection/core name
            expression: The streaming expression to execute

        Returns:
            The results of the streaming expression
        """
        client = self._get_client(corpus)
        
        return client._send_request('post', 'stream', body={"expr": expression})

    def ping(self) -> bool:
        """
        Check if the Solr server is reachable.

        Returns:
            True if the server is reachable, False otherwise
        """
        try:
            # Use a temporary client for ping - create one for the base collection
            temp_client = pysolr.Solr(f"{self.solr_url}/admin/cores", timeout=self.timeout)
            return temp_client.ping()
        except Exception:
            logger.warning("Failed to ping Solr server")
            return False
