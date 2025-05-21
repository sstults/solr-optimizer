"""
PySolr Execution Agent - Implementation of SolrExecutionAgent using PySolr.

This module provides a concrete implementation of the SolrExecutionAgent
interface using the PySolr library to communicate with Apache Solr.
"""

import json
import logging
from typing import Any, Dict, List

import pysolr
import requests

from solr_optimizer.agents.solr.solr_execution_agent import SolrExecutionAgent
from solr_optimizer.models.query_config import QueryConfig

logger = logging.getLogger(__name__)


class PySolrExecutionAgent(SolrExecutionAgent):
    """
    Implementation of SolrExecutionAgent using the PySolr library.
    """

    def __init__(self, solr_url: str, timeout: int = 10,
                 always_commit: bool = False):
        """
        Initialize the PySolr Execution Agent.

        Args:
            solr_url: Base URL for the Solr instance
                      (e.g., 'http://localhost:8983/solr')
            timeout: Connection timeout in seconds
            always_commit: Whether to always commit after write operations
        """
        self.base_url = solr_url.rstrip("/")
        self.timeout = timeout
        self.always_commit = always_commit
        self.solr_clients = {}  # Cache for Solr clients by collection
        logger.info(f"Initialized PySolrExecutionAgent with base URL: "
                    f"{self.base_url}")

    def _get_client(self, collection: str) -> pysolr.Solr:
        """
        Get or create a PySolr client for the specified collection.

        Args:
            collection: Solr collection name

        Returns:
            A PySolr client for the collection
        """
        if collection not in self.solr_clients:
            collection_url = f"{self.base_url}/{collection}"
            self.solr_clients[collection] = pysolr.Solr(
                collection_url, timeout=self.timeout,
                always_commit=self.always_commit
            )
        return self.solr_clients[collection]

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
        client = self._get_client(corpus)
        results = {}

        # Convert query_config to Solr params
        params = query_config.to_solr_params()

        # Add debug info if needed for explain
        params["debugQuery"] = "true"
        params["debug.explain.structured"] = "true"

        for query in queries:
            logger.debug(f"Executing query: {query} with params: {params}")

            try:
                response = client.search(query, **params)

                # Extract document IDs and scores
                documents = [doc["id"] for doc in response.docs]
                scores = {doc["id"]: doc.get("score", 0.0)
                          for doc in response.docs}

                # Extract explain info if available
                explain_info = {}
                if hasattr(response, "debug") and "explain" in response.debug:
                    explain_info = response.debug["explain"]

                results[query] = {
                    "documents": documents,
                    "scores": scores,
                    "explain_info": explain_info,
                    "total_results": response.hits,
                    "qtime": response.qtime if hasattr(response, "qtime")
                    else None,
                }

            except Exception as e:
                logger.error(f"Error executing query {query}: {str(e)}")
                results[query] = {
                    "documents": [],
                    "scores": {},
                    "explain_info": {},
                    "error": str(e),
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
        schema_url = f"{self.base_url}/{corpus}/schema"

        try:
            response = requests.get(schema_url)
            response.raise_for_status()
            return response.json().get("schema", {})
        except Exception as e:
            logger.error(f"Error fetching schema for {corpus}: {str(e)}")
            return {}

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
        client = self._get_client(corpus)
        params = query_config.to_solr_params()
        params["debugQuery"] = "true"
        params["debug.explain.structured"] = "true"

        try:
            response = client.search(query, **params)

            if hasattr(response, "debug") and "explain" in response.debug:
                # Find the explanation for the specific document
                if doc_id in response.debug["explain"]:
                    return response.debug["explain"][doc_id]

            logger.warning(f"No explain info found for document {doc_id}")
            return {}
        except Exception as e:
            logger.error(f"Error getting explain info: {str(e)}")
            return {}

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
        streaming_url = f"{self.base_url}/{corpus}/stream"

        try:
            response = requests.post(
                streaming_url,
                data=json.dumps({"expr": expression}),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error executing streaming expression: {str(e)}")
            return {"error": str(e)}

    def ping(self) -> bool:
        """
        Check if the Solr server is reachable.

        Returns:
            True if the server is reachable, False otherwise
        """
        admin_url = f"{self.base_url}/admin/ping"

        try:
            response = requests.get(admin_url)
            response.raise_for_status()
            return response.status_code == 200
        except Exception:
            logger.warning("Failed to ping Solr server")
            return False
