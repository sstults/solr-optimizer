"""
Unit tests for the PySolrExecutionAgent class.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pysolr
from requests.exceptions import RequestException

from solr_optimizer.agents.solr.pysolr_execution_agent import PySolrExecutionAgent
from solr_optimizer.models.query_config import QueryConfig


class TestPySolrExecutionAgent:
    """Test cases for the PySolrExecutionAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.solr_url = "http://localhost:8983/solr"
        self.agent = PySolrExecutionAgent(self.solr_url, timeout=5)
        
        # Sample query config
        self.query_config = QueryConfig(
            query_parser="edismax",
            query_fields={"title": 2.0, "content": 1.0},
            phrase_fields={"title": 3.0},
            boost_queries=["category:electronics^1.5"],
            minimum_match="75%",
            tie_breaker=0.3
        )

    def test_initialization(self):
        """Test PySolrExecutionAgent initialization."""
        assert self.agent.solr_url == self.solr_url
        assert self.agent.timeout == 5
        assert self.agent.always_commit is False

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        agent = PySolrExecutionAgent(
            "http://example.com:8983/solr",
            timeout=30,
            always_commit=True
        )
        
        assert agent.solr_url == "http://example.com:8983/solr"
        assert agent.timeout == 30
        assert agent.always_commit is True

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_get_client(self, mock_solr_class):
        """Test Solr client creation."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        
        client = self.agent._get_client("test_collection")
        
        assert client == mock_client
        mock_solr_class.assert_called_once_with(
            "http://localhost:8983/solr/test_collection",
            timeout=5,
            always_commit=False
        )

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_execute_queries_success(self, mock_solr_class):
        """Test successful query execution."""
        # Mock Solr client
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        
        # Mock search results
        mock_client.search.return_value = Mock(
            docs=[
                {"id": "doc1", "title": "Test Document 1"},
                {"id": "doc2", "title": "Test Document 2"}
            ],
            hits=100,
            qtime=50
        )
        
        queries = ["test query 1", "test query 2"]
        results = self.agent.execute_queries("test_collection", queries, self.query_config)
        
        assert len(results) == 2
        assert "test query 1" in results
        assert "test query 2" in results
        
        for query in queries:
            query_result = results[query]
            assert "docs" in query_result
            assert "numFound" in query_result
            assert "responseTime" in query_result
            assert len(query_result["docs"]) == 2
            assert query_result["numFound"] == 100

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_execute_queries_with_params(self, mock_solr_class):
        """Test query execution with proper parameter formatting."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        mock_client.search.return_value = Mock(docs=[], hits=0, qtime=10)
        
        queries = ["test query"]
        self.agent.execute_queries("test_collection", queries, self.query_config)
        
        # Verify search was called with correct parameters
        call_args = mock_client.search.call_args
        assert call_args[0][0] == "test query"  # The query text
        
        params = call_args[1]
        assert params["defType"] == "edismax"
        assert params["qf"] == "title^2.0 content^1.0"
        assert params["pf"] == "title^3.0"
        assert params["bq"] == ["category:electronics^1.5"]
        assert params["mm"] == "75%"
        assert params["tie"] == 0.3

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_execute_queries_solr_error(self, mock_solr_class):
        """Test handling of Solr errors during query execution."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        mock_client.search.side_effect = pysolr.SolrError("Solr is down")
        
        queries = ["test query"]
        
        with pytest.raises(pysolr.SolrError):
            self.agent.execute_queries("test_collection", queries, self.query_config)

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_execute_queries_network_error(self, mock_solr_class):
        """Test handling of network errors during query execution."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        mock_client.search.side_effect = RequestException("Network error")
        
        queries = ["test query"]
        
        with pytest.raises(RequestException):
            self.agent.execute_queries("test_collection", queries, self.query_config)

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_fetch_schema_success(self, mock_solr_class):
        """Test successful schema fetching."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        
        # Mock schema response
        mock_schema = {
            "schema": {
                "fields": [
                    {"name": "id", "type": "string", "stored": True, "indexed": True},
                    {"name": "title", "type": "text_general", "stored": True, "indexed": True},
                    {"name": "content", "type": "text_general", "stored": True, "indexed": True}
                ],
                "fieldTypes": [
                    {"name": "string", "class": "solr.StrField"},
                    {"name": "text_general", "class": "solr.TextField"}
                ]
            }
        }
        
        # Mock the _send_request method
        with patch.object(mock_client, '_send_request') as mock_send:
            mock_send.return_value = mock_schema
            
            schema = self.agent.fetch_schema("test_collection")
            
            assert schema == mock_schema
            mock_send.assert_called_once_with('get', 'schema')

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_fetch_schema_error(self, mock_solr_class):
        """Test handling of schema fetch errors."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        
        with patch.object(mock_client, '_send_request') as mock_send:
            mock_send.side_effect = pysolr.SolrError("Schema not found")
            
            with pytest.raises(pysolr.SolrError):
                self.agent.fetch_schema("test_collection")

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_get_explain_info_success(self, mock_solr_class):
        """Test successful explain info retrieval."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        
        # Mock explain response
        mock_explain = {
            "explain": {
                "doc1": "1.0 = (MATCH) sum of:\n  0.5 = title boost\n  0.5 = content match"
            }
        }
        
        mock_client.search.return_value = Mock(debug=mock_explain)
        
        explain_info = self.agent.get_explain_info(
            "test_collection", 
            "test query", 
            "doc1", 
            self.query_config
        )
        
        assert explain_info == mock_explain
        
        # Verify search was called with debug parameters
        call_args = mock_client.search.call_args
        params = call_args[1]
        assert params["debug"] == "true"
        assert params["debugQuery"] == "true"

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_execute_streaming_expression_success(self, mock_solr_class):
        """Test successful streaming expression execution."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        
        # Mock streaming response
        mock_response = {
            "result-set": {
                "docs": [
                    {"id": "doc1", "score": 1.0},
                    {"id": "doc2", "score": 0.8}
                ]
            }
        }
        
        with patch.object(mock_client, '_send_request') as mock_send:
            mock_send.return_value = mock_response
            
            expression = "search(collection1, q=*:*, fl=id,score, sort=score desc)"
            result = self.agent.execute_streaming_expression("test_collection", expression)
            
            assert result == mock_response
            mock_send.assert_called_once_with('post', 'stream', body={"expr": expression})

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_execute_streaming_expression_error(self, mock_solr_class):
        """Test handling of streaming expression errors."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        
        with patch.object(mock_client, '_send_request') as mock_send:
            mock_send.side_effect = pysolr.SolrError("Invalid expression")
            
            expression = "invalid expression"
            
            with pytest.raises(pysolr.SolrError):
                self.agent.execute_streaming_expression("test_collection", expression)

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_ping_success(self, mock_solr_class):
        """Test successful ping."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        mock_client.ping.return_value = True
        
        result = self.agent.ping()
        
        assert result is True
        mock_client.ping.assert_called_once()

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_ping_failure(self, mock_solr_class):
        """Test ping failure."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        mock_client.ping.side_effect = Exception("Connection failed")
        
        result = self.agent.ping()
        
        assert result is False

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_client_caching(self, mock_solr_class):
        """Test that Solr clients are cached per collection."""
        mock_client1 = Mock()
        mock_client2 = Mock()
        mock_solr_class.side_effect = [mock_client1, mock_client2]
        
        # First call should create new client
        client1a = self.agent._get_client("collection1")
        assert client1a == mock_client1
        
        # Second call with same collection should return cached client
        client1b = self.agent._get_client("collection1")
        assert client1b == mock_client1
        
        # Call with different collection should create new client
        client2 = self.agent._get_client("collection2")
        assert client2 == mock_client2
        
        # Should have created only 2 clients total
        assert mock_solr_class.call_count == 2

    def test_empty_queries_list(self):
        """Test execution with empty queries list."""
        result = self.agent.execute_queries("test_collection", [], self.query_config)
        assert result == {}

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_query_with_no_results(self, mock_solr_class):
        """Test query execution that returns no results."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        mock_client.search.return_value = Mock(docs=[], hits=0, qtime=5)
        
        queries = ["nonexistent query"]
        results = self.agent.execute_queries("test_collection", queries, self.query_config)
        
        assert len(results) == 1
        query_result = results["nonexistent query"]
        assert len(query_result["docs"]) == 0
        assert query_result["numFound"] == 0
        assert query_result["responseTime"] == 5

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_query_with_large_result_set(self, mock_solr_class):
        """Test query execution with large result set."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        
        # Create mock docs for large result set
        mock_docs = [{"id": f"doc{i}", "title": f"Document {i}"} for i in range(100)]
        mock_client.search.return_value = Mock(docs=mock_docs, hits=10000, qtime=200)
        
        queries = ["popular query"]
        results = self.agent.execute_queries("test_collection", queries, self.query_config)
        
        query_result = results["popular query"]
        assert len(query_result["docs"]) == 100
        assert query_result["numFound"] == 10000
        assert query_result["responseTime"] == 200

    @patch('solr_optimizer.agents.solr.pysolr_execution_agent.pysolr.Solr')
    def test_query_config_none(self, mock_solr_class):
        """Test query execution with None query config."""
        mock_client = Mock()
        mock_solr_class.return_value = mock_client
        mock_client.search.return_value = Mock(docs=[], hits=0, qtime=5)
        
        queries = ["test query"]
        results = self.agent.execute_queries("test_collection", queries, None)
        
        # Should still work but with minimal parameters
        call_args = mock_client.search.call_args
        params = call_args[1]
        # Should not have any special query config parameters
        assert "defType" not in params or params.get("defType") is None
