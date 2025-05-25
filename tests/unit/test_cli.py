"""
Unit tests for the CLI module functions.
"""

import pytest
import tempfile
import csv
from unittest.mock import Mock, patch, mock_open
from io import StringIO

from solr_optimizer.cli.main import (
    setup_logging,
    create_experiment_manager,
    load_queries_from_csv,
    load_judgments_from_csv,
    load_judgments_from_trec
)


class TestCLIUtilities:
    """Test cases for CLI utility functions."""

    def test_setup_logging_default(self):
        """Test logging setup with default verbosity."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging()
            mock_basic_config.assert_called_once_with(
                level=20,  # logging.INFO
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    def test_setup_logging_verbose(self):
        """Test logging setup with verbose mode."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging(verbose=True)
            mock_basic_config.assert_called_once_with(
                level=10,  # logging.DEBUG
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    def test_create_experiment_manager(self):
        """Test experiment manager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test that it creates the manager without mocking everything
            manager = create_experiment_manager("http://localhost:8983/solr", temp_dir)
            # Just verify we get a manager instance back
            assert manager is not None

    def test_load_queries_from_csv(self):
        """Test loading queries from CSV file."""
        csv_content = """query
first test query
second test query
third test query"""
        
        with patch('builtins.open', mock_open(read_data=csv_content)):
            queries = load_queries_from_csv('test.csv')
            
            assert len(queries) == 3
            assert queries[0] == "first test query"
            assert queries[1] == "second test query"
            assert queries[2] == "third test query"

    def test_load_queries_from_csv_missing_column(self):
        """Test loading queries from CSV with missing query column."""
        csv_content = """id,text
1,first test query"""
        
        with patch('builtins.open', mock_open(read_data=csv_content)):
            with pytest.raises(ValueError):
                load_queries_from_csv('test.csv')

    def test_load_judgments_from_csv(self):
        """Test loading judgments from CSV file."""
        csv_content = """query,document_id,relevance
query1,doc1,3
query1,doc2,1
query2,doc3,2
query2,doc4,0"""
        
        with patch('builtins.open', mock_open(read_data=csv_content)):
            judgments = load_judgments_from_csv('test.csv')
            
            assert len(judgments) == 2
            assert judgments["query1"]["doc1"] == 3.0
            assert judgments["query1"]["doc2"] == 1.0
            assert judgments["query2"]["doc3"] == 2.0
            assert judgments["query2"]["doc4"] == 0.0

    def test_load_judgments_from_csv_missing_columns(self):
        """Test loading judgments from CSV with missing columns."""
        csv_content = """query,document,score
query1,doc1,3"""
        
        with patch('builtins.open', mock_open(read_data=csv_content)):
            with pytest.raises(ValueError):
                load_judgments_from_csv('test.csv')

    def test_load_judgments_from_trec(self):
        """Test loading judgments from TREC format file."""
        trec_content = """query1 0 doc1 3
query1 0 doc2 1
query2 0 doc3 2
query2 0 doc4 0"""
        
        with patch('builtins.open', mock_open(read_data=trec_content)):
            judgments = load_judgments_from_trec('test.trec')
            
            assert len(judgments) == 2
            assert judgments["query1"]["doc1"] == 3.0
            assert judgments["query1"]["doc2"] == 1.0
            assert judgments["query2"]["doc3"] == 2.0
            assert judgments["query2"]["doc4"] == 0.0

    def test_load_judgments_from_trec_malformed_line(self):
        """Test loading judgments from TREC with malformed lines."""
        trec_content = """query1 0 doc1 3
malformed line
query2 0 doc2 1"""
        
        with patch('builtins.open', mock_open(read_data=trec_content)):
            judgments = load_judgments_from_trec('test.trec')
            
            # Should skip malformed lines and continue
            assert len(judgments) == 2
            assert judgments["query1"]["doc1"] == 3.0
            assert judgments["query2"]["doc2"] == 1.0

    def test_file_not_found_handling(self):
        """Test handling of file not found errors."""
        with patch('builtins.open') as mock_file:
            mock_file.side_effect = FileNotFoundError("File not found")
            
            with pytest.raises(FileNotFoundError):
                load_queries_from_csv('nonexistent.csv')

    def test_load_queries_empty_file(self):
        """Test loading queries from empty CSV file."""
        csv_content = """query"""
        
        with patch('builtins.open', mock_open(read_data=csv_content)):
            queries = load_queries_from_csv('empty.csv')
            assert len(queries) == 0

    def test_load_judgments_empty_file(self):
        """Test loading judgments from empty CSV file."""
        csv_content = """query,document_id,relevance"""
        
        with patch('builtins.open', mock_open(read_data=csv_content)):
            judgments = load_judgments_from_csv('empty.csv')
            assert len(judgments) == 0
