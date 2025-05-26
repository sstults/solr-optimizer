"""
Unit tests for the Reference Service.
"""

import pytest
import tempfile
import csv
import json
from pathlib import Path
from unittest.mock import Mock, patch

from solr_optimizer.services.reference_service import ReferenceService
from solr_optimizer.models.corpus_config import CorpusReference, QuerySet, ReferenceRegistry
from solr_optimizer.models.experiment_config import ExperimentConfig


class TestReferenceService:
    """Test cases for the ReferenceService class."""

    def test_init_with_new_registry_file(self):
        """Test initializing service with new registry file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "new_registry.json"
            service = ReferenceService(registry_file)
            
            assert service.registry_file == registry_file
            assert isinstance(service.registry, ReferenceRegistry)
            assert len(service.registry.corpora) == 0
            assert len(service.registry.query_sets) == 0

    def test_init_with_existing_registry_file(self):
        """Test initializing service with existing registry file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "existing_registry.json"
            
            # Create a registry file with some data
            initial_registry = ReferenceRegistry()
            corpus = CorpusReference(
                name="test_corpus",
                collection="test_collection",
                solr_url="http://localhost:8983/solr"
            )
            initial_registry.add_corpus(corpus)
            initial_registry.save_to_file(registry_file)
            
            # Load service
            service = ReferenceService(registry_file)
            
            assert len(service.registry.corpora) == 1
            assert "test_corpus" in service.registry.corpora

    def test_init_with_corrupted_registry_file(self):
        """Test initializing service with corrupted registry file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "corrupted_registry.json"
            
            # Create a corrupted JSON file
            with open(registry_file, 'w') as f:
                f.write("{ invalid json")
            
            # Should handle gracefully and create new registry
            service = ReferenceService(registry_file)
            assert isinstance(service.registry, ReferenceRegistry)
            assert len(service.registry.corpora) == 0

    def test_add_corpus(self):
        """Test adding a corpus reference."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            corpus = service.add_corpus(
                name="ecommerce",
                collection="products",
                solr_url="http://localhost:8983/solr",
                description="E-commerce product catalog",
                metadata={"version": "1.0"}
            )
            
            assert isinstance(corpus, CorpusReference)
            assert corpus.name == "ecommerce"
            assert corpus.collection == "products"
            assert corpus.solr_url == "http://localhost:8983/solr"
            assert corpus.description == "E-commerce product catalog"
            assert corpus.metadata == {"version": "1.0"}
            
            # Check it was added to registry
            assert len(service.registry.corpora) == 1
            assert "ecommerce" in service.registry.corpora
            
            # Check it was saved to file
            assert registry_file.exists()

    def test_add_corpus_duplicate_name(self):
        """Test adding corpus with duplicate name raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            service.add_corpus("test", "collection1", "http://localhost:8983/solr")
            
            with pytest.raises(ValueError, match="Corpus 'test' already exists"):
                service.add_corpus("test", "collection2", "http://localhost:8983/solr")

    def test_add_query_set_from_data(self):
        """Test adding query set from provided data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            queries = ["laptop", "smartphone", "tablet"]
            judgments = {
                "laptop": {"doc1": 3, "doc2": 1, "doc3": 0},
                "smartphone": {"doc4": 2, "doc5": 3},
                "tablet": {"doc6": 1, "doc7": 2}
            }
            
            query_set = service.add_query_set_from_data(
                name="test_queries",
                queries=queries,
                judgments=judgments,
                description="Test query set",
                judgment_scale="0-3: not relevant to highly relevant",
                metadata={"source": "manual"}
            )
            
            assert isinstance(query_set, QuerySet)
            assert query_set.name == "test_queries"
            assert query_set.queries == queries
            assert query_set.judgments == judgments
            assert query_set.description == "Test query set"
            assert query_set.judgment_scale == "0-3: not relevant to highly relevant"
            assert query_set.metadata == {"source": "manual"}
            
            # Check it was added to registry
            assert len(service.registry.query_sets) == 1
            assert "test_queries" in service.registry.query_sets

    def test_add_query_set_from_csv(self):
        """Test adding query set from CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            csv_file = Path(temp_dir) / "queries.csv"
            
            # Create test CSV file
            csv_data = [
                ["query", "document_id", "relevance_score"],
                ["laptop", "doc1", "3"],
                ["laptop", "doc2", "1"],
                ["smartphone", "doc3", "2"],
                ["smartphone", "doc4", "3"]
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            service = ReferenceService(registry_file)
            query_set = service.add_query_set_from_csv(
                name="csv_queries",
                csv_file=csv_file,
                description="Queries from CSV"
            )
            
            assert query_set.name == "csv_queries"
            assert len(query_set.queries) == 2
            assert "laptop" in query_set.queries
            assert "smartphone" in query_set.queries
            assert query_set.judgments["laptop"]["doc1"] == 3
            assert query_set.judgments["laptop"]["doc2"] == 1
            assert query_set.judgments["smartphone"]["doc3"] == 2
            assert query_set.judgments["smartphone"]["doc4"] == 3
            assert query_set.description == "Queries from CSV"
            assert query_set.metadata["source_file"] == str(csv_file)

    def test_add_query_set_from_csv_invalid_format(self):
        """Test adding query set from invalid CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            csv_file = Path(temp_dir) / "invalid_queries.csv"
            
            # Create CSV with missing required columns
            csv_data = [
                ["query", "document"],  # Missing relevance_score column
                ["laptop", "doc1"],
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            service = ReferenceService(registry_file)
            
            with pytest.raises(ValueError, match="CSV must contain columns"):
                service.add_query_set_from_csv("invalid", csv_file)

    def test_add_query_set_from_csv_empty_file(self):
        """Test adding query set from empty CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            csv_file = Path(temp_dir) / "empty_queries.csv"
            
            # Create CSV with only headers
            csv_data = [["query", "document_id", "relevance_score"]]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            service = ReferenceService(registry_file)
            
            with pytest.raises(ValueError, match="No valid queries found in CSV file"):
                service.add_query_set_from_csv("empty", csv_file)

    def test_add_query_set_from_trec(self):
        """Test adding query set from TREC format files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            queries_file = Path(temp_dir) / "queries.txt"
            qrels_file = Path(temp_dir) / "qrels.txt"
            
            # Create queries file
            with open(queries_file, 'w') as f:
                f.write("1:laptop computers\n")
                f.write("2:smartphone reviews\n")
                f.write("3:tablet comparison\n")
            
            # Create qrels file
            with open(qrels_file, 'w') as f:
                f.write("1 0 doc1 3\n")
                f.write("1 0 doc2 1\n")
                f.write("2 0 doc3 2\n")
                f.write("2 0 doc4 3\n")
                f.write("3 0 doc5 1\n")
            
            service = ReferenceService(registry_file)
            query_set = service.add_query_set_from_trec(
                name="trec_queries",
                queries_file=queries_file,
                qrels_file=qrels_file,
                description="TREC format queries"
            )
            
            assert query_set.name == "trec_queries"
            assert len(query_set.queries) == 3
            assert "laptop computers" in query_set.queries
            assert "smartphone reviews" in query_set.queries
            assert "tablet comparison" in query_set.queries
            assert query_set.judgments["laptop computers"]["doc1"] == 3
            assert query_set.judgments["smartphone reviews"]["doc3"] == 2
            assert query_set.description == "TREC format queries"

    def test_add_query_set_from_trec_simple_format(self):
        """Test adding query set from TREC files with simple query format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            queries_file = Path(temp_dir) / "simple_queries.txt"
            qrels_file = Path(temp_dir) / "qrels.txt"
            
            # Create queries file without IDs
            with open(queries_file, 'w') as f:
                f.write("laptop computers\n")
                f.write("smartphone reviews\n")
            
            # Create qrels file using line numbers as IDs
            with open(qrels_file, 'w') as f:
                f.write("1 0 doc1 3\n")
                f.write("2 0 doc3 2\n")
            
            service = ReferenceService(registry_file)
            query_set = service.add_query_set_from_trec(
                name="simple_trec",
                queries_file=queries_file,
                qrels_file=qrels_file
            )
            
            assert len(query_set.queries) == 2
            assert "laptop computers" in query_set.queries
            assert "smartphone reviews" in query_set.queries

    def test_get_corpus(self):
        """Test getting a corpus by name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            original_corpus = service.add_corpus("test", "collection", "http://localhost:8983/solr")
            retrieved_corpus = service.get_corpus("test")
            
            assert retrieved_corpus == original_corpus

    def test_get_nonexistent_corpus(self):
        """Test getting a nonexistent corpus raises KeyError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            with pytest.raises(KeyError, match="Corpus 'nonexistent' not found"):
                service.get_corpus("nonexistent")

    def test_get_query_set(self):
        """Test getting a query set by name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            original_qs = service.add_query_set_from_data(
                "test", ["q1"], {"q1": {"doc1": 1}}
            )
            retrieved_qs = service.get_query_set("test")
            
            assert retrieved_qs == original_qs

    def test_get_nonexistent_query_set(self):
        """Test getting a nonexistent query set raises KeyError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            with pytest.raises(KeyError, match="Query set 'nonexistent' not found"):
                service.get_query_set("nonexistent")

    def test_list_corpora(self):
        """Test listing available corpora."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            # Empty list initially
            assert service.list_corpora() == []
            
            # Add some corpora
            service.add_corpus("corpus1", "c1", "http://localhost:8983/solr")
            service.add_corpus("corpus2", "c2", "http://localhost:8983/solr")
            
            corpora = service.list_corpora()
            assert len(corpora) == 2
            assert "corpus1" in corpora
            assert "corpus2" in corpora

    def test_list_query_sets(self):
        """Test listing available query sets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            # Empty list initially
            assert service.list_query_sets() == []
            
            # Add some query sets
            service.add_query_set_from_data("qs1", ["q1"], {"q1": {"doc1": 1}})
            service.add_query_set_from_data("qs2", ["q2"], {"q2": {"doc2": 2}})
            
            query_sets = service.list_query_sets()
            assert len(query_sets) == 2
            assert "qs1" in query_sets
            assert "qs2" in query_sets

    def test_remove_corpus(self):
        """Test removing a corpus."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            service.add_corpus("test", "collection", "http://localhost:8983/solr")
            assert len(service.list_corpora()) == 1
            
            service.remove_corpus("test")
            assert len(service.list_corpora()) == 0

    def test_remove_query_set(self):
        """Test removing a query set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            service.add_query_set_from_data("test", ["q1"], {"q1": {"doc1": 1}})
            assert len(service.list_query_sets()) == 1
            
            service.remove_query_set("test")
            assert len(service.list_query_sets()) == 0

    def test_create_experiment_config(self):
        """Test creating experiment configuration from references."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            # Add corpus and query set
            service.add_corpus("ecommerce", "products", "http://localhost:8983/solr")
            service.add_query_set_from_data(
                "test_queries",
                ["laptop", "smartphone"],
                {
                    "laptop": {"doc1": 3, "doc2": 1},
                    "smartphone": {"doc3": 2, "doc4": 3}
                }
            )
            
            # Create experiment config
            config = service.create_experiment_config(
                experiment_id="exp1",
                corpus_name="ecommerce",
                query_set_name="test_queries",
                primary_metric="ndcg",
                metric_depth=10,
                secondary_metrics=["precision", "recall"],
                description="Test experiment"
            )
            
            assert isinstance(config, ExperimentConfig)
            assert config.experiment_id == "exp1"
            assert config.corpus == "products"  # Should use collection name
            assert config.queries == ["laptop", "smartphone"]
            assert config.judgments["laptop"]["doc1"] == 3
            assert config.primary_metric == "ndcg"
            assert config.metric_depth == 10
            assert config.secondary_metrics == ["precision", "recall"]
            assert config.description == "Test experiment"

    def test_create_experiment_config_nonexistent_corpus(self):
        """Test creating experiment config with nonexistent corpus."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            service.add_query_set_from_data("test_queries", ["q1"], {"q1": {"doc1": 1}})
            
            with pytest.raises(KeyError, match="Corpus 'nonexistent' not found"):
                service.create_experiment_config(
                    "exp1", "nonexistent", "test_queries", "ndcg", 10
                )

    def test_create_experiment_config_nonexistent_query_set(self):
        """Test creating experiment config with nonexistent query set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            service.add_corpus("test_corpus", "collection", "http://localhost:8983/solr")
            
            with pytest.raises(KeyError, match="Query set 'nonexistent' not found"):
                service.create_experiment_config(
                    "exp1", "test_corpus", "nonexistent", "ndcg", 10
                )

    def test_get_registry_summary(self):
        """Test getting registry summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            # Add some data
            service.add_corpus(
                "ecommerce", "products", "http://localhost:8983/solr",
                description="E-commerce corpus"
            )
            service.add_query_set_from_data(
                "test_queries", ["q1", "q2"], 
                {"q1": {"doc1": 1}, "q2": {"doc2": 2}},
                description="Test queries",
                judgment_scale="0-3"
            )
            
            summary = service.get_registry_summary()
            
            assert summary["corpora_count"] == 1
            assert summary["query_sets_count"] == 1
            
            assert len(summary["corpora"]) == 1
            corpus_info = summary["corpora"][0]
            assert corpus_info["name"] == "ecommerce"
            assert corpus_info["collection"] == "products"
            assert corpus_info["solr_url"] == "http://localhost:8983/solr"
            assert corpus_info["description"] == "E-commerce corpus"
            
            assert len(summary["query_sets"]) == 1
            qs_info = summary["query_sets"][0]
            assert qs_info["name"] == "test_queries"
            assert qs_info["query_count"] == 2
            assert qs_info["judgment_scale"] == "0-3"
            assert qs_info["description"] == "Test queries"

    def test_export_query_set_to_csv(self):
        """Test exporting query set to CSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            output_file = Path(temp_dir) / "exported_queries.csv"
            service = ReferenceService(registry_file)
            
            # Add query set
            service.add_query_set_from_data(
                "test_queries",
                ["laptop", "smartphone"],
                {
                    "laptop": {"doc1": 3, "doc2": 1},
                    "smartphone": {"doc3": 2}
                }
            )
            
            # Export to CSV
            service.export_query_set_to_csv("test_queries", output_file)
            
            # Verify exported file
            assert output_file.exists()
            
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 3  # 2 for laptop + 1 for smartphone
            
            # Check headers
            assert reader.fieldnames == ['query', 'document_id', 'relevance_score']
            
            # Check data (order might vary)
            laptop_rows = [r for r in rows if r['query'] == 'laptop']
            smartphone_rows = [r for r in rows if r['query'] == 'smartphone']
            
            assert len(laptop_rows) == 2
            assert len(smartphone_rows) == 1
            
            # Verify specific entries
            doc1_row = next(r for r in laptop_rows if r['document_id'] == 'doc1')
            assert doc1_row['relevance_score'] == '3'

    def test_export_nonexistent_query_set_to_csv(self):
        """Test exporting nonexistent query set raises KeyError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            output_file = Path(temp_dir) / "exported_queries.csv"
            service = ReferenceService(registry_file)
            
            with pytest.raises(KeyError, match="Query set 'nonexistent' not found"):
                service.export_query_set_to_csv("nonexistent", output_file)

    @patch('solr_optimizer.services.reference_service.ReferenceService.save_registry')
    def test_save_registry_error_handling(self, mock_save):
        """Test save registry error handling."""
        mock_save.side_effect = IOError("Permission denied")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_file = Path(temp_dir) / "test_registry.json"
            service = ReferenceService(registry_file)
            
            with pytest.raises(IOError):
                service.add_corpus("test", "collection", "http://localhost:8983/solr")

    def test_default_registry_file_path(self):
        """Test default registry file path."""
        service = ReferenceService()
        assert service.registry_file == Path("references.json")
