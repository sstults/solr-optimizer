"""
Unit tests for corpus configuration models.
"""

import pytest
import json
import tempfile
from pathlib import Path

from solr_optimizer.models.corpus_config import CorpusReference, QuerySet, ReferenceRegistry


class TestCorpusReference:
    """Test cases for the CorpusReference class."""

    def test_valid_corpus_creation(self):
        """Test that a valid corpus reference can be created."""
        corpus = CorpusReference(
            name="test_corpus",
            collection="test_collection",
            solr_url="http://localhost:8983/solr",
            description="Test corpus description",
            metadata={"version": "1.0", "source": "test"}
        )

        assert corpus.name == "test_corpus"
        assert corpus.collection == "test_collection"
        assert corpus.solr_url == "http://localhost:8983/solr"
        assert corpus.description == "Test corpus description"
        assert corpus.metadata == {"version": "1.0", "source": "test"}

    def test_minimal_corpus_creation(self):
        """Test corpus creation with minimal required fields."""
        corpus = CorpusReference(
            name="minimal_corpus",
            collection="minimal_collection", 
            solr_url="http://localhost:8983/solr"
        )

        assert corpus.name == "minimal_corpus"
        assert corpus.collection == "minimal_collection"
        assert corpus.solr_url == "http://localhost:8983/solr"
        assert corpus.description is None
        assert corpus.metadata == {}

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        corpus = CorpusReference(
            name="test",
            collection="test",
            solr_url="http://localhost:8983/solr"
        )
        assert isinstance(corpus.metadata, dict)
        assert len(corpus.metadata) == 0


class TestQuerySet:
    """Test cases for the QuerySet class."""

    def test_valid_query_set_creation(self):
        """Test that a valid query set can be created."""
        queries = ["query 1", "query 2", "query 3"]
        judgments = {
            "query 1": {"doc1": 3, "doc2": 1, "doc3": 0},
            "query 2": {"doc4": 2, "doc5": 3},
            "query 3": {"doc6": 1, "doc7": 2, "doc8": 3}
        }

        query_set = QuerySet(
            name="test_queries",
            queries=queries,
            judgments=judgments,
            description="Test query set",
            judgment_scale="0-3: not relevant to highly relevant",
            metadata={"source": "manual"}
        )

        assert query_set.name == "test_queries"
        assert query_set.queries == queries
        assert query_set.judgments == judgments
        assert query_set.description == "Test query set"
        assert query_set.judgment_scale == "0-3: not relevant to highly relevant"
        assert query_set.metadata == {"source": "manual"}

    def test_minimal_query_set_creation(self):
        """Test query set creation with minimal required fields."""
        queries = ["query 1"]
        judgments = {"query 1": {"doc1": 1}}

        query_set = QuerySet(
            name="minimal",
            queries=queries,
            judgments=judgments
        )

        assert query_set.name == "minimal"
        assert query_set.queries == queries
        assert query_set.judgments == judgments
        assert query_set.description is None
        assert query_set.judgment_scale is None
        assert query_set.metadata == {}

    def test_missing_judgments_validation(self):
        """Test validation for missing judgments."""
        queries = ["query 1", "query 2", "query 3"]
        judgments = {
            "query 1": {"doc1": 1},
            "query 2": {"doc2": 2}
            # Missing query 3
        }

        with pytest.raises(ValueError, match="Missing judgments for queries: query 3"):
            QuerySet(
                name="test",
                queries=queries,
                judgments=judgments
            )

    def test_empty_queries_validation(self):
        """Test that empty queries list is handled."""
        query_set = QuerySet(
            name="empty",
            queries=[],
            judgments={}
        )
        assert len(query_set.queries) == 0
        assert len(query_set.judgments) == 0

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        query_set = QuerySet(
            name="test",
            queries=["q1"],
            judgments={"q1": {"doc1": 1}}
        )
        assert isinstance(query_set.metadata, dict)
        assert len(query_set.metadata) == 0


class TestReferenceRegistry:
    """Test cases for the ReferenceRegistry class."""

    def test_empty_registry_creation(self):
        """Test creating an empty registry."""
        registry = ReferenceRegistry()
        assert len(registry.corpora) == 0
        assert len(registry.query_sets) == 0

    def test_add_corpus(self):
        """Test adding a corpus to the registry."""
        registry = ReferenceRegistry()
        corpus = CorpusReference(
            name="test_corpus",
            collection="test_collection",
            solr_url="http://localhost:8983/solr"
        )

        registry.add_corpus(corpus)
        assert len(registry.corpora) == 1
        assert "test_corpus" in registry.corpora
        assert registry.corpora["test_corpus"] == corpus

    def test_add_duplicate_corpus(self):
        """Test that adding duplicate corpus raises error."""
        registry = ReferenceRegistry()
        corpus1 = CorpusReference(
            name="test_corpus",
            collection="collection1",
            solr_url="http://localhost:8983/solr"
        )
        corpus2 = CorpusReference(
            name="test_corpus",  # Same name
            collection="collection2",
            solr_url="http://localhost:8983/solr"
        )

        registry.add_corpus(corpus1)
        with pytest.raises(ValueError, match="Corpus 'test_corpus' already exists"):
            registry.add_corpus(corpus2)

    def test_add_query_set(self):
        """Test adding a query set to the registry."""
        registry = ReferenceRegistry()
        query_set = QuerySet(
            name="test_queries",
            queries=["query1"],
            judgments={"query1": {"doc1": 1}}
        )

        registry.add_query_set(query_set)
        assert len(registry.query_sets) == 1
        assert "test_queries" in registry.query_sets
        assert registry.query_sets["test_queries"] == query_set

    def test_add_duplicate_query_set(self):
        """Test that adding duplicate query set raises error."""
        registry = ReferenceRegistry()
        qs1 = QuerySet(name="test", queries=["q1"], judgments={"q1": {"doc1": 1}})
        qs2 = QuerySet(name="test", queries=["q2"], judgments={"q2": {"doc2": 2}})

        registry.add_query_set(qs1)
        with pytest.raises(ValueError, match="Query set 'test' already exists"):
            registry.add_query_set(qs2)

    def test_get_corpus(self):
        """Test getting a corpus by name."""
        registry = ReferenceRegistry()
        corpus = CorpusReference(
            name="test_corpus",
            collection="test_collection",
            solr_url="http://localhost:8983/solr"
        )
        registry.add_corpus(corpus)

        retrieved = registry.get_corpus("test_corpus")
        assert retrieved == corpus

    def test_get_nonexistent_corpus(self):
        """Test getting a nonexistent corpus raises KeyError."""
        registry = ReferenceRegistry()
        with pytest.raises(KeyError, match="Corpus 'nonexistent' not found"):
            registry.get_corpus("nonexistent")

    def test_get_query_set(self):
        """Test getting a query set by name."""
        registry = ReferenceRegistry()
        query_set = QuerySet(
            name="test_queries",
            queries=["query1"],
            judgments={"query1": {"doc1": 1}}
        )
        registry.add_query_set(query_set)

        retrieved = registry.get_query_set("test_queries")
        assert retrieved == query_set

    def test_get_nonexistent_query_set(self):
        """Test getting a nonexistent query set raises KeyError."""
        registry = ReferenceRegistry()
        with pytest.raises(KeyError, match="Query set 'nonexistent' not found"):
            registry.get_query_set("nonexistent")

    def test_list_corpora(self):
        """Test listing corpus names."""
        registry = ReferenceRegistry()
        
        # Empty registry
        assert registry.list_corpora() == []

        # Add corpora
        corpus1 = CorpusReference(name="corpus1", collection="c1", solr_url="http://localhost:8983/solr")
        corpus2 = CorpusReference(name="corpus2", collection="c2", solr_url="http://localhost:8983/solr")
        registry.add_corpus(corpus1)
        registry.add_corpus(corpus2)

        corpora = registry.list_corpora()
        assert len(corpora) == 2
        assert "corpus1" in corpora
        assert "corpus2" in corpora

    def test_list_query_sets(self):
        """Test listing query set names."""
        registry = ReferenceRegistry()
        
        # Empty registry
        assert registry.list_query_sets() == []

        # Add query sets
        qs1 = QuerySet(name="qs1", queries=["q1"], judgments={"q1": {"doc1": 1}})
        qs2 = QuerySet(name="qs2", queries=["q2"], judgments={"q2": {"doc2": 2}})
        registry.add_query_set(qs1)
        registry.add_query_set(qs2)

        query_sets = registry.list_query_sets()
        assert len(query_sets) == 2
        assert "qs1" in query_sets
        assert "qs2" in query_sets

    def test_remove_corpus(self):
        """Test removing a corpus."""
        registry = ReferenceRegistry()
        corpus = CorpusReference(name="test", collection="test", solr_url="http://localhost:8983/solr")
        registry.add_corpus(corpus)

        assert len(registry.corpora) == 1
        registry.remove_corpus("test")
        assert len(registry.corpora) == 0

    def test_remove_nonexistent_corpus(self):
        """Test removing a nonexistent corpus raises KeyError."""
        registry = ReferenceRegistry()
        with pytest.raises(KeyError, match="Corpus 'nonexistent' not found"):
            registry.remove_corpus("nonexistent")

    def test_remove_query_set(self):
        """Test removing a query set."""
        registry = ReferenceRegistry()
        query_set = QuerySet(name="test", queries=["q1"], judgments={"q1": {"doc1": 1}})
        registry.add_query_set(query_set)

        assert len(registry.query_sets) == 1
        registry.remove_query_set("test")
        assert len(registry.query_sets) == 0

    def test_remove_nonexistent_query_set(self):
        """Test removing a nonexistent query set raises KeyError."""
        registry = ReferenceRegistry()
        with pytest.raises(KeyError, match="Query set 'nonexistent' not found"):
            registry.remove_query_set("nonexistent")

    def test_to_dict(self):
        """Test converting registry to dictionary."""
        registry = ReferenceRegistry()
        
        corpus = CorpusReference(
            name="test_corpus",
            collection="test_collection",
            solr_url="http://localhost:8983/solr",
            description="Test corpus"
        )
        
        query_set = QuerySet(
            name="test_queries",
            queries=["query1"],
            judgments={"query1": {"doc1": 1}},
            description="Test queries"
        )
        
        registry.add_corpus(corpus)
        registry.add_query_set(query_set)

        data = registry.to_dict()
        assert "corpora" in data
        assert "query_sets" in data
        assert "test_corpus" in data["corpora"]
        assert "test_queries" in data["query_sets"]
        
        # Check corpus data
        corpus_data = data["corpora"]["test_corpus"]
        assert corpus_data["name"] == "test_corpus"
        assert corpus_data["collection"] == "test_collection"
        assert corpus_data["solr_url"] == "http://localhost:8983/solr"
        assert corpus_data["description"] == "Test corpus"
        
        # Check query set data
        qs_data = data["query_sets"]["test_queries"]
        assert qs_data["name"] == "test_queries"
        assert qs_data["queries"] == ["query1"]
        assert qs_data["judgments"] == {"query1": {"doc1": 1}}
        assert qs_data["description"] == "Test queries"

    def test_from_dict(self):
        """Test creating registry from dictionary."""
        data = {
            "corpora": {
                "test_corpus": {
                    "name": "test_corpus",
                    "collection": "test_collection",
                    "solr_url": "http://localhost:8983/solr",
                    "description": "Test corpus",
                    "metadata": {}
                }
            },
            "query_sets": {
                "test_queries": {
                    "name": "test_queries",
                    "queries": ["query1"],
                    "judgments": {"query1": {"doc1": 1}},
                    "description": "Test queries",
                    "judgment_scale": None,
                    "metadata": {}
                }
            }
        }

        registry = ReferenceRegistry.from_dict(data)
        
        assert len(registry.corpora) == 1
        assert len(registry.query_sets) == 1
        
        corpus = registry.get_corpus("test_corpus")
        assert corpus.name == "test_corpus"
        assert corpus.collection == "test_collection"
        assert corpus.solr_url == "http://localhost:8983/solr"
        assert corpus.description == "Test corpus"
        
        query_set = registry.get_query_set("test_queries")
        assert query_set.name == "test_queries"
        assert query_set.queries == ["query1"]
        assert query_set.judgments == {"query1": {"doc1": 1}}
        assert query_set.description == "Test queries"

    def test_save_and_load_from_file(self):
        """Test saving and loading registry from file."""
        registry = ReferenceRegistry()
        
        corpus = CorpusReference(
            name="test_corpus",
            collection="test_collection",
            solr_url="http://localhost:8983/solr"
        )
        
        query_set = QuerySet(
            name="test_queries",
            queries=["query1"],
            judgments={"query1": {"doc1": 1}}
        )
        
        registry.add_corpus(corpus)
        registry.add_query_set(query_set)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_registry.json"
            
            # Save registry
            registry.save_to_file(file_path)
            assert file_path.exists()
            
            # Load registry
            loaded_registry = ReferenceRegistry.load_from_file(file_path)
            
            assert len(loaded_registry.corpora) == 1
            assert len(loaded_registry.query_sets) == 1
            
            loaded_corpus = loaded_registry.get_corpus("test_corpus")
            assert loaded_corpus.name == corpus.name
            assert loaded_corpus.collection == corpus.collection
            assert loaded_corpus.solr_url == corpus.solr_url
            
            loaded_query_set = loaded_registry.get_query_set("test_queries")
            assert loaded_query_set.name == query_set.name
            assert loaded_query_set.queries == query_set.queries
            assert loaded_query_set.judgments == query_set.judgments

    def test_empty_dict_to_registry(self):
        """Test creating registry from empty dictionary."""
        registry = ReferenceRegistry.from_dict({})
        assert len(registry.corpora) == 0
        assert len(registry.query_sets) == 0

    def test_partial_dict_to_registry(self):
        """Test creating registry from partial dictionary."""
        # Only corpora, no query sets
        data = {
            "corpora": {
                "test_corpus": {
                    "name": "test_corpus",
                    "collection": "test_collection",
                    "solr_url": "http://localhost:8983/solr",
                    "description": None,
                    "metadata": {}
                }
            }
        }

        registry = ReferenceRegistry.from_dict(data)
        assert len(registry.corpora) == 1
        assert len(registry.query_sets) == 0
