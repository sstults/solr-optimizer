"""
Unit tests for database persistence services.
"""

import pytest
import tempfile
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from solr_optimizer.persistence.database_service import DatabaseService, SQLiteService, PostgreSQLService
from solr_optimizer.persistence.persistence_interface import PersistenceInterface
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.iteration_result import IterationResult, QueryResult, MetricResult
from solr_optimizer.models.query_config import QueryConfig
from solr_optimizer.models.corpus_config import CorpusReference, QuerySet, ReferenceRegistry


class TestDatabaseService:
    """Test cases for the base DatabaseService class."""

    def test_is_abstract_interface_implementation(self):
        """Test that DatabaseService implements PersistenceInterface."""
        assert issubclass(DatabaseService, PersistenceInterface)

    def test_get_connection_not_implemented(self):
        """Test that _get_connection raises NotImplementedError."""
        service = DatabaseService("test_connection")
        
        with pytest.raises(NotImplementedError):
            service._get_connection()

    def test_initialization(self):
        """Test basic initialization."""
        service = DatabaseService("test_connection_string")
        assert service.connection_string == "test_connection_string"
        assert service._connection is None


class TestSQLiteService:
    """Test cases for the SQLiteService class."""

    def test_initialization(self):
        """Test SQLite service initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            
            assert service.db_path == db_path
            assert str(db_path) in service.connection_string

    def test_initialization_default_path(self):
        """Test SQLite service with default path."""
        service = SQLiteService()
        assert service.db_path == Path("solr_optimizer.db")

    def test_get_connection(self):
        """Test getting SQLite connection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            
            conn = service._get_connection()
            assert isinstance(conn, sqlite3.Connection)
            assert conn is service._connection
            
            # Second call should return same connection
            conn2 = service._get_connection()
            assert conn2 is conn

    def test_initialize_creates_tables(self):
        """Test that initialize creates necessary tables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            conn = service._get_connection()
            cursor = conn.cursor()
            
            # Check that tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "experiments" in tables
            assert "iterations" in tables
            assert "reference_registry" in tables

    def test_close_connection(self):
        """Test closing database connection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            
            # Get connection
            conn = service._get_connection()
            assert service._connection is not None
            
            # Close connection
            service.close()
            assert service._connection is None

    def test_save_and_load_experiment(self):
        """Test saving and loading experiment configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Create test experiment
            experiment = ExperimentConfig(
                experiment_id="test_exp",
                corpus="test_collection",
                queries=["query1", "query2"],
                judgments={
                    "query1": {"doc1": 3, "doc2": 1},
                    "query2": {"doc3": 2, "doc4": 3}
                },
                primary_metric="ndcg",
                metric_depth=10,
                secondary_metrics=["precision", "recall"],
                description="Test experiment"
            )
            
            # Save experiment
            service.save_experiment(experiment)
            
            # Load experiment
            loaded_experiment = service.load_experiment("test_exp")
            
            assert loaded_experiment is not None
            assert loaded_experiment.experiment_id == "test_exp"
            assert loaded_experiment.corpus == "test_collection"
            assert loaded_experiment.queries == ["query1", "query2"]
            assert loaded_experiment.judgments["query1"]["doc1"] == 3
            assert loaded_experiment.primary_metric == "ndcg"
            assert loaded_experiment.metric_depth == 10
            assert loaded_experiment.secondary_metrics == ["precision", "recall"]
            assert loaded_experiment.description == "Test experiment"

    def test_load_nonexistent_experiment(self):
        """Test loading nonexistent experiment returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            result = service.load_experiment("nonexistent")
            assert result is None

    def test_list_experiments(self):
        """Test listing experiments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Initially empty
            experiments = service.list_experiments()
            assert len(experiments) == 0
            
            # Add experiments
            exp1 = ExperimentConfig(
                experiment_id="exp1",
                corpus="collection1",
                queries=["q1"],
                judgments={"q1": {"doc1": 1}},
                primary_metric="ndcg",
                metric_depth=10
            )
            
            exp2 = ExperimentConfig(
                experiment_id="exp2",
                corpus="collection2",
                queries=["q2"],
                judgments={"q2": {"doc2": 2}},
                primary_metric="precision",
                metric_depth=5
            )
            
            service.save_experiment(exp1)
            service.save_experiment(exp2)
            
            # List experiments
            experiments = service.list_experiments()
            assert len(experiments) == 2
            assert "exp1" in experiments
            assert "exp2" in experiments

    def test_delete_experiment(self):
        """Test deleting experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Add experiment
            experiment = ExperimentConfig(
                experiment_id="test_exp",
                corpus="test_collection",
                queries=["q1"],
                judgments={"q1": {"doc1": 1}},
                primary_metric="ndcg",
                metric_depth=10
            )
            service.save_experiment(experiment)
            
            # Add iteration
            iteration = IterationResult(
                iteration_id="iter1",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime.now()
            )
            service.save_iteration(iteration)
            
            # Verify they exist
            assert service.load_experiment("test_exp") is not None
            assert len(service.list_iterations("test_exp")) == 1
            
            # Delete experiment
            deleted = service.delete_experiment("test_exp")
            assert deleted is True
            
            # Verify deletion
            assert service.load_experiment("test_exp") is None
            assert len(service.list_iterations("test_exp")) == 0

    def test_delete_nonexistent_experiment(self):
        """Test deleting nonexistent experiment returns False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            deleted = service.delete_experiment("nonexistent")
            assert deleted is False

    def test_save_and_load_iteration(self):
        """Test saving and loading iteration results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Create test iteration
            query_config = QueryConfig(
                query_parser="edismax",
                query_fields={"title": 2.0, "content": 1.0}
            )
            
            query_results = {
                "laptop": QueryResult(
                    query="laptop",
                    documents=["doc1", "doc2"],
                    scores=[0.9, 0.7],
                    explain_info={"doc1": "explanation"}
                )
            }
            
            metric_results = [
                MetricResult(
                    metric_name="ndcg",
                    value=0.85,
                    per_query={"laptop": 0.85}
                )
            ]
            
            iteration = IterationResult(
                iteration_id="iter1",
                experiment_id="test_exp",
                query_config=query_config,
                query_results=query_results,
                metric_results=metric_results,
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                compared_to="baseline",
                metric_deltas={"ndcg": 0.05},
                notes="Test iteration"
            )
            
            # Save iteration
            service.save_iteration(iteration)
            
            # Load iteration
            loaded_iteration = service.load_iteration("test_exp", "iter1")
            
            assert loaded_iteration is not None
            assert loaded_iteration.iteration_id == "iter1"
            assert loaded_iteration.experiment_id == "test_exp"
            assert loaded_iteration.query_config.query_parser == "edismax"
            assert loaded_iteration.query_results["laptop"].query == "laptop"
            assert len(loaded_iteration.metric_results) == 1
            assert loaded_iteration.metric_results[0].metric_name == "ndcg"
            assert loaded_iteration.compared_to == "baseline"
            assert loaded_iteration.metric_deltas["ndcg"] == 0.05
            assert loaded_iteration.notes == "Test iteration"

    def test_load_nonexistent_iteration(self):
        """Test loading nonexistent iteration returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            result = service.load_iteration("test_exp", "nonexistent")
            assert result is None

    def test_list_iterations(self):
        """Test listing iterations for an experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Initially empty
            iterations = service.list_iterations("test_exp")
            assert len(iterations) == 0
            
            # Add iterations
            iter1 = IterationResult(
                iteration_id="iter1",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime(2024, 1, 1)
            )
            
            iter2 = IterationResult(
                iteration_id="iter2", 
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime(2024, 1, 2)
            )
            
            service.save_iteration(iter1)
            service.save_iteration(iter2)
            
            # List iterations
            iterations = service.list_iterations("test_exp")
            assert len(iterations) == 2
            assert "iter1" in iterations
            assert "iter2" in iterations

    def test_get_latest_iteration(self):
        """Test getting latest iteration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # No iterations initially
            latest = service.get_latest_iteration("test_exp")
            assert latest is None
            
            # Add iterations with different timestamps
            iter1 = IterationResult(
                iteration_id="iter1",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime(2024, 1, 1)
            )
            
            iter2 = IterationResult(
                iteration_id="iter2",
                experiment_id="test_exp", 
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime(2024, 1, 2)
            )
            
            service.save_iteration(iter1)
            service.save_iteration(iter2)
            
            # Get latest iteration
            latest = service.get_latest_iteration("test_exp")
            assert latest is not None
            assert latest.iteration_id == "iter2"  # More recent

    def test_delete_iteration(self):
        """Test deleting iteration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Add iteration
            iteration = IterationResult(
                iteration_id="iter1",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime.now()
            )
            service.save_iteration(iteration)
            
            # Verify it exists
            assert service.load_iteration("test_exp", "iter1") is not None
            
            # Delete iteration
            deleted = service.delete_iteration("test_exp", "iter1")
            assert deleted is True
            
            # Verify deletion
            assert service.load_iteration("test_exp", "iter1") is None

    def test_delete_nonexistent_iteration(self):
        """Test deleting nonexistent iteration returns False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            deleted = service.delete_iteration("test_exp", "nonexistent")
            assert deleted is False

    def test_save_and_load_reference_registry(self):
        """Test saving and loading reference registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Create test registry
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
            
            # Save registry
            service.save_reference_registry(registry)
            
            # Load registry
            loaded_registry = service.load_reference_registry()
            
            assert len(loaded_registry.corpora) == 1
            assert len(loaded_registry.query_sets) == 1
            assert "test_corpus" in loaded_registry.corpora
            assert "test_queries" in loaded_registry.query_sets

    def test_load_empty_reference_registry(self):
        """Test loading empty reference registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Load empty registry
            registry = service.load_reference_registry()
            assert isinstance(registry, ReferenceRegistry)
            assert len(registry.corpora) == 0
            assert len(registry.query_sets) == 0

    def test_search_iterations_by_experiment(self):
        """Test searching iterations by experiment ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Add iterations for different experiments
            iter1 = IterationResult(
                iteration_id="iter1",
                experiment_id="exp1",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime(2024, 1, 1)
            )
            
            iter2 = IterationResult(
                iteration_id="iter2",
                experiment_id="exp2",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime(2024, 1, 2)
            )
            
            service.save_iteration(iter1)
            service.save_iteration(iter2)
            
            # Search by experiment ID
            results = service.search_iterations(experiment_id="exp1")
            assert len(results) == 1
            assert results[0].iteration_id == "iter1"

    def test_search_iterations_by_date_range(self):
        """Test searching iterations by date range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Add iterations with different timestamps
            iter1 = IterationResult(
                iteration_id="iter1",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime(2024, 1, 1)
            )
            
            iter2 = IterationResult(
                iteration_id="iter2",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime(2024, 1, 15)
            )
            
            iter3 = IterationResult(
                iteration_id="iter3",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime(2024, 2, 1)
            )
            
            service.save_iteration(iter1)
            service.save_iteration(iter2)
            service.save_iteration(iter3)
            
            # Search within date range
            results = service.search_iterations(
                start_date=datetime(2024, 1, 10),
                end_date=datetime(2024, 1, 20)
            )
            assert len(results) == 1
            assert results[0].iteration_id == "iter2"

    def test_get_experiment_summary(self):
        """Test getting experiment summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Add experiment
            experiment = ExperimentConfig(
                experiment_id="test_exp",
                corpus="test_collection",
                queries=["q1"],
                judgments={"q1": {"doc1": 1}},
                primary_metric="ndcg",
                metric_depth=10,
                description="Test experiment"
            )
            service.save_experiment(experiment)
            
            # Add iterations
            iter1 = IterationResult(
                iteration_id="iter1",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime(2024, 1, 1)
            )
            
            iter2 = IterationResult(
                iteration_id="iter2",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime(2024, 1, 2)
            )
            
            service.save_iteration(iter1)
            service.save_iteration(iter2)
            
            # Get summary
            summary = service.get_experiment_summary("test_exp")
            
            assert summary is not None
            assert summary["experiment_id"] == "test_exp"
            assert summary["corpus"] == "test_collection"
            assert summary["primary_metric"] == "ndcg"
            assert summary["metric_depth"] == 10
            assert summary["description"] == "Test experiment"
            assert summary["iteration_count"] == 2

    def test_get_experiment_summary_nonexistent(self):
        """Test getting summary for nonexistent experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            summary = service.get_experiment_summary("nonexistent")
            assert summary is None

    def test_get_metric_history(self):
        """Test getting metric history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Add iterations with metrics
            iter1 = IterationResult(
                iteration_id="iter1",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[
                    MetricResult(metric_name="ndcg", value=0.8, per_query={})
                ],
                timestamp=datetime(2024, 1, 1)
            )
            
            iter2 = IterationResult(
                iteration_id="iter2",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[
                    MetricResult(metric_name="ndcg", value=0.85, per_query={})
                ],
                timestamp=datetime(2024, 1, 2)
            )
            
            service.save_iteration(iter1)
            service.save_iteration(iter2)
            
            # Get metric history
            history = service.get_metric_history("test_exp", "ndcg")
            
            assert len(history) == 2
            assert history[0]["iteration_id"] == "iter1"
            assert history[0]["metric_value"] == 0.8
            assert history[1]["iteration_id"] == "iter2"
            assert history[1]["metric_value"] == 0.85

    def test_vacuum(self):
        """Test vacuum operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Should not raise exception
            service.vacuum()

    def test_get_storage_statistics(self):
        """Test getting storage statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Add some data
            experiment = ExperimentConfig(
                experiment_id="test_exp",
                corpus="test_collection",
                queries=["q1"],
                judgments={"q1": {"doc1": 1}},
                primary_metric="ndcg",
                metric_depth=10
            )
            service.save_experiment(experiment)
            
            iteration = IterationResult(
                iteration_id="iter1",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime.now()
            )
            service.save_iteration(iteration)
            
            # Get statistics
            stats = service.get_storage_statistics()
            
            assert stats["experiment_count"] == 1
            assert stats["iteration_count"] == 1
            assert stats["backend"] == "SQLiteService"


class TestPostgreSQLService:
    """Test cases for the PostgreSQLService class."""

    def test_initialization(self):
        """Test PostgreSQL service initialization."""
        service = PostgreSQLService(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )
        
        assert service.host == "localhost"
        assert service.port == 5432
        assert service.database == "test_db"
        assert service.username == "test_user"
        assert service.password == "test_pass"
        assert "postgresql://" in service.connection_string

    @patch('solr_optimizer.persistence.database_service.psycopg2')
    def test_get_connection_with_psycopg2(self, mock_psycopg2):
        """Test getting PostgreSQL connection with psycopg2 available."""
        mock_connection = Mock()
        mock_psycopg2.connect.return_value = mock_connection
        
        service = PostgreSQLService(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )
        
        conn = service._get_connection()
        
        assert conn is mock_connection
        mock_psycopg2.connect.assert_called_once_with(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass",
            cursor_factory=mock_psycopg2.extras.RealDictCursor
        )

    def test_get_connection_without_psycopg2(self):
        """Test getting PostgreSQL connection without psycopg2 installed."""
        service = PostgreSQLService(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )
        
        with patch('solr_optimizer.persistence.database_service.psycopg2', side_effect=ImportError):
            with pytest.raises(ImportError, match="psycopg2 is required for PostgreSQL support"):
                service._get_connection()


class TestSerializationHelpers:
    """Test cases for serialization helper methods."""

    def test_serialize_deserialize_query_config(self):
        """Test serializing and deserializing QueryConfig."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            
            query_config = QueryConfig(
                query_parser="edismax",
                query_fields={"title": 2.0, "content": 1.0},
                phrase_fields={"title": 3.0},
                boost_queries=["category:electronics^1.5"],
                minimum_match="75%"
            )
            
            # Serialize
            serialized = service._serialize_query_config(query_config)
            assert isinstance(serialized, str)
            
            # Deserialize
            deserialized = service._deserialize_query_config(serialized)
            assert isinstance(deserialized, QueryConfig)
            assert deserialized.query_parser == "edismax"
            assert deserialized.query_fields["title"] == 2.0
            assert deserialized.phrase_fields["title"] == 3.0
            assert deserialized.boost_queries == ["category:electronics^1.5"]
            assert deserialized.minimum_match == "75%"

    def test_serialize_deserialize_query_results(self):
        """Test serializing and deserializing query results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            
            query_results = {
                "laptop": QueryResult(
                    query="laptop",
                    documents=["doc1", "doc2"],
                    scores=[0.9, 0.7],
                    explain_info={"doc1": "explanation"}
                ),
                "smartphone": QueryResult(
                    query="smartphone",
                    documents=["doc3"],
                    scores=[0.8],
                    explain_info={}
                )
            }
            
            # Serialize
            serialized = service._serialize_query_results(query_results)
            assert isinstance(serialized, str)
            
            # Deserialize
            deserialized = service._deserialize_query_results(serialized)
            assert isinstance(deserialized, dict)
            assert len(deserialized) == 2
            assert deserialized["laptop"].query == "laptop"
            assert deserialized["laptop"].documents == ["doc1", "doc2"]
            assert deserialized["smartphone"].query == "smartphone"

    def test_serialize_deserialize_metric_results(self):
        """Test serializing and deserializing metric results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            
            metric_results = [
                MetricResult(
                    metric_name="ndcg",
                    value=0.85,
                    per_query={"laptop": 0.8, "smartphone": 0.9}
                ),
                MetricResult(
                    metric_name="precision",
                    value=0.75,
                    per_query={"laptop": 0.7, "smartphone": 0.8}
                )
            ]
            
            # Serialize
            serialized = service._serialize_metric_results(metric_results)
            assert isinstance(serialized, str)
            
            # Deserialize
            deserialized = service._deserialize_metric_results(serialized)
            assert isinstance(deserialized, list)
            assert len(deserialized) == 2
            assert deserialized[0].metric_name == "ndcg"
            assert deserialized[0].value == 0.85
            assert deserialized[1].metric_name == "precision"
            assert deserialized[1].value == 0.75


class TestExportImport:
    """Test cases for export and import functionality."""

    def test_export_experiment(self):
        """Test exporting experiment data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Add experiment
            experiment = ExperimentConfig(
                experiment_id="test_exp",
                corpus="test_collection",
                queries=["q1"],
                judgments={"q1": {"doc1": 1}},
                primary_metric="ndcg",
                metric_depth=10,
                description="Test experiment"
            )
            service.save_experiment(experiment)
            
            # Add iteration
            iteration = IterationResult(
                iteration_id="iter1",
                experiment_id="test_exp",
                query_config=QueryConfig(),
                query_results={},
                metric_results=[],
                timestamp=datetime.now()
            )
            service.save_iteration(iteration)
            
            # Export experiment
            export_data = service.export_experiment("test_exp")
            
            assert "experiment" in export_data
            assert "iterations" in export_data
            assert "export_timestamp" in export_data
            assert export_data["experiment"]["experiment_id"] == "test_exp"
            assert len(export_data["iterations"]) == 1

    def test_export_nonexistent_experiment(self):
        """Test exporting nonexistent experiment raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            with pytest.raises(ValueError, match="Experiment nonexistent not found"):
                service.export_experiment("nonexistent")

    def test_import_experiment(self):
        """Test importing experiment data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            service = SQLiteService(db_path)
            service.initialize()
            
            # Create export data
            export_data = {
                "experiment": {
                    "experiment_id": "imported_exp",
                    "corpus": "imported_collection",
                    "queries": ["q1"],
                    "judgments": {"q1": {"doc1": 1}},
                    "primary_metric": "ndcg",
                    "metric_depth": 10,
                    "secondary_metrics": [],
                    "description": "Imported experiment"
                },
                "iterations": [
                    {
                        "iteration_id": "iter1",
                        "experiment_id": "imported_exp",
                        "query_config": {},
                        "query_results": {},
                        "metric_results": [],
                        "timestamp": datetime.now().isoformat(),
                        "compared_to": None,
                        "metric_deltas": {},
                        "notes": None
                    }
                ],
                "export_timestamp": datetime.now().isoformat()
            }
            
            # Import experiment
            service.import_experiment(export_data)
            
            # Verify import
            imported_experiment = service.load_experiment("imported_exp")
            assert imported_experiment is not None
            assert imported_experiment.experiment_id == "imported_exp"
            assert imported_experiment.corpus == "imported_collection"
            
            iterations = service.list_iterations("imported_exp")
            assert len(iterations) == 1
            assert "iter1" in iterations
