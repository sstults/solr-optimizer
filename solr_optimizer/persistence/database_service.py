"""
Database Service - Database-backed persistence implementations.

This module provides database implementations of the persistence interface
using SQLite and PostgreSQL.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import pickle

from .persistence_interface import PersistenceInterface
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.iteration_result import IterationResult, QueryResult, MetricResult
from solr_optimizer.models.query_config import QueryConfig
from solr_optimizer.models.corpus_config import CorpusReference, QuerySet, ReferenceRegistry


class DatabaseService(PersistenceInterface):
    """
    Base database service implementation.
    
    This class provides common database operations and can be subclassed
    for specific database implementations.
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize the database service.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        self.logger = logging.getLogger(self.__class__.__name__)
        self._connection = None
    
    def _get_connection(self):
        """Get database connection (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _get_connection")
    
    def initialize(self) -> None:
        """Initialize database schema."""
        self._create_tables()
        self._create_indexes()
    
    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def _create_tables(self) -> None:
        """Create database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                corpus TEXT NOT NULL,
                primary_metric TEXT NOT NULL,
                metric_depth INTEGER NOT NULL,
                secondary_metrics TEXT,
                description TEXT,
                queries TEXT NOT NULL,
                judgments TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Iterations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                query_config TEXT NOT NULL,
                query_results TEXT NOT NULL,
                metric_results TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                compared_to TEXT,
                metric_deltas TEXT,
                notes TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
                UNIQUE(experiment_id, iteration_id)
            )
        """)
        
        # Reference registry table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reference_registry (
                id INTEGER PRIMARY KEY,
                registry_data TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
    
    def _create_indexes(self) -> None:
        """Create database indexes for performance."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_iterations_experiment ON iterations(experiment_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_iterations_timestamp ON iterations(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at)")
        
        conn.commit()
    
    def save_experiment(self, experiment: ExperimentConfig) -> None:
        """Save experiment configuration."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO experiments 
            (experiment_id, corpus, primary_metric, metric_depth, secondary_metrics, 
             description, queries, judgments, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment.experiment_id,
            experiment.corpus,
            experiment.primary_metric,
            experiment.metric_depth,
            json.dumps(experiment.secondary_metrics),
            experiment.description,
            json.dumps(experiment.queries),
            json.dumps(experiment.judgments),
            datetime.now()
        ))
        
        conn.commit()
        self.logger.info(f"Saved experiment: {experiment.experiment_id}")
    
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Load experiment configuration."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT experiment_id, corpus, primary_metric, metric_depth, 
                   secondary_metrics, description, queries, judgments
            FROM experiments WHERE experiment_id = ?
        """, (experiment_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return ExperimentConfig(
            experiment_id=row[0],
            corpus=row[1],
            primary_metric=row[2],
            metric_depth=row[3],
            secondary_metrics=json.loads(row[4]) if row[4] else [],
            description=row[5],
            queries=json.loads(row[6]),
            judgments=json.loads(row[7])
        )
    
    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT experiment_id FROM experiments ORDER BY created_at DESC")
        return [row[0] for row in cursor.fetchall()]
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment and all iterations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Delete iterations first
        cursor.execute("DELETE FROM iterations WHERE experiment_id = ?", (experiment_id,))
        
        # Delete experiment
        cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        
        if deleted:
            self.logger.info(f"Deleted experiment: {experiment_id}")
        
        return deleted
    
    def save_iteration(self, iteration: IterationResult) -> None:
        """Save iteration result."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Serialize complex objects
        query_config_data = self._serialize_query_config(iteration.query_config)
        query_results_data = self._serialize_query_results(iteration.query_results)
        metric_results_data = self._serialize_metric_results(iteration.metric_results)
        
        cursor.execute("""
            INSERT OR REPLACE INTO iterations 
            (iteration_id, experiment_id, query_config, query_results, metric_results,
             timestamp, compared_to, metric_deltas, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            iteration.iteration_id,
            iteration.experiment_id,
            query_config_data,
            query_results_data,
            metric_results_data,
            iteration.timestamp,
            iteration.compared_to,
            json.dumps(iteration.metric_deltas),
            iteration.notes
        ))
        
        conn.commit()
        self.logger.info(f"Saved iteration: {iteration.iteration_id}")
    
    def load_iteration(self, experiment_id: str, iteration_id: str) -> Optional[IterationResult]:
        """Load iteration result."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT iteration_id, experiment_id, query_config, query_results, 
                   metric_results, timestamp, compared_to, metric_deltas, notes
            FROM iterations WHERE experiment_id = ? AND iteration_id = ?
        """, (experiment_id, iteration_id))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return IterationResult(
            iteration_id=row[0],
            experiment_id=row[1],
            query_config=self._deserialize_query_config(row[2]),
            query_results=self._deserialize_query_results(row[3]),
            metric_results=self._deserialize_metric_results(row[4]),
            timestamp=row[5] if isinstance(row[5], datetime) else datetime.fromisoformat(row[5]),
            compared_to=row[6],
            metric_deltas=json.loads(row[7]) if row[7] else {},
            notes=row[8]
        )
    
    def list_iterations(self, experiment_id: str) -> List[str]:
        """List all iteration IDs for an experiment."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT iteration_id FROM iterations 
            WHERE experiment_id = ? ORDER BY timestamp DESC
        """, (experiment_id,))
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_latest_iteration(self, experiment_id: str) -> Optional[IterationResult]:
        """Get the latest iteration for an experiment."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT iteration_id FROM iterations 
            WHERE experiment_id = ? ORDER BY timestamp DESC LIMIT 1
        """, (experiment_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return self.load_iteration(experiment_id, row[0])
    
    def delete_iteration(self, experiment_id: str, iteration_id: str) -> bool:
        """Delete an iteration result."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM iterations 
            WHERE experiment_id = ? AND iteration_id = ?
        """, (experiment_id, iteration_id))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        
        if deleted:
            self.logger.info(f"Deleted iteration: {iteration_id}")
        
        return deleted
    
    def save_reference_registry(self, registry: ReferenceRegistry) -> None:
        """Save the reference registry."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        registry_data = json.dumps(registry.to_dict())
        
        cursor.execute("""
            INSERT OR REPLACE INTO reference_registry (id, registry_data, updated_at)
            VALUES (1, ?, ?)
        """, (registry_data, datetime.now()))
        
        conn.commit()
    
    def load_reference_registry(self) -> ReferenceRegistry:
        """Load the reference registry."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT registry_data FROM reference_registry WHERE id = 1")
        row = cursor.fetchone()
        
        if not row:
            return ReferenceRegistry()
        
        registry_data = json.loads(row[0])
        return ReferenceRegistry.from_dict(registry_data)
    
    def search_iterations(self, experiment_id: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         metric_name: Optional[str] = None,
                         min_metric_value: Optional[float] = None) -> List[IterationResult]:
        """Search for iterations based on criteria."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT experiment_id, iteration_id FROM iterations WHERE 1=1"
        params = []
        
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        results = []
        
        for row in cursor.fetchall():
            iteration = self.load_iteration(row[0], row[1])
            if iteration:
                # Apply metric filters if specified
                if metric_name or min_metric_value:
                    metric = iteration.get_metric_by_name(metric_name) if metric_name else iteration.get_primary_metric()
                    if metric and (min_metric_value is None or metric.value >= min_metric_value):
                        results.append(iteration)
                else:
                    results.append(iteration)
        
        return results
    
    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment summary with statistics."""
        experiment = self.load_experiment(experiment_id)
        if not experiment:
            return None
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM iterations WHERE experiment_id = ?
        """, (experiment_id,))
        
        stats = cursor.fetchone()
        
        return {
            "experiment_id": experiment_id,
            "corpus": experiment.corpus,
            "primary_metric": experiment.primary_metric,
            "metric_depth": experiment.metric_depth,
            "description": experiment.description,
            "iteration_count": stats[0] if stats else 0,
            "first_iteration": stats[1] if stats else None,
            "last_iteration": stats[2] if stats else None
        }
    
    def get_metric_history(self, experiment_id: str, metric_name: str) -> List[Dict[str, Any]]:
        """Get metric history for an experiment."""
        iterations = self.search_iterations(experiment_id=experiment_id)
        history = []
        
        for iteration in iterations:
            metric = iteration.get_metric_by_name(metric_name)
            if metric:
                history.append({
                    "iteration_id": iteration.iteration_id,
                    "timestamp": iteration.timestamp,
                    "metric_value": metric.value
                })
        
        return sorted(history, key=lambda x: x["timestamp"])
    
    def export_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Export complete experiment data."""
        experiment = self.load_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        iterations = []
        for iteration_id in self.list_iterations(experiment_id):
            iteration = self.load_iteration(experiment_id, iteration_id)
            if iteration:
                iterations.append(iteration.to_dict())
        
        return {
            "experiment": experiment.__dict__,
            "iterations": iterations,
            "export_timestamp": datetime.now().isoformat()
        }
    
    def import_experiment(self, experiment_data: Dict[str, Any]) -> None:
        """Import complete experiment data."""
        # Import experiment
        exp_data = experiment_data["experiment"]
        experiment = ExperimentConfig(**exp_data)
        self.save_experiment(experiment)
        
        # Import iterations
        for iter_data in experiment_data.get("iterations", []):
            iteration = self._dict_to_iteration_result(iter_data)
            self.save_iteration(iteration)
    
    def vacuum(self) -> None:
        """Perform database maintenance."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("VACUUM")
        conn.commit()
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM experiments")
        experiment_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM iterations")
        iteration_count = cursor.fetchone()[0]
        
        return {
            "experiment_count": experiment_count,
            "iteration_count": iteration_count,
            "backend": self.__class__.__name__
        }
    
    # Serialization helpers
    def _serialize_query_config(self, config: QueryConfig) -> str:
        """Serialize QueryConfig to JSON string."""
        return json.dumps(config.__dict__)
    
    def _deserialize_query_config(self, data: str) -> QueryConfig:
        """Deserialize QueryConfig from JSON string."""
        config_dict = json.loads(data)
        return QueryConfig(**config_dict)
    
    def _serialize_query_results(self, results: Dict[str, QueryResult]) -> str:
        """Serialize query results to JSON string."""
        serializable = {}
        for query, result in results.items():
            serializable[query] = {
                "query": result.query,
                "documents": result.documents,
                "scores": result.scores,
                "explain_info": result.explain_info
            }
        return json.dumps(serializable)
    
    def _deserialize_query_results(self, data: str) -> Dict[str, QueryResult]:
        """Deserialize query results from JSON string."""
        data_dict = json.loads(data)
        results = {}
        for query, result_data in data_dict.items():
            results[query] = QueryResult(**result_data)
        return results
    
    def _serialize_metric_results(self, results: List[MetricResult]) -> str:
        """Serialize metric results to JSON string."""
        serializable = []
        for result in results:
            serializable.append({
                "metric_name": result.metric_name,
                "value": result.value,
                "per_query": result.per_query
            })
        return json.dumps(serializable)
    
    def _deserialize_metric_results(self, data: str) -> List[MetricResult]:
        """Deserialize metric results from JSON string."""
        data_list = json.loads(data)
        results = []
        for result_data in data_list:
            results.append(MetricResult(**result_data))
        return results
    
    def _dict_to_iteration_result(self, data: Dict[str, Any]) -> IterationResult:
        """Convert dictionary to IterationResult."""
        # This would need to handle the complex nested structure
        # For now, simplified implementation
        return IterationResult(**data)


class SQLiteService(DatabaseService):
    """SQLite implementation of database service."""
    
    def __init__(self, db_path: Union[str, Path] = "solr_optimizer.db"):
        """
        Initialize SQLite service.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        super().__init__(str(self.db_path))
    
    def _get_connection(self):
        """Get SQLite connection."""
        if not self._connection:
            self._connection = sqlite3.connect(
                self.db_path, 
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection


class PostgreSQLService(DatabaseService):
    """PostgreSQL implementation of database service."""
    
    def __init__(self, host: str, port: int, database: str, 
                 username: str, password: str):
        """
        Initialize PostgreSQL service.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
        """
        connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        super().__init__(connection_string)
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
    
    def _get_connection(self):
        """Get PostgreSQL connection."""
        if not self._connection:
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                
                self._connection = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.username,
                    password=self.password,
                    cursor_factory=RealDictCursor
                )
            except ImportError:
                raise ImportError("psycopg2 is required for PostgreSQL support. Install with: pip install psycopg2-binary")
        
        return self._connection
    
    def _create_tables(self) -> None:
        """Create PostgreSQL tables with appropriate types."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                corpus TEXT NOT NULL,
                primary_metric TEXT NOT NULL,
                metric_depth INTEGER NOT NULL,
                secondary_metrics JSONB,
                description TEXT,
                queries JSONB NOT NULL,
                judgments JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Iterations table  
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS iterations (
                id SERIAL PRIMARY KEY,
                iteration_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                query_config JSONB NOT NULL,
                query_results JSONB NOT NULL,
                metric_results JSONB NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                compared_to TEXT,
                metric_deltas JSONB,
                notes TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
                UNIQUE(experiment_id, iteration_id)
            )
        """)
        
        # Reference registry table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reference_registry (
                id INTEGER PRIMARY KEY DEFAULT 1,
                registry_data JSONB NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK (id = 1)
            )
        """)
        
        conn.commit()
