"""
Solr Optimizer Persistence Layer

This package contains persistence implementations for different storage backends.
"""

from .database_service import DatabaseService, SQLiteService, PostgreSQLService
from .persistence_interface import PersistenceInterface

__all__ = [
    'DatabaseService',
    'SQLiteService', 
    'PostgreSQLService',
    'PersistenceInterface'
]
