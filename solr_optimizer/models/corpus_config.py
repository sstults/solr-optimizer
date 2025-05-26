"""
Corpus Configuration - Model classes for corpus and query set references.

This module defines classes for managing named corpus references and query sets
that can be reused across experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class CorpusReference:
    """
    Reference to a named corpus (Solr collection).
    
    Attributes:
        name: Unique name for this corpus reference
        collection: Name of the Solr collection/core
        solr_url: Base URL for the Solr instance
        description: Optional description of the corpus
        metadata: Additional metadata about the corpus
    """
    name: str
    collection: str
    solr_url: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuerySet:
    """
    Named set of queries with associated judgments.
    
    Attributes:
        name: Unique name for this query set
        queries: List of query strings
        judgments: Nested dictionary mapping query -> document_id -> relevance_score
        description: Optional description of the query set
        judgment_scale: Description of the relevance judgment scale (e.g., "0-3: not relevant to highly relevant")
        metadata: Additional metadata about the query set
    """
    name: str
    queries: List[str]
    judgments: Dict[str, Dict[str, int]]
    description: Optional[str] = None
    judgment_scale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the query set configuration."""
        # Ensure all queries have judgments
        missing_queries = [q for q in self.queries if q not in self.judgments]
        if missing_queries:
            raise ValueError(f"Missing judgments for queries: {', '.join(missing_queries)}")


@dataclass
class ReferenceRegistry:
    """
    Registry for managing corpus and query set references.
    
    Attributes:
        corpora: Dictionary of corpus name to CorpusReference
        query_sets: Dictionary of query set name to QuerySet
    """
    corpora: Dict[str, CorpusReference] = field(default_factory=dict)
    query_sets: Dict[str, QuerySet] = field(default_factory=dict)
    
    def add_corpus(self, corpus: CorpusReference) -> None:
        """
        Add a corpus reference to the registry.
        
        Args:
            corpus: The corpus reference to add
            
        Raises:
            ValueError: If a corpus with the same name already exists
        """
        if corpus.name in self.corpora:
            raise ValueError(f"Corpus '{corpus.name}' already exists")
        self.corpora[corpus.name] = corpus
    
    def add_query_set(self, query_set: QuerySet) -> None:
        """
        Add a query set to the registry.
        
        Args:
            query_set: The query set to add
            
        Raises:
            ValueError: If a query set with the same name already exists
        """
        if query_set.name in self.query_sets:
            raise ValueError(f"Query set '{query_set.name}' already exists")
        self.query_sets[query_set.name] = query_set
    
    def get_corpus(self, name: str) -> CorpusReference:
        """
        Get a corpus reference by name.
        
        Args:
            name: Name of the corpus reference
            
        Returns:
            The corpus reference
            
        Raises:
            KeyError: If the corpus reference doesn't exist
        """
        if name not in self.corpora:
            raise KeyError(f"Corpus '{name}' not found")
        return self.corpora[name]
    
    def get_query_set(self, name: str) -> QuerySet:
        """
        Get a query set by name.
        
        Args:
            name: Name of the query set
            
        Returns:
            The query set
            
        Raises:
            KeyError: If the query set doesn't exist
        """
        if name not in self.query_sets:
            raise KeyError(f"Query set '{name}' not found")
        return self.query_sets[name]
    
    def list_corpora(self) -> List[str]:
        """
        List all available corpus names.
        
        Returns:
            List of corpus names
        """
        return list(self.corpora.keys())
    
    def list_query_sets(self) -> List[str]:
        """
        List all available query set names.
        
        Returns:
            List of query set names
        """
        return list(self.query_sets.keys())
    
    def remove_corpus(self, name: str) -> None:
        """
        Remove a corpus reference.
        
        Args:
            name: Name of the corpus to remove
            
        Raises:
            KeyError: If the corpus doesn't exist
        """
        if name not in self.corpora:
            raise KeyError(f"Corpus '{name}' not found")
        del self.corpora[name]
    
    def remove_query_set(self, name: str) -> None:
        """
        Remove a query set.
        
        Args:
            name: Name of the query set to remove
            
        Raises:
            KeyError: If the query set doesn't exist
        """
        if name not in self.query_sets:
            raise KeyError(f"Query set '{name}' not found")
        del self.query_sets[name]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the registry to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the registry
        """
        return {
            'corpora': {
                name: {
                    'name': corpus.name,
                    'collection': corpus.collection,
                    'solr_url': corpus.solr_url,
                    'description': corpus.description,
                    'metadata': corpus.metadata
                }
                for name, corpus in self.corpora.items()
            },
            'query_sets': {
                name: {
                    'name': qs.name,
                    'queries': qs.queries,
                    'judgments': qs.judgments,
                    'description': qs.description,
                    'judgment_scale': qs.judgment_scale,
                    'metadata': qs.metadata
                }
                for name, qs in self.query_sets.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReferenceRegistry':
        """
        Create a registry from a dictionary.
        
        Args:
            data: Dictionary representation of the registry
            
        Returns:
            ReferenceRegistry instance
        """
        registry = cls()
        
        # Load corpora
        for name, corpus_data in data.get('corpora', {}).items():
            corpus = CorpusReference(**corpus_data)
            registry.corpora[name] = corpus
        
        # Load query sets
        for name, qs_data in data.get('query_sets', {}).items():
            query_set = QuerySet(**qs_data)
            registry.query_sets[name] = query_set
        
        return registry
    
    def save_to_file(self, file_path: Path) -> None:
        """
        Save the registry to a JSON file.
        
        Args:
            file_path: Path to save the registry file
        """
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'ReferenceRegistry':
        """
        Load the registry from a JSON file.
        
        Args:
            file_path: Path to the registry file
            
        Returns:
            ReferenceRegistry instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
