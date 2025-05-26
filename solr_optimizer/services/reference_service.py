"""
Reference Service - Service for managing corpus and query set references.

This module provides a high-level service for managing named corpus references
and query sets, including persistence and retrieval operations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import csv
import json

from solr_optimizer.models.corpus_config import CorpusReference, QuerySet, ReferenceRegistry
from solr_optimizer.models.experiment_config import ExperimentConfig


class ReferenceService:
    """
    Service for managing corpus and query set references.
    
    This service provides high-level operations for managing named references
    to corpora and query sets, including loading from various file formats
    and persistence.
    """
    
    def __init__(self, registry_file: Optional[Path] = None):
        """
        Initialize the reference service.
        
        Args:
            registry_file: Optional path to registry file. If None, uses default location.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry_file = registry_file or Path("references.json")
        self.registry = self._load_registry()
    
    def _load_registry(self) -> ReferenceRegistry:
        """
        Load the reference registry from file.
        
        Returns:
            ReferenceRegistry instance
        """
        if self.registry_file.exists():
            try:
                return ReferenceRegistry.load_from_file(self.registry_file)
            except Exception as e:
                self.logger.warning(f"Failed to load registry from {self.registry_file}: {e}")
                self.logger.info("Creating new registry")
        
        return ReferenceRegistry()
    
    def save_registry(self) -> None:
        """Save the current registry to file."""
        try:
            self.registry.save_to_file(self.registry_file)
            self.logger.info(f"Registry saved to {self.registry_file}")
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
            raise
    
    def add_corpus(self, name: str, collection: str, solr_url: str, 
                   description: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> CorpusReference:
        """
        Add a new corpus reference.
        
        Args:
            name: Unique name for the corpus
            collection: Solr collection name
            solr_url: Base URL for Solr instance
            description: Optional description
            metadata: Optional metadata dictionary
            
        Returns:
            The created CorpusReference
            
        Raises:
            ValueError: If a corpus with the same name already exists
        """
        corpus = CorpusReference(
            name=name,
            collection=collection,
            solr_url=solr_url,
            description=description,
            metadata=metadata or {}
        )
        
        self.registry.add_corpus(corpus)
        self.save_registry()
        self.logger.info(f"Added corpus reference: {name}")
        return corpus
    
    def add_query_set_from_data(self, name: str, queries: List[str], 
                               judgments: Dict[str, Dict[str, int]],
                               description: Optional[str] = None,
                               judgment_scale: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> QuerySet:
        """
        Add a new query set from provided data.
        
        Args:
            name: Unique name for the query set
            queries: List of query strings
            judgments: Judgments dictionary
            description: Optional description
            judgment_scale: Optional description of judgment scale
            metadata: Optional metadata dictionary
            
        Returns:
            The created QuerySet
            
        Raises:
            ValueError: If a query set with the same name already exists
        """
        query_set = QuerySet(
            name=name,
            queries=queries,
            judgments=judgments,
            description=description,
            judgment_scale=judgment_scale,
            metadata=metadata or {}
        )
        
        self.registry.add_query_set(query_set)
        self.save_registry()
        self.logger.info(f"Added query set: {name}")
        return query_set
    
    def add_query_set_from_csv(self, name: str, csv_file: Path, 
                              description: Optional[str] = None,
                              judgment_scale: Optional[str] = None) -> QuerySet:
        """
        Add a query set from a CSV file.
        
        Expected CSV format:
        query,document_id,relevance_score
        
        Args:
            name: Unique name for the query set
            csv_file: Path to CSV file
            description: Optional description
            judgment_scale: Optional description of judgment scale
            
        Returns:
            The created QuerySet
            
        Raises:
            ValueError: If file format is invalid or query set name exists
        """
        queries = []
        judgments = {}
        
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Validate required columns
            required_cols = ['query', 'document_id', 'relevance_score']
            if not all(col in reader.fieldnames for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            for row in reader:
                query = row['query'].strip()
                doc_id = row['document_id'].strip()
                relevance = int(row['relevance_score'])
                
                if query not in queries:
                    queries.append(query)
                
                if query not in judgments:
                    judgments[query] = {}
                
                judgments[query][doc_id] = relevance
        
        if not queries:
            raise ValueError("No valid queries found in CSV file")
        
        return self.add_query_set_from_data(
            name=name,
            queries=queries,
            judgments=judgments,
            description=description,
            judgment_scale=judgment_scale,
            metadata={"source_file": str(csv_file)}
        )
    
    def add_query_set_from_trec(self, name: str, queries_file: Path, qrels_file: Path,
                               description: Optional[str] = None,
                               judgment_scale: Optional[str] = None) -> QuerySet:
        """
        Add a query set from TREC format files.
        
        Args:
            name: Unique name for the query set
            queries_file: Path to queries file (one query per line, with optional ID)
            qrels_file: Path to qrels file (TREC format: query_id 0 doc_id relevance)
            description: Optional description
            judgment_scale: Optional description of judgment scale
            
        Returns:
            The created QuerySet
        """
        # Load queries
        queries = []
        query_map = {}  # id -> query text
        
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Try to parse as "id:query" or just use line number as ID
                if ':' in line:
                    query_id, query_text = line.split(':', 1)
                    query_id = query_id.strip()
                    query_text = query_text.strip()
                else:
                    query_id = str(line_num)
                    query_text = line
                
                queries.append(query_text)
                query_map[query_id] = query_text
        
        # Load judgments
        judgments = {}
        
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 4:
                    continue
                
                query_id, _, doc_id, relevance = parts
                relevance = int(relevance)
                
                if query_id in query_map:
                    query_text = query_map[query_id]
                    
                    if query_text not in judgments:
                        judgments[query_text] = {}
                    
                    judgments[query_text][doc_id] = relevance
        
        if not queries:
            raise ValueError("No valid queries found in queries file")
        
        return self.add_query_set_from_data(
            name=name,
            queries=queries,
            judgments=judgments,
            description=description,
            judgment_scale=judgment_scale,
            metadata={
                "source_queries": str(queries_file),
                "source_qrels": str(qrels_file)
            }
        )
    
    def get_corpus(self, name: str) -> CorpusReference:
        """
        Get a corpus reference by name.
        
        Args:
            name: Name of the corpus
            
        Returns:
            CorpusReference
            
        Raises:
            KeyError: If corpus doesn't exist
        """
        return self.registry.get_corpus(name)
    
    def get_query_set(self, name: str) -> QuerySet:
        """
        Get a query set by name.
        
        Args:
            name: Name of the query set
            
        Returns:
            QuerySet
            
        Raises:
            KeyError: If query set doesn't exist
        """
        return self.registry.get_query_set(name)
    
    def list_corpora(self) -> List[str]:
        """
        List all available corpus names.
        
        Returns:
            List of corpus names
        """
        return self.registry.list_corpora()
    
    def list_query_sets(self) -> List[str]:
        """
        List all available query set names.
        
        Returns:
            List of query set names
        """
        return self.registry.list_query_sets()
    
    def remove_corpus(self, name: str) -> None:
        """
        Remove a corpus reference.
        
        Args:
            name: Name of the corpus to remove
        """
        self.registry.remove_corpus(name)
        self.save_registry()
        self.logger.info(f"Removed corpus reference: {name}")
    
    def remove_query_set(self, name: str) -> None:
        """
        Remove a query set.
        
        Args:
            name: Name of the query set to remove
        """
        self.registry.remove_query_set(name)
        self.save_registry()
        self.logger.info(f"Removed query set: {name}")
    
    def create_experiment_config(self, experiment_id: str, corpus_name: str, 
                                query_set_name: str, primary_metric: str,
                                metric_depth: int, secondary_metrics: Optional[List[str]] = None,
                                description: Optional[str] = None) -> ExperimentConfig:
        """
        Create an experiment configuration using named references.
        
        Args:
            experiment_id: Unique experiment identifier
            corpus_name: Name of registered corpus
            query_set_name: Name of registered query set
            primary_metric: Primary optimization metric
            metric_depth: Metric depth (e.g., 10 for nDCG@10)
            secondary_metrics: Optional list of secondary metrics
            description: Optional experiment description
            
        Returns:
            ExperimentConfig instance
            
        Raises:
            KeyError: If corpus or query set doesn't exist
        """
        corpus = self.get_corpus(corpus_name)
        query_set = self.get_query_set(query_set_name)
        
        return ExperimentConfig(
            experiment_id=experiment_id,
            corpus=corpus.collection,  # Use collection name for Solr
            queries=query_set.queries,
            judgments=query_set.judgments,
            primary_metric=primary_metric,
            metric_depth=metric_depth,
            secondary_metrics=secondary_metrics or [],
            description=description
        )
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current registry.
        
        Returns:
            Dictionary with registry statistics and info
        """
        return {
            "corpora_count": len(self.registry.corpora),
            "query_sets_count": len(self.registry.query_sets),
            "corpora": [
                {
                    "name": name,
                    "collection": corpus.collection,
                    "solr_url": corpus.solr_url,
                    "description": corpus.description
                }
                for name, corpus in self.registry.corpora.items()
            ],
            "query_sets": [
                {
                    "name": name,
                    "query_count": len(qs.queries),
                    "judgment_scale": qs.judgment_scale,
                    "description": qs.description
                }
                for name, qs in self.registry.query_sets.items()
            ]
        }
    
    def export_query_set_to_csv(self, query_set_name: str, output_file: Path) -> None:
        """
        Export a query set to CSV format.
        
        Args:
            query_set_name: Name of the query set to export
            output_file: Path for output CSV file
            
        Raises:
            KeyError: If query set doesn't exist
        """
        query_set = self.get_query_set(query_set_name)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['query', 'document_id', 'relevance_score'])
            
            for query in query_set.queries:
                for doc_id, relevance in query_set.judgments[query].items():
                    writer.writerow([query, doc_id, relevance])
        
        self.logger.info(f"Exported query set '{query_set_name}' to {output_file}")
