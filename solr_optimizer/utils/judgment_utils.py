"""
Judgment Utilities - Utilities for loading and saving relevance judgments.

This module provides utilities for working with relevance judgments in various
formats including CSV, TREC qrels, and JSON.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class JudgmentEntry:
    """
    Single relevance judgment entry.
    
    Attributes:
        query: Query text
        document_id: Document identifier
        relevance: Relevance score
        metadata: Optional metadata about the judgment
    """
    query: str
    document_id: str
    relevance: int
    metadata: Optional[Dict[str, Any]] = None


class JudgmentLoader:
    """
    Utility class for loading relevance judgments from various formats.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_from_csv(self, file_path: Path, 
                     query_col: str = "query",
                     doc_col: str = "document_id", 
                     relevance_col: str = "relevance_score") -> Dict[str, Dict[str, int]]:
        """
        Load judgments from CSV file.
        
        Args:
            file_path: Path to CSV file
            query_col: Name of query column
            doc_col: Name of document ID column
            relevance_col: Name of relevance score column
            
        Returns:
            Dictionary mapping query -> document_id -> relevance_score
            
        Raises:
            ValueError: If required columns are missing or file is invalid
        """
        judgments = {}
        
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Validate columns
                if not all(col in reader.fieldnames for col in [query_col, doc_col, relevance_col]):
                    raise ValueError(f"CSV must contain columns: {query_col}, {doc_col}, {relevance_col}")
                
                for row_num, row in enumerate(reader, 2):  # Start at 2 for header
                    try:
                        query = row[query_col].strip()
                        doc_id = row[doc_col].strip()
                        relevance = int(row[relevance_col])
                        
                        if not query or not doc_id:
                            self.logger.warning(f"Empty query or doc_id in row {row_num}, skipping")
                            continue
                        
                        if query not in judgments:
                            judgments[query] = {}
                        
                        judgments[query][doc_id] = relevance
                        
                    except (ValueError, KeyError) as e:
                        self.logger.warning(f"Error processing row {row_num}: {e}")
                        continue
                        
        except Exception as e:
            raise ValueError(f"Failed to load CSV file {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(judgments)} queries with judgments from {file_path}")
        return judgments
    
    def load_from_trec_qrels(self, file_path: Path) -> Dict[str, Dict[str, int]]:
        """
        Load judgments from TREC qrels format.
        
        Expected format: query_id iteration doc_id relevance
        (iteration is typically 0 and ignored)
        
        Args:
            file_path: Path to qrels file
            
        Returns:
            Dictionary mapping query_id -> document_id -> relevance_score
            
        Raises:
            ValueError: If file format is invalid
        """
        judgments = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 4:
                        self.logger.warning(f"Invalid format in line {line_num}: {line}")
                        continue
                    
                    try:
                        query_id = parts[0]
                        # parts[1] is iteration, typically ignored
                        doc_id = parts[2]
                        relevance = int(parts[3])
                        
                        if query_id not in judgments:
                            judgments[query_id] = {}
                        
                        judgments[query_id][doc_id] = relevance
                        
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Error parsing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            raise ValueError(f"Failed to load TREC qrels file {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(judgments)} queries with judgments from {file_path}")
        return judgments
    
    def load_from_json(self, file_path: Path) -> Dict[str, Dict[str, int]]:
        """
        Load judgments from JSON file.
        
        Expected format: {"query1": {"doc1": 1, "doc2": 0}, ...}
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary mapping query -> document_id -> relevance_score
            
        Raises:
            ValueError: If file format is invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if not isinstance(data, dict):
                raise ValueError("JSON must contain a dictionary at root level")
            
            judgments = {}
            for query, docs in data.items():
                if not isinstance(docs, dict):
                    self.logger.warning(f"Skipping invalid entry for query '{query}': not a dictionary")
                    continue
                
                judgments[query] = {}
                for doc_id, relevance in docs.items():
                    try:
                        judgments[query][doc_id] = int(relevance)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Invalid relevance score for {query}/{doc_id}: {e}")
                        continue
            
            self.logger.info(f"Loaded {len(judgments)} queries with judgments from {file_path}")
            return judgments
            
        except Exception as e:
            raise ValueError(f"Failed to load JSON file {file_path}: {e}")
    
    def load_judgments_with_queries(self, queries_file: Path, qrels_file: Path) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
        """
        Load queries and judgments from separate files.
        
        Args:
            queries_file: File containing queries (one per line or query_id:query_text format)
            qrels_file: File containing judgments in TREC format
            
        Returns:
            Tuple of (queries_list, judgments_dict)
        """
        # Load query mapping
        query_map = {}  # query_id -> query_text
        queries = []
        
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse as "id:text" or just use line number as ID
                if ':' in line and not line.startswith('http'):  # Avoid URLs
                    query_id, query_text = line.split(':', 1)
                    query_id = query_id.strip()
                    query_text = query_text.strip()
                else:
                    query_id = str(line_num)
                    query_text = line
                
                queries.append(query_text)
                query_map[query_id] = query_text
        
        # Load judgments using TREC format
        qrels_judgments = self.load_from_trec_qrels(qrels_file)
        
        # Convert from query_id-based to query_text-based
        judgments = {}
        for query_id, doc_judgments in qrels_judgments.items():
            if query_id in query_map:
                query_text = query_map[query_id]
                judgments[query_text] = doc_judgments
            else:
                self.logger.warning(f"Query ID '{query_id}' in qrels not found in queries file")
        
        return queries, judgments


class JudgmentSaver:
    """
    Utility class for saving relevance judgments to various formats.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_to_csv(self, judgments: Dict[str, Dict[str, int]], file_path: Path,
                   include_metadata: bool = False) -> None:
        """
        Save judgments to CSV file.
        
        Args:
            judgments: Dictionary mapping query -> document_id -> relevance_score
            file_path: Output CSV file path
            include_metadata: Whether to include metadata columns
        """
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['query', 'document_id', 'relevance_score']
            if include_metadata:
                fieldnames.append('metadata')
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for query, docs in judgments.items():
                for doc_id, relevance in docs.items():
                    row = {
                        'query': query,
                        'document_id': doc_id,
                        'relevance_score': relevance
                    }
                    if include_metadata:
                        row['metadata'] = ''  # Could be extended to support actual metadata
                    
                    writer.writerow(row)
        
        self.logger.info(f"Saved judgments to {file_path}")
    
    def save_to_trec_qrels(self, judgments: Dict[str, Dict[str, int]], file_path: Path,
                          query_id_map: Optional[Dict[str, str]] = None) -> None:
        """
        Save judgments to TREC qrels format.
        
        Args:
            judgments: Dictionary mapping query -> document_id -> relevance_score
            file_path: Output qrels file path
            query_id_map: Optional mapping from query text to query ID
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for query_idx, (query, docs) in enumerate(judgments.items()):
                # Use provided query ID or generate one
                if query_id_map and query in query_id_map:
                    query_id = query_id_map[query]
                else:
                    query_id = str(query_idx + 1)
                
                for doc_id, relevance in docs.items():
                    f.write(f"{query_id} 0 {doc_id} {relevance}\n")
        
        self.logger.info(f"Saved judgments to {file_path}")
    
    def save_to_json(self, judgments: Dict[str, Dict[str, int]], file_path: Path,
                    indent: int = 2) -> None:
        """
        Save judgments to JSON file.
        
        Args:
            judgments: Dictionary mapping query -> document_id -> relevance_score
            file_path: Output JSON file path
            indent: JSON indentation level
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(judgments, f, indent=indent, ensure_ascii=False)
        
        self.logger.info(f"Saved judgments to {file_path}")


class JudgmentValidator:
    """
    Utility class for validating relevance judgments.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_judgments(self, judgments: Dict[str, Dict[str, int]], 
                          queries: Optional[List[str]] = None,
                          min_relevance: int = 0,
                          max_relevance: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate relevance judgments and return validation report.
        
        Args:
            judgments: Dictionary mapping query -> document_id -> relevance_score
            queries: Optional list of expected queries
            min_relevance: Minimum allowed relevance score
            max_relevance: Maximum allowed relevance score (None for no limit)
            
        Returns:
            Validation report dictionary
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {
                'total_queries': len(judgments),
                'total_judgments': sum(len(docs) for docs in judgments.values()),
                'relevance_distribution': {},
                'queries_without_judgments': [],
                'empty_queries': []
            }
        }
        
        # Check for empty judgments
        if not judgments:
            report['valid'] = False
            report['errors'].append("No judgments provided")
            return report
        
        # Collect relevance scores for distribution analysis
        all_scores = []
        
        for query, docs in judgments.items():
            if not docs:
                report['warnings'].append(f"Query '{query}' has no judgments")
                report['statistics']['empty_queries'].append(query)
                continue
            
            for doc_id, relevance in docs.items():
                # Validate relevance score type
                if not isinstance(relevance, int):
                    report['valid'] = False
                    report['errors'].append(f"Non-integer relevance score for {query}/{doc_id}: {relevance}")
                    continue
                
                # Validate relevance score range
                if relevance < min_relevance:
                    report['valid'] = False
                    report['errors'].append(f"Relevance score below minimum for {query}/{doc_id}: {relevance} < {min_relevance}")
                
                if max_relevance is not None and relevance > max_relevance:
                    report['valid'] = False
                    report['errors'].append(f"Relevance score above maximum for {query}/{doc_id}: {relevance} > {max_relevance}")
                
                all_scores.append(relevance)
        
        # Calculate relevance distribution
        if all_scores:
            score_counts = {}
            for score in all_scores:
                score_counts[score] = score_counts.get(score, 0) + 1
            report['statistics']['relevance_distribution'] = score_counts
        
        # Check against expected queries
        if queries:
            missing_queries = [q for q in queries if q not in judgments]
            if missing_queries:
                report['warnings'].append(f"Missing judgments for queries: {missing_queries}")
                report['statistics']['queries_without_judgments'] = missing_queries
            
            extra_queries = [q for q in judgments.keys() if q not in queries]
            if extra_queries:
                report['warnings'].append(f"Extra queries with judgments: {extra_queries}")
        
        # Log validation results
        if report['valid']:
            self.logger.info("Judgment validation passed")
        else:
            self.logger.warning(f"Judgment validation failed with {len(report['errors'])} errors")
        
        if report['warnings']:
            self.logger.info(f"Judgment validation completed with {len(report['warnings'])} warnings")
        
        return report
    
    def suggest_judgment_scale(self, judgments: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """
        Analyze judgments and suggest appropriate scale description.
        
        Args:
            judgments: Dictionary mapping query -> document_id -> relevance_score
            
        Returns:
            Dictionary with scale analysis and suggestions
        """
        all_scores = []
        for docs in judgments.values():
            all_scores.extend(docs.values())
        
        if not all_scores:
            return {"suggestion": "No judgments to analyze"}
        
        unique_scores = sorted(set(all_scores))
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        analysis = {
            "min_score": min_score,
            "max_score": max_score,
            "unique_scores": unique_scores,
            "total_judgments": len(all_scores),
            "distribution": {score: all_scores.count(score) for score in unique_scores}
        }
        
        # Suggest scale description based on range
        if unique_scores == [0, 1]:
            analysis["suggested_scale"] = "Binary relevance (0: not relevant, 1: relevant)"
        elif unique_scores == [0, 1, 2]:
            analysis["suggested_scale"] = "3-point scale (0: not relevant, 1: somewhat relevant, 2: highly relevant)"
        elif unique_scores == [0, 1, 2, 3]:
            analysis["suggested_scale"] = "4-point scale (0: not relevant, 1: marginally relevant, 2: relevant, 3: highly relevant)"
        elif len(unique_scores) <= 5:
            analysis["suggested_scale"] = f"{len(unique_scores)}-point scale ({min_score}-{max_score})"
        else:
            analysis["suggested_scale"] = f"Continuous scale ({min_score}-{max_score})"
        
        return analysis


# Convenience functions for common operations
def load_judgments(file_path: Path, format: str = "auto") -> Dict[str, Dict[str, int]]:
    """
    Load judgments from file with automatic format detection.
    
    Args:
        file_path: Path to judgment file
        format: Format hint ("auto", "csv", "trec", "json")
        
    Returns:
        Dictionary mapping query -> document_id -> relevance_score
    """
    loader = JudgmentLoader()
    
    if format == "auto":
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            format = "csv"
        elif suffix == ".json":
            format = "json"
        else:
            format = "trec"  # Default to TREC format
    
    if format == "csv":
        return loader.load_from_csv(file_path)
    elif format == "json":
        return loader.load_from_json(file_path)
    elif format == "trec":
        return loader.load_from_trec_qrels(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_judgments(judgments: Dict[str, Dict[str, int]], file_path: Path, format: str = "auto") -> None:
    """
    Save judgments to file with automatic format detection.
    
    Args:
        judgments: Dictionary mapping query -> document_id -> relevance_score
        file_path: Output file path
        format: Format hint ("auto", "csv", "trec", "json")
    """
    saver = JudgmentSaver()
    
    if format == "auto":
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            format = "csv"
        elif suffix == ".json":
            format = "json"
        else:
            format = "trec"
    
    if format == "csv":
        saver.save_to_csv(judgments, file_path)
    elif format == "json":
        saver.save_to_json(judgments, file_path)
    elif format == "trec":
        saver.save_to_trec_qrels(judgments, file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
