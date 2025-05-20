"""
Query Tuning Agent - Interface for generating and modifying query configurations.

This module defines the QueryTuningAgent interface which is responsible for
suggesting modifications to query configurations to improve relevance metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.query_config import QueryConfig
from solr_optimizer.models.iteration_result import IterationResult


class QueryTuningAgent(ABC):
    """
    Agent responsible for generating and modifying query configurations to improve relevance metrics.
    """

    @abstractmethod
    def analyze_schema(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Solr schema to identify potentially useful fields and configurations.
        
        Args:
            schema_info: Schema information from Solr
            
        Returns:
            Analysis results with recommendations for query configuration
        """
        pass
    
    @abstractmethod
    def generate_initial_config(self, experiment_config: ExperimentConfig, 
                               schema_info: Dict[str, Any]) -> QueryConfig:
        """
        Generate an initial query configuration based on experiment settings and schema.
        
        Args:
            experiment_config: The experiment configuration
            schema_info: Schema information from Solr
            
        Returns:
            An initial query configuration to test
        """
        pass
    
    @abstractmethod
    def suggest_next_config(self, previous_result: IterationResult, 
                           schema_info: Dict[str, Any]) -> QueryConfig:
        """
        Suggest a new query configuration based on the results of a previous iteration.
        
        Args:
            previous_result: The results of the previous iteration
            schema_info: Schema information from Solr
            
        Returns:
            A new query configuration to test
        """
        pass
    
    @abstractmethod
    def adjust_parameters(self, result: IterationResult, target_metric: str, 
                         direction: str) -> QueryConfig:
        """
        Adjust specific parameters to improve a target metric.
        
        Args:
            result: The results of the previous iteration
            target_metric: The metric to optimize (e.g., 'ndcg@10')
            direction: Either 'increase' or 'decrease'
            
        Returns:
            A new query configuration with adjusted parameters
        """
        pass
