"""
Logging Agent - Interface for storing experiment history and configuration.

This module defines the LoggingAgent interface which is responsible for
recording experiment iterations, configurations, and results.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.query_config import QueryConfig
from solr_optimizer.models.iteration_result import IterationResult


class LoggingAgent(ABC):
    """
    Agent responsible for storing experiment history and configuration.
    """

    @abstractmethod
    def log_iteration(self, iteration_result: IterationResult) -> bool:
        """
        Log an iteration result to storage.
        
        Args:
            iteration_result: The iteration result to log
            
        Returns:
            True if logging was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_iteration(self, experiment_id: str, iteration_id: str) -> Optional[IterationResult]:
        """
        Retrieve a specific iteration result from storage.
        
        Args:
            experiment_id: The experiment ID
            iteration_id: The iteration ID
            
        Returns:
            The iteration result, or None if not found
        """
        pass
    
    @abstractmethod
    def list_iterations(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        List all iterations for an experiment.
        
        Args:
            experiment_id: The experiment ID
            
        Returns:
            List of iteration summary dictionaries
        """
        pass
    
    @abstractmethod
    def save_experiment(self, experiment: ExperimentConfig) -> bool:
        """
        Save an experiment configuration to storage.
        
        Args:
            experiment: The experiment configuration
            
        Returns:
            True if saving was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """
        Retrieve an experiment configuration from storage.
        
        Args:
            experiment_id: The experiment ID
            
        Returns:
            The experiment configuration, or None if not found
        """
        pass
    
    @abstractmethod
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.
        
        Returns:
            List of experiment summary dictionaries
        """
        pass
    
    @abstractmethod
    def tag_iteration(self, experiment_id: str, iteration_id: str, tag: str) -> bool:
        """
        Tag an iteration with a user-friendly name or category.
        
        Args:
            experiment_id: The experiment ID
            iteration_id: The iteration ID
            tag: The tag to apply
            
        Returns:
            True if tagging was successful, False otherwise
        """
        pass
