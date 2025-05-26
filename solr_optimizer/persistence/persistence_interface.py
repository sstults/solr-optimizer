"""
Persistence Interface - Abstract interface for storage backends.

This module defines the abstract interface that all persistence implementations
must follow, providing a common API for different storage backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.iteration_result import IterationResult
from solr_optimizer.models.corpus_config import CorpusReference, QuerySet, ReferenceRegistry


class PersistenceInterface(ABC):
    """
    Abstract interface for persistence implementations.
    
    This interface defines the common operations that all storage backends
    must support for experiments, iterations, and reference data.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the storage backend.
        
        This should create any necessary tables, indexes, or file structures.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Close the storage backend and clean up resources.
        """
        pass
    
    # Experiment operations
    @abstractmethod
    def save_experiment(self, experiment: ExperimentConfig) -> None:
        """
        Save an experiment configuration.
        
        Args:
            experiment: The experiment configuration to save
        """
        pass
    
    @abstractmethod
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """
        Load an experiment configuration.
        
        Args:
            experiment_id: The experiment identifier
            
        Returns:
            The experiment configuration or None if not found
        """
        pass
    
    @abstractmethod
    def list_experiments(self) -> List[str]:
        """
        List all experiment IDs.
        
        Returns:
            List of experiment identifiers
        """
        pass
    
    @abstractmethod
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment and all its iterations.
        
        Args:
            experiment_id: The experiment identifier
            
        Returns:
            True if experiment was deleted, False if not found
        """
        pass
    
    # Iteration operations
    @abstractmethod
    def save_iteration(self, iteration: IterationResult) -> None:
        """
        Save an iteration result.
        
        Args:
            iteration: The iteration result to save
        """
        pass
    
    @abstractmethod
    def load_iteration(self, experiment_id: str, iteration_id: str) -> Optional[IterationResult]:
        """
        Load an iteration result.
        
        Args:
            experiment_id: The experiment identifier
            iteration_id: The iteration identifier
            
        Returns:
            The iteration result or None if not found
        """
        pass
    
    @abstractmethod
    def list_iterations(self, experiment_id: str) -> List[str]:
        """
        List all iteration IDs for an experiment.
        
        Args:
            experiment_id: The experiment identifier
            
        Returns:
            List of iteration identifiers
        """
        pass
    
    @abstractmethod
    def get_latest_iteration(self, experiment_id: str) -> Optional[IterationResult]:
        """
        Get the latest iteration for an experiment.
        
        Args:
            experiment_id: The experiment identifier
            
        Returns:
            The latest iteration result or None if no iterations exist
        """
        pass
    
    @abstractmethod
    def delete_iteration(self, experiment_id: str, iteration_id: str) -> bool:
        """
        Delete an iteration result.
        
        Args:
            experiment_id: The experiment identifier
            iteration_id: The iteration identifier
            
        Returns:
            True if iteration was deleted, False if not found
        """
        pass
    
    # Reference registry operations
    @abstractmethod
    def save_reference_registry(self, registry: ReferenceRegistry) -> None:
        """
        Save the reference registry.
        
        Args:
            registry: The reference registry to save
        """
        pass
    
    @abstractmethod
    def load_reference_registry(self) -> ReferenceRegistry:
        """
        Load the reference registry.
        
        Returns:
            The reference registry (empty if none exists)
        """
        pass
    
    # Query and search operations
    @abstractmethod
    def search_iterations(self, experiment_id: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         metric_name: Optional[str] = None,
                         min_metric_value: Optional[float] = None) -> List[IterationResult]:
        """
        Search for iterations based on criteria.
        
        Args:
            experiment_id: Optional experiment ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            metric_name: Optional metric name filter
            min_metric_value: Optional minimum metric value filter
            
        Returns:
            List of matching iteration results
        """
        pass
    
    @abstractmethod
    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of an experiment including iteration statistics.
        
        Args:
            experiment_id: The experiment identifier
            
        Returns:
            Dictionary with experiment summary or None if not found
        """
        pass
    
    @abstractmethod
    def get_metric_history(self, experiment_id: str, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get the history of a specific metric for an experiment.
        
        Args:
            experiment_id: The experiment identifier
            metric_name: The name of the metric
            
        Returns:
            List of dictionaries with iteration_id, timestamp, and metric_value
        """
        pass
    
    # Backup and restore operations
    @abstractmethod
    def export_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Export an experiment and all its iterations.
        
        Args:
            experiment_id: The experiment identifier
            
        Returns:
            Dictionary containing the complete experiment data
        """
        pass
    
    @abstractmethod
    def import_experiment(self, experiment_data: Dict[str, Any]) -> None:
        """
        Import an experiment and all its iterations.
        
        Args:
            experiment_data: Dictionary containing the complete experiment data
        """
        pass
    
    # Maintenance operations
    @abstractmethod
    def vacuum(self) -> None:
        """
        Perform maintenance operations to optimize storage.
        
        This might include compacting databases, cleaning up temporary files, etc.
        """
        pass
    
    @abstractmethod
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about storage usage.
        
        Returns:
            Dictionary with storage statistics
        """
        pass
