"""
Query Configuration - Model class for Solr query parameters.

This module defines the QueryConfig class that represents the configuration
parameters for Solr queries in an optimization experiment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class QueryConfig:
    """
    Configuration parameters for Solr queries.
    
    Attributes:
        iteration_id: Optional identifier for this configuration iteration
        query_parser: Query parser to use (e.g., 'edismax', 'lucene')
        query_fields: Dictionary of fields to query with their boosts
        phrase_fields: Dictionary of phrase boost fields with their boosts
        boost_queries: List of boost query expressions
        boost_functions: List of boost function expressions
        minimum_match: Minimum should match parameter
        tie_breaker: Tie breaker value for field scores
        additional_params: Any additional Solr parameters not covered by specific fields
    """
    
    query_parser: str = "edismax"
    query_fields: Dict[str, float] = field(default_factory=dict)  # field -> boost value
    phrase_fields: Dict[str, float] = field(default_factory=dict)  # field -> boost value
    boost_queries: List[str] = field(default_factory=list)
    boost_functions: List[str] = field(default_factory=list)
    minimum_match: Optional[str] = None
    tie_breaker: float = 0.0
    iteration_id: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_solr_params(self) -> Dict[str, Any]:
        """
        Convert the configuration to Solr query parameters.
        
        Returns:
            Dictionary of parameters to pass to Solr
        """
        params = {
            "defType": self.query_parser,
            "tie": self.tie_breaker
        }
        
        # Query fields (qf)
        if self.query_fields:
            params["qf"] = " ".join([f"{field}^{boost}" for field, boost in self.query_fields.items()])
        
        # Phrase fields (pf)
        if self.phrase_fields:
            params["pf"] = " ".join([f"{field}^{boost}" for field, boost in self.phrase_fields.items()])
        
        # Boost queries (bq)
        if self.boost_queries:
            params["bq"] = self.boost_queries
        
        # Boost functions (bf)
        if self.boost_functions:
            params["bf"] = self.boost_functions
        
        # Minimum should match (mm)
        if self.minimum_match:
            params["mm"] = self.minimum_match
        
        # Add any additional parameters
        params.update(self.additional_params)
        
        return params
