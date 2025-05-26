"""
Solr Optimizer Models

This package contains data models for experiments, configurations, and results.
"""

from .experiment_config import ExperimentConfig
from .query_config import QueryConfig
from .iteration_result import IterationResult, QueryResult, MetricResult
from .corpus_config import CorpusReference, QuerySet, ReferenceRegistry

# Re-export AI models from base_ai_agent for convenience
from ..agents.ai.base_ai_agent import OptimizationContext, AgentRecommendation

__all__ = [
    'ExperimentConfig',
    'QueryConfig', 
    'IterationResult',
    'QueryResult',
    'MetricResult',
    'CorpusReference',
    'QuerySet',
    'ReferenceRegistry',
    'OptimizationContext',
    'AgentRecommendation'
]
