"""
Base AI Agent using Pydantic AI for intelligent query optimization.

This module provides the base class for all AI-powered optimization agents
that use Pydantic AI to provide intelligent recommendations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.iteration_result import IterationResult
from solr_optimizer.models.query_config import QueryConfig


@dataclass
class OptimizationContext:
    """Context information for AI agents."""
    experiment_config: ExperimentConfig
    schema_info: Dict[str, Any]
    previous_results: List[IterationResult]
    current_metrics: Dict[str, float]
    optimization_goal: str
    constraints: Dict[str, Any]


class AgentRecommendation(BaseModel):
    """Structured recommendation from an AI agent."""
    confidence: float
    reasoning: str
    suggested_changes: Dict[str, Any]
    expected_impact: str
    risk_level: str  # low, medium, high
    priority: int  # 1-10 scale


class BaseAIAgent(ABC):
    """
    Base class for AI-powered optimization agents using Pydantic AI.
    
    All specialized agents inherit from this class and implement
    domain-specific optimization logic.
    """

    def __init__(self, model: str = "openai:gpt-4", **kwargs):
        """
        Initialize the AI agent.
        
        Args:
            model: The AI model to use (default: gpt-4)
            **kwargs: Additional configuration parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.agent = self._create_agent()
        self.config = kwargs

    @abstractmethod
    def _create_agent(self) -> Agent:
        """
        Create the Pydantic AI agent with domain-specific system prompt.
        
        Returns:
            Configured Pydantic AI agent
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        
        Returns:
            System prompt describing the agent's role and expertise
        """
        pass

    @abstractmethod
    def analyze_and_recommend(self, context: OptimizationContext) -> AgentRecommendation:
        """
        Analyze the optimization context and provide recommendations.
        
        Args:
            context: Current optimization context
            
        Returns:
            Structured recommendation for improvements
        """
        pass

    def _prepare_context_for_ai(self, context: OptimizationContext) -> Dict[str, Any]:
        """
        Prepare context data for AI consumption.
        
        Args:
            context: Optimization context
            
        Returns:
            Simplified context data for AI agent
        """
        return {
            "corpus": context.experiment_config.corpus,
            "primary_metric": context.experiment_config.primary_metric,
            "metric_depth": context.experiment_config.metric_depth,
            "schema_fields": list(context.schema_info.get("fields", {}).keys()),
            "field_types": {
                name: field.get("type", "unknown") 
                for name, field in context.schema_info.get("fields", {}).items()
            },
            "current_metrics": context.current_metrics,
            "optimization_goal": context.optimization_goal,
            "previous_iterations": len(context.previous_results),
            "best_score": max(
                [r.metrics.get("overall", {}).get(context.experiment_config.primary_metric, 0.0) 
                 for r in context.previous_results], 
                default=0.0
            ),
            "constraints": context.constraints
        }

    async def get_recommendation_async(self, context: OptimizationContext) -> AgentRecommendation:
        """
        Get recommendation asynchronously using Pydantic AI.
        
        Args:
            context: Optimization context
            
        Returns:
            AI-generated recommendation
        """
        try:
            ai_context = self._prepare_context_for_ai(context)
            
            # Run the AI agent with the prepared context
            result = await self.agent.run(
                user_prompt=self._create_user_prompt(context),
                message_history=[]
            )
            
            return self._parse_ai_response(result.data, context)
            
        except Exception as e:
            self.logger.error(f"Error getting AI recommendation: {e}")
            return self._get_fallback_recommendation(context)

    def get_recommendation(self, context: OptimizationContext) -> AgentRecommendation:
        """
        Get recommendation synchronously (wrapper for async method).
        
        Args:
            context: Optimization context
            
        Returns:
            AI-generated recommendation
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.get_recommendation_async(context))

    @abstractmethod
    def _create_user_prompt(self, context: OptimizationContext) -> str:
        """
        Create a user prompt for the AI agent based on the context.
        
        Args:
            context: Optimization context
            
        Returns:
            User prompt for the AI agent
        """
        pass

    @abstractmethod
    def _parse_ai_response(self, ai_response: Any, context: OptimizationContext) -> AgentRecommendation:
        """
        Parse the AI response into a structured recommendation.
        
        Args:
            ai_response: Raw response from AI agent
            context: Optimization context
            
        Returns:
            Structured recommendation
        """
        pass

    @abstractmethod
    def _get_fallback_recommendation(self, context: OptimizationContext) -> AgentRecommendation:
        """
        Provide a fallback recommendation when AI fails.
        
        Args:
            context: Optimization context
            
        Returns:
            Fallback recommendation
        """
        pass

    def validate_recommendation(self, recommendation: AgentRecommendation) -> bool:
        """
        Validate that a recommendation is safe and reasonable.
        
        Args:
            recommendation: The recommendation to validate
            
        Returns:
            True if recommendation is valid
        """
        # Basic validation
        if not (0.0 <= recommendation.confidence <= 1.0):
            return False
            
        if not (1 <= recommendation.priority <= 10):
            return False
            
        if recommendation.risk_level not in ["low", "medium", "high"]:
            return False
            
        return True

    def log_recommendation(self, recommendation: AgentRecommendation, context: OptimizationContext):
        """
        Log the recommendation for debugging and analysis.
        
        Args:
            recommendation: The recommendation to log
            context: The optimization context
        """
        self.logger.info(
            f"AI Recommendation - Confidence: {recommendation.confidence:.2f}, "
            f"Priority: {recommendation.priority}, Risk: {recommendation.risk_level}"
        )
        self.logger.debug(f"Reasoning: {recommendation.reasoning}")
        self.logger.debug(f"Suggested changes: {recommendation.suggested_changes}")
