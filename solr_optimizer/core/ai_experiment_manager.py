"""
AI-Enhanced Experiment Manager - Experiment manager with AI-powered optimization.

This module provides an experiment manager that integrates AI agents for
intelligent query optimization recommendations.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any

from solr_optimizer.agents.ai.base_ai_agent import OptimizationContext, AgentRecommendation
from solr_optimizer.agents.ai.orchestrator import QueryOptimizationOrchestrator
from solr_optimizer.agents.comparison.comparison_agent import ComparisonAgent
from solr_optimizer.agents.logging.logging_agent import LoggingAgent
from solr_optimizer.agents.metrics.metrics_agent import MetricsAgent
from solr_optimizer.agents.query.query_tuning_agent import QueryTuningAgent
from solr_optimizer.agents.solr.solr_execution_agent import SolrExecutionAgent
from solr_optimizer.core.default_experiment_manager import DefaultExperimentManager
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.iteration_result import IterationResult
from solr_optimizer.models.query_config import QueryConfig

logger = logging.getLogger(__name__)


class AIExperimentManager(DefaultExperimentManager):
    """
    AI-enhanced experiment manager that uses AI agents for optimization.
    
    This manager extends the default experiment manager with AI-powered
    optimization capabilities through specialized agents and orchestration.
    """

    def __init__(
        self,
        query_tuning_agent: QueryTuningAgent,
        solr_execution_agent: SolrExecutionAgent,
        metrics_agent: MetricsAgent,
        logging_agent: LoggingAgent,
        comparison_agent: ComparisonAgent,
        ai_model: str = "openai:gpt-4",
        ai_config: Optional[Dict[str, Any]] = None,
        enable_ai: bool = True,
    ):
        """
        Initialize the AI-enhanced experiment manager.

        Args:
            query_tuning_agent: Agent for generating query configurations
            solr_execution_agent: Agent for executing Solr queries
            metrics_agent: Agent for calculating relevance metrics
            logging_agent: Agent for logging experiment history
            comparison_agent: Agent for comparing iteration results
            ai_model: AI model to use for optimization (default: gpt-4)
            ai_config: Additional AI configuration parameters
            enable_ai: Whether to enable AI-powered optimization
        """
        super().__init__(
            query_tuning_agent=query_tuning_agent,
            solr_execution_agent=solr_execution_agent,
            metrics_agent=metrics_agent,
            logging_agent=logging_agent,
            comparison_agent=comparison_agent,
        )
        
        self.ai_model = ai_model
        self.ai_config = ai_config or {}
        self.enable_ai = enable_ai
        
        # Initialize AI orchestrator if AI is enabled
        self.orchestrator = None
        if self.enable_ai:
            try:
                self.orchestrator = QueryOptimizationOrchestrator(
                    model=ai_model, **self.ai_config
                )
                logger.info(f"AI orchestrator initialized with model: {ai_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize AI orchestrator: {e}")
                self.enable_ai = False

    def get_ai_recommendation(self, experiment_id: str, constraints: Optional[Dict[str, Any]] = None) -> Optional[AgentRecommendation]:
        """
        Get AI-powered optimization recommendation.

        Args:
            experiment_id: The experiment ID to optimize
            constraints: Optional constraints for optimization

        Returns:
            AI recommendation or None if AI is disabled/failed
        """
        if not self.enable_ai or not self.orchestrator:
            logger.warning("AI optimization is not available")
            return None

        try:
            # Get experiment configuration
            experiment_config = self.logging_agent.get_experiment(experiment_id)
            if not experiment_config:
                raise ValueError(f"Experiment not found: {experiment_id}")

            # Get current state and metrics
            current_state = self.get_current_state(experiment_id)
            current_metrics = {}
            if current_state and current_state.metric_results:
                current_metrics = current_state.metric_results.get('overall', {})

            # Get previous results
            previous_results = self.get_iteration_history(experiment_id)

            # Get schema information from Solr
            schema_info = self._get_schema_info(experiment_config.corpus)

            # Create optimization context
            context = OptimizationContext(
                experiment_config=experiment_config,
                schema_info=schema_info,
                previous_results=previous_results,
                current_metrics=current_metrics,
                optimization_goal=f"Improve {experiment_config.primary_metric}@{experiment_config.metric_depth}",
                constraints=constraints or {}
            )

            # Get coordinated recommendation from AI orchestrator
            recommendation = self.orchestrator.get_coordinated_recommendation(context)
            
            # Log the AI recommendation
            self._log_ai_recommendation(experiment_id, recommendation, context)
            
            return recommendation

        except Exception as e:
            logger.error(f"Failed to get AI recommendation: {e}")
            return None

    def run_ai_optimized_iteration(self, experiment_id: str, constraints: Optional[Dict[str, Any]] = None) -> Optional[IterationResult]:
        """
        Run an AI-optimized iteration.

        Args:
            experiment_id: The experiment ID
            constraints: Optional constraints for optimization

        Returns:
            Iteration result or None if AI optimization failed
        """
        # Get AI recommendation
        recommendation = self.get_ai_recommendation(experiment_id, constraints)
        if not recommendation:
            logger.warning("No AI recommendation available, falling back to basic optimization")
            return None

        # Convert AI recommendation to query configuration
        query_config = self._recommendation_to_query_config(recommendation)
        if not query_config:
            logger.error("Failed to convert AI recommendation to query configuration")
            return None

        # Run iteration with AI-generated configuration
        try:
            result = self.run_iteration(experiment_id, query_config)
            
            # Log that this was AI-generated
            result.metadata = result.metadata or {}
            result.metadata['ai_generated'] = True
            result.metadata['ai_model'] = self.ai_model
            result.metadata['ai_confidence'] = recommendation.confidence
            result.metadata['ai_risk_level'] = recommendation.risk_level
            result.metadata['ai_reasoning'] = recommendation.reasoning
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to run AI-optimized iteration: {e}")
            return None

    def preview_ai_recommendation(self, experiment_id: str, constraints: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Preview AI recommendation without executing it.

        Args:
            experiment_id: The experiment ID
            constraints: Optional constraints for optimization

        Returns:
            Formatted recommendation preview or None if AI is disabled
        """
        recommendation = self.get_ai_recommendation(experiment_id, constraints)
        if not recommendation:
            return None

        return {
            "confidence": recommendation.confidence,
            "risk_level": recommendation.risk_level,
            "priority": recommendation.priority,
            "reasoning": recommendation.reasoning,
            "expected_impact": recommendation.expected_impact,
            "suggested_changes": recommendation.suggested_changes,
            "preview_query_config": self._recommendation_to_query_config(recommendation)
        }

    def _get_schema_info(self, corpus: str) -> Dict[str, Any]:
        """
        Get schema information from Solr.

        Args:
            corpus: The corpus/collection name

        Returns:
            Schema information dictionary
        """
        try:
            # Try to get schema info from Solr execution agent
            if hasattr(self.solr_execution_agent, 'get_schema'):
                return self.solr_execution_agent.get_schema(corpus)
            else:
                # Fallback to basic schema info
                logger.warning("Solr execution agent doesn't support schema retrieval, using fallback")
                return {
                    "fields": {
                        "title": {"type": "text_general"},
                        "content": {"type": "text_general"},
                        "id": {"type": "string"}
                    }
                }
        except Exception as e:
            logger.warning(f"Failed to get schema info: {e}")
            return {"fields": {}}

    def _recommendation_to_query_config(self, recommendation: AgentRecommendation) -> Optional[QueryConfig]:
        """
        Convert AI recommendation to query configuration.

        Args:
            recommendation: AI recommendation

        Returns:
            Query configuration or None if conversion failed
        """
        try:
            suggested_changes = recommendation.suggested_changes
            
            # Generate iteration ID
            iteration_id = f"ai-{uuid.uuid4().hex[:8]}"
            
            # Create query config
            query_config = QueryConfig(
                iteration_id=iteration_id,
                description=f"AI-generated optimization (confidence: {recommendation.confidence:.2f})",
                query_parser="edismax"  # Default parser
            )

            # Handle orchestrated recommendations
            if "prioritized_changes" in suggested_changes:
                # This is an orchestrated recommendation
                prioritized_changes = suggested_changes["prioritized_changes"]
                if prioritized_changes and len(prioritized_changes) > 0:
                    # Apply the highest priority change first
                    primary_change = prioritized_changes[0]
                    self._apply_change_to_config(query_config, primary_change)
            else:
                # This is a direct recommendation from a single agent
                self._apply_change_to_config(query_config, suggested_changes)

            return query_config

        except Exception as e:
            logger.error(f"Failed to convert recommendation to query config: {e}")
            return None

    def _apply_change_to_config(self, query_config: QueryConfig, changes: Dict[str, Any]):
        """
        Apply changes to query configuration.

        Args:
            query_config: Query configuration to modify
            changes: Changes to apply
        """
        # Apply common Solr parameters
        if "qf" in changes:
            query_config.qf = changes["qf"]
        if "pf" in changes:
            query_config.pf = changes["pf"]
        if "mm" in changes:
            query_config.mm = changes["mm"]
        if "tie" in changes:
            query_config.additional_params = query_config.additional_params or {}
            query_config.additional_params["tie"] = str(changes["tie"])
        if "boost" in changes:
            query_config.boost = changes["boost"]
        if "query_parser" in changes:
            query_config.query_parser = changes["query_parser"]

        # Apply additional parameters
        if "additional_params" in changes:
            query_config.additional_params = query_config.additional_params or {}
            query_config.additional_params.update(changes["additional_params"])

    def _log_ai_recommendation(self, experiment_id: str, recommendation: AgentRecommendation, context: OptimizationContext):
        """
        Log AI recommendation for analysis and debugging.

        Args:
            experiment_id: The experiment ID
            recommendation: The AI recommendation
            context: The optimization context
        """
        try:
            log_data = {
                "experiment_id": experiment_id,
                "ai_model": self.ai_model,
                "recommendation": {
                    "confidence": recommendation.confidence,
                    "risk_level": recommendation.risk_level,
                    "priority": recommendation.priority,
                    "reasoning": recommendation.reasoning,
                    "expected_impact": recommendation.expected_impact,
                    "suggested_changes_count": len(recommendation.suggested_changes)
                },
                "context": {
                    "primary_metric": context.experiment_config.primary_metric,
                    "current_metric_value": context.current_metrics.get(context.experiment_config.primary_metric, 0.0),
                    "previous_iterations": len(context.previous_results),
                    "optimization_goal": context.optimization_goal
                }
            }
            
            # Log at info level for important decisions
            logger.info(f"AI Recommendation Generated - "
                       f"Confidence: {recommendation.confidence:.2f}, "
                       f"Risk: {recommendation.risk_level}, "
                       f"Priority: {recommendation.priority}")
            
            # Log detailed data at debug level
            logger.debug(f"AI Recommendation Details: {log_data}")
            
        except Exception as e:
            logger.warning(f"Failed to log AI recommendation: {e}")

    def get_ai_status(self) -> Dict[str, Any]:
        """
        Get AI system status and configuration.

        Returns:
            AI status information
        """
        return {
            "ai_enabled": self.enable_ai,
            "ai_model": self.ai_model,
            "orchestrator_available": self.orchestrator is not None,
            "ai_config": self.ai_config
        }
