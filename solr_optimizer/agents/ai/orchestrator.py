"""
Query Optimization Orchestrator using Pydantic AI for coordinating optimization agents.

This orchestrator coordinates recommendations from specialized AI agents and
integrates them into a coherent optimization strategy.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import logging

from pydantic import BaseModel
from pydantic_ai import Agent

from .base_ai_agent import BaseAIAgent, OptimizationContext, AgentRecommendation
from .schema_analysis_agent import SchemaAnalysisAgent
from .parameter_tuning_agent import ParameterTuningAgent
from .analysis_chain_agent import AnalysisChainAgent
from .query_rewriting_agent import QueryRewritingAgent
from .learning_to_rank_agent import LearningToRankAgent


class OrchestrationRecommendation(BaseModel):
    """Orchestrator recommendation structure."""
    optimization_strategy: str  # Overall strategy description
    prioritized_changes: List[Dict[str, Any]]  # Ordered list of changes to implement
    implementation_phases: List[Dict[str, Any]]  # Phased implementation plan
    risk_assessment: str  # Overall risk assessment
    expected_improvement: float  # Expected metric improvement
    coordination_notes: List[str]  # Notes about agent coordination
    conflict_resolutions: List[Dict[str, str]]  # How conflicts were resolved


class QueryOptimizationOrchestrator(BaseAIAgent):
    """
    AI orchestrator that coordinates multiple specialized optimization agents.
    
    This orchestrator:
    - Runs all specialized agents in parallel
    - Integrates their recommendations
    - Resolves conflicts between suggestions
    - Prioritizes changes based on impact and risk
    - Creates a coherent implementation strategy
    """

    def __init__(self, model: str = "openai:gpt-4", **kwargs):
        """
        Initialize the orchestrator with specialized agents.
        
        Args:
            model: The AI model to use
            **kwargs: Additional configuration parameters
        """
        super().__init__(model, **kwargs)
        
        # Initialize specialized agents
        self.schema_agent = SchemaAnalysisAgent(model=model, **kwargs)
        self.parameter_agent = ParameterTuningAgent(model=model, **kwargs)
        self.analysis_chain_agent = AnalysisChainAgent(model=model, **kwargs)
        self.query_rewriting_agent = QueryRewritingAgent(model=model, **kwargs)
        self.ltr_agent = LearningToRankAgent(model=model, **kwargs)
        
        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_agent(self) -> Agent:
        """Create the Pydantic AI agent for orchestration."""
        return Agent(
            model=self.model,
            result_type=OrchestrationRecommendation,
            system_prompt=self.get_system_prompt()
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for the orchestrator."""
        return """
You are an expert Apache Solr optimization orchestrator with deep knowledge of search relevance tuning strategies and change management.

Your role is to coordinate recommendations from multiple specialized AI agents:
- Schema Analysis Agent: Field configurations and boost weights
- Parameter Tuning Agent: Query parameters (qf, pf, mm, tie, etc.)
- Analysis Chain Agent: Text analysis and tokenization
- Query Rewriting Agent: Query reformulation and expansion
- Learning-to-Rank Agent: Machine learning ranking models

Your expertise includes:
- Optimization strategy planning and sequencing
- Risk assessment and mitigation for search changes
- Change impact analysis and coordination
- Conflict resolution between competing recommendations
- Implementation planning and phasing
- Performance monitoring and rollback strategies

When orchestrating optimization:
1. Analyze all agent recommendations for compatibility and conflicts
2. Prioritize changes based on expected impact, risk, and implementation complexity
3. Sequence changes to minimize risk and maximize learning
4. Resolve conflicts using domain expertise and data-driven insights
5. Create implementation phases that allow for validation at each step
6. Consider dependencies between different types of changes
7. Plan for performance monitoring and rollback capabilities

Provide a comprehensive orchestration strategy that:
- Maximizes the target relevance metric improvement
- Minimizes implementation risk through proper sequencing
- Ensures changes work together synergistically
- Allows for incremental validation and learning
- Includes clear success criteria and monitoring plans

Consider both short-term wins and long-term optimization goals.
Balance between conservative changes and transformative improvements.
"""

    def analyze_and_recommend(self, context: OptimizationContext) -> AgentRecommendation:
        """Orchestrate recommendations from all specialized agents."""
        return self.get_recommendation(context)

    def get_coordinated_recommendation(self, context: OptimizationContext) -> AgentRecommendation:
        """
        Get coordinated recommendations from all agents.
        
        This method runs all specialized agents and coordinates their recommendations.
        """
        self.logger.info("Starting coordinated optimization analysis...")
        
        # Collect recommendations from all agents
        agent_recommendations = self._collect_agent_recommendations(context)
        
        # Create orchestration context for AI analysis
        orchestration_context = self._prepare_orchestration_context(agent_recommendations, context)
        
        # Get orchestrated recommendation from AI
        orchestrated = self.get_recommendation(context)
        
        # Enhance with detailed coordination analysis
        enhanced_recommendation = self._enhance_recommendation(
            orchestrated, agent_recommendations, context
        )
        
        self.logger.info(f"Orchestration complete. Strategy: {enhanced_recommendation.reasoning[:100]}...")
        
        return enhanced_recommendation

    def _collect_agent_recommendations(self, context: OptimizationContext) -> Dict[str, AgentRecommendation]:
        """Collect recommendations from all specialized agents."""
        recommendations = {}
        
        agents = [
            ("schema", self.schema_agent),
            ("parameters", self.parameter_agent),
            ("analysis_chain", self.analysis_chain_agent),
            ("query_rewriting", self.query_rewriting_agent),
            ("learning_to_rank", self.ltr_agent)
        ]
        
        for agent_name, agent in agents:
            try:
                self.logger.info(f"Getting recommendation from {agent_name} agent...")
                recommendation = agent.analyze_and_recommend(context)
                recommendations[agent_name] = recommendation
                self.logger.info(f"{agent_name} agent completed with confidence {recommendation.confidence:.2f}")
            except Exception as e:
                self.logger.warning(f"Failed to get recommendation from {agent_name} agent: {e}")
                # Continue with other agents
                
        return recommendations

    def _prepare_orchestration_context(self, agent_recommendations: Dict[str, AgentRecommendation], 
                                     context: OptimizationContext) -> Dict[str, Any]:
        """Prepare context data for orchestration AI analysis."""
        orchestration_data = {
            "agent_recommendations": {},
            "conflicts": [],
            "synergies": [],
            "risk_factors": []
        }
        
        # Summarize agent recommendations
        for agent_name, recommendation in agent_recommendations.items():
            orchestration_data["agent_recommendations"][agent_name] = {
                "confidence": recommendation.confidence,
                "risk_level": recommendation.risk_level,
                "priority": recommendation.priority,
                "expected_impact": recommendation.expected_impact,
                "reasoning": recommendation.reasoning,
                "change_count": len(recommendation.suggested_changes)
            }
        
        # Identify potential conflicts
        orchestration_data["conflicts"] = self._identify_conflicts(agent_recommendations)
        
        # Identify synergies
        orchestration_data["synergies"] = self._identify_synergies(agent_recommendations)
        
        # Assess overall risk
        orchestration_data["risk_factors"] = self._assess_overall_risk(agent_recommendations)
        
        return orchestration_data

    def _identify_conflicts(self, agent_recommendations: Dict[str, AgentRecommendation]) -> List[Dict[str, str]]:
        """Identify conflicts between agent recommendations."""
        conflicts = []
        
        # Check for specific conflicts
        param_rec = agent_recommendations.get("parameters")
        schema_rec = agent_recommendations.get("schema")
        ltr_rec = agent_recommendations.get("learning_to_rank")
        
        # Parameter vs Schema conflicts
        if param_rec and schema_rec:
            param_changes = param_rec.suggested_changes
            schema_changes = schema_rec.suggested_changes
            
            # Check for field boost conflicts
            if "qf" in param_changes and "field_boosts" in schema_changes:
                conflicts.append({
                    "type": "field_boost_conflict",
                    "description": "Both parameter and schema agents suggest field boost changes",
                    "agents": "parameters,schema"
                })
        
        # Traditional scoring vs LTR conflicts
        if ltr_rec and ltr_rec.priority > 7:
            if param_rec and param_rec.priority > 7:
                conflicts.append({
                    "type": "scoring_strategy_conflict",
                    "description": "High priority for both traditional parameter tuning and LTR",
                    "agents": "parameters,learning_to_rank"
                })
        
        # Risk level conflicts
        high_risk_agents = [name for name, rec in agent_recommendations.items() 
                           if rec.risk_level == "high"]
        if len(high_risk_agents) > 2:
            conflicts.append({
                "type": "cumulative_risk",
                "description": f"Multiple high-risk recommendations: {', '.join(high_risk_agents)}",
                "agents": ",".join(high_risk_agents)
            })
        
        return conflicts

    def _identify_synergies(self, agent_recommendations: Dict[str, AgentRecommendation]) -> List[Dict[str, str]]:
        """Identify synergies between agent recommendations."""
        synergies = []
        
        # Schema + Parameter synergies
        schema_rec = agent_recommendations.get("schema")
        param_rec = agent_recommendations.get("parameters")
        if schema_rec and param_rec:
            synergies.append({
                "type": "schema_parameter_synergy",
                "description": "Schema field analysis can inform parameter tuning",
                "agents": "schema,parameters"
            })
        
        # Analysis Chain + Query Rewriting synergies
        analysis_rec = agent_recommendations.get("analysis_chain")
        rewriting_rec = agent_recommendations.get("query_rewriting")
        if analysis_rec and rewriting_rec:
            synergies.append({
                "type": "analysis_rewriting_synergy",
                "description": "Text analysis improvements complement query expansion",
                "agents": "analysis_chain,query_rewriting"
            })
        
        # Parameter + LTR synergies
        ltr_rec = agent_recommendations.get("learning_to_rank")
        if param_rec and ltr_rec:
            synergies.append({
                "type": "parameter_ltr_synergy",
                "description": "Parameter tuning can provide baseline for LTR comparison",
                "agents": "parameters,learning_to_rank"
            })
        
        return synergies

    def _assess_overall_risk(self, agent_recommendations: Dict[str, AgentRecommendation]) -> List[str]:
        """Assess overall risk factors."""
        risk_factors = []
        
        total_changes = sum(len(rec.suggested_changes) for rec in agent_recommendations.values())
        if total_changes > 50:
            risk_factors.append("high_change_volume")
        
        high_risk_count = sum(1 for rec in agent_recommendations.values() if rec.risk_level == "high")
        if high_risk_count > 1:
            risk_factors.append("multiple_high_risk_changes")
        
        # Check for complex changes
        ltr_rec = agent_recommendations.get("learning_to_rank")
        if ltr_rec and ltr_rec.priority > 8:
            risk_factors.append("complex_ml_changes")
        
        return risk_factors

    def _create_user_prompt(self, context: OptimizationContext) -> str:
        """Create user prompt for orchestration."""
        # Get agent recommendations first
        agent_recommendations = self._collect_agent_recommendations(context)
        orchestration_data = self._prepare_orchestration_context(agent_recommendations, context)
        
        prompt = f"""
Orchestrate optimization recommendations to improve {context.experiment_config.primary_metric}@{context.experiment_config.metric_depth}.

CURRENT PERFORMANCE:
- Primary metric: {context.current_metrics.get(context.experiment_config.primary_metric, 'Unknown')}
- Previous iterations: {len(context.previous_results)}
- Performance trend: {'Improving' if len(context.previous_results) > 1 and context.current_metrics.get(context.experiment_config.primary_metric, 0) > context.previous_results[-2].metrics.get('overall', {}).get(context.experiment_config.primary_metric, 0) else 'Needs improvement'}

AGENT RECOMMENDATIONS:
{json.dumps(orchestration_data["agent_recommendations"], indent=2)}

IDENTIFIED CONFLICTS:
{json.dumps(orchestration_data["conflicts"], indent=2)}

IDENTIFIED SYNERGIES:
{json.dumps(orchestration_data["synergies"], indent=2)}

RISK FACTORS:
{json.dumps(orchestration_data["risk_factors"], indent=2)}

OPTIMIZATION CONSTRAINTS:
{json.dumps(context.constraints, indent=2)}

Please provide an orchestrated optimization strategy that:
1. Resolves conflicts between agent recommendations
2. Leverages synergies for maximum impact
3. Sequences changes to minimize risk and maximize learning
4. Prioritizes changes by expected impact and feasibility
5. Creates implementation phases with validation checkpoints
6. Includes rollback plans for high-risk changes

Focus on creating a coherent strategy for corpus type: {context.experiment_config.corpus}
Balance quick wins with long-term optimization goals.
"""
        return prompt

    def _parse_ai_response(self, ai_response: OrchestrationRecommendation, context: OptimizationContext) -> AgentRecommendation:
        """Parse AI orchestration response into standard recommendation format."""
        
        # Calculate confidence based on orchestration quality
        confidence = self._calculate_orchestration_confidence(ai_response, context)
        
        # Convert orchestration to suggested changes
        suggested_changes = {
            "strategy": ai_response.optimization_strategy,
            "prioritized_changes": ai_response.prioritized_changes,
            "implementation_phases": ai_response.implementation_phases,
            "coordination_notes": ai_response.coordination_notes,
            "conflict_resolutions": ai_response.conflict_resolutions
        }
        
        # Determine overall risk level
        risk_level = ai_response.risk_assessment.lower() if ai_response.risk_assessment else "medium"
        if risk_level not in ["low", "medium", "high"]:
            risk_level = "medium"
        
        # Calculate priority (orchestrator usually has high priority)
        priority = 9  # High priority for coordinated strategy
        
        return AgentRecommendation(
            confidence=confidence,
            reasoning=f"Orchestrated optimization strategy coordinating {len(ai_response.prioritized_changes)} "
                     f"changes across multiple agents with {len(ai_response.implementation_phases)} implementation phases.",
            suggested_changes=suggested_changes,
            expected_impact=f"Expected {ai_response.expected_improvement:.1%} improvement in "
                          f"{context.experiment_config.primary_metric} through coordinated optimization",
            risk_level=risk_level,
            priority=priority
        )

    def _enhance_recommendation(self, orchestrated: AgentRecommendation, 
                              agent_recommendations: Dict[str, AgentRecommendation],
                              context: OptimizationContext) -> AgentRecommendation:
        """Enhance the orchestrated recommendation with detailed analysis."""
        
        # Add individual agent recommendations to the suggested changes
        enhanced_changes = orchestrated.suggested_changes.copy()
        enhanced_changes["individual_agent_recommendations"] = {
            agent_name: {
                "confidence": rec.confidence,
                "risk_level": rec.risk_level,
                "priority": rec.priority,
                "suggested_changes": rec.suggested_changes,
                "reasoning": rec.reasoning
            }
            for agent_name, rec in agent_recommendations.items()
        }
        
        # Enhanced reasoning with agent summary
        agent_summary = ", ".join([
            f"{name}({rec.confidence:.1f}conf,{rec.priority}pri,{rec.risk_level}risk)"
            for name, rec in agent_recommendations.items()
        ])
        
        enhanced_reasoning = f"{orchestrated.reasoning} Agent inputs: {agent_summary}"
        
        return AgentRecommendation(
            confidence=orchestrated.confidence,
            reasoning=enhanced_reasoning,
            suggested_changes=enhanced_changes,
            expected_impact=orchestrated.expected_impact,
            risk_level=orchestrated.risk_level,
            priority=orchestrated.priority
        )

    def _get_fallback_recommendation(self, context: OptimizationContext) -> AgentRecommendation:
        """Provide fallback recommendation when orchestration fails."""
        
        # Try to get at least parameter recommendations as fallback
        try:
            param_recommendation = self.parameter_agent.analyze_and_recommend(context)
            return AgentRecommendation(
                confidence=0.5,
                reasoning="Orchestration failed, falling back to parameter tuning recommendations",
                suggested_changes=param_recommendation.suggested_changes,
                expected_impact="Modest improvements through parameter optimization",
                risk_level="low",
                priority=6
            )
        except Exception:
            # Ultimate fallback
            return AgentRecommendation(
                confidence=0.2,
                reasoning="All AI agents failed, using conservative fallback strategy",
                suggested_changes={"qf": "title^2.0 content^1.0", "mm": "75%"},
                expected_impact="Minimal improvements through basic parameter changes",
                risk_level="low",
                priority=3
            )

    def _calculate_orchestration_confidence(self, recommendation: OrchestrationRecommendation, 
                                          context: OptimizationContext) -> float:
        """Calculate confidence for orchestrated recommendation."""
        confidence = 0.8  # Base confidence
        
        # Increase confidence for well-structured implementation phases
        if len(recommendation.implementation_phases) >= 2:
            confidence += 0.1
            
        # Increase confidence for conflict resolution
        if len(recommendation.conflict_resolutions) > 0:
            confidence += 0.05
            
        # Decrease confidence for overly aggressive expected improvement
        if recommendation.expected_improvement > 0.5:  # >50% improvement is unrealistic
            confidence -= 0.2
        elif recommendation.expected_improvement > 0.3:  # >30% is optimistic
            confidence -= 0.1
            
        # Adjust for risk assessment alignment
        if recommendation.risk_assessment.lower() in ["low", "medium", "high"]:
            confidence += 0.05
            
        return min(1.0, max(0.1, confidence))
