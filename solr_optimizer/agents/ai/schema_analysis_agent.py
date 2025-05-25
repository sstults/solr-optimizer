"""
Schema Analysis Agent using Pydantic AI for schema optimization.

This agent analyzes Solr document schemas and field configurations to recommend
optimal field boost weights and suggest schema modifications for better searchability.
"""

from typing import Any, Dict, List
import json

from pydantic import BaseModel
from pydantic_ai import Agent

from .base_ai_agent import BaseAIAgent, OptimizationContext, AgentRecommendation


class SchemaRecommendation(BaseModel):
    """Schema-specific recommendation structure."""
    field_boosts: Dict[str, float]
    field_analysis_suggestions: Dict[str, str]
    new_field_suggestions: List[Dict[str, Any]]
    copy_field_suggestions: List[Dict[str, str]]
    dynamic_field_suggestions: List[str]


class SchemaAnalysisAgent(BaseAIAgent):
    """
    AI agent specialized in analyzing Solr schemas and recommending optimizations.
    
    This agent focuses on:
    - Field boost weight optimization
    - Analysis chain improvements
    - Copy field recommendations
    - Dynamic field suggestions
    - Field type optimizations
    """

    def _create_agent(self) -> Agent:
        """Create the Pydantic AI agent for schema analysis."""
        return Agent(
            model=self.model,
            result_type=SchemaRecommendation,
            system_prompt=self.get_system_prompt()
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for the schema analysis agent."""
        return """
You are an expert Apache Solr schema analyst with deep knowledge of search relevance optimization.

Your expertise includes:
- Solr field types and their optimal configurations
- Text analysis chains (tokenizers, filters, analyzers)
- Field boosting strategies for different content types
- Copy field patterns for improved searchability
- Dynamic field configurations
- Schema design patterns for different domains

When analyzing a schema, consider:
1. Field types and their suitability for search and relevance
2. Analysis chains and how they affect tokenization and matching
3. Field boost weights based on content importance and user query patterns
4. Copy field opportunities to improve search coverage
5. Missing fields that could enhance relevance
6. Dynamic field patterns for flexible schema evolution

Provide concrete, actionable recommendations with clear reasoning based on search best practices.
Focus on changes that will most likely improve the specified relevance metric.

Always consider the trade-offs between precision and recall, and explain your reasoning.
Prioritize recommendations that are safe to implement and have proven effectiveness.
"""

    def analyze_and_recommend(self, context: OptimizationContext) -> AgentRecommendation:
        """Analyze schema and provide recommendations."""
        return self.get_recommendation(context)

    def _create_user_prompt(self, context: OptimizationContext) -> str:
        """Create user prompt for schema analysis."""
        schema_fields = context.schema_info.get("fields", {})
        
        # Analyze current query performance by field
        field_performance = self._analyze_field_performance(context)
        
        prompt = f"""
Analyze this Solr schema and recommend optimizations to improve {context.experiment_config.primary_metric}@{context.experiment_config.metric_depth}.

CURRENT SCHEMA:
{json.dumps(schema_fields, indent=2)}

CURRENT PERFORMANCE:
- Primary metric: {context.current_metrics.get(context.experiment_config.primary_metric, 'Unknown')}
- Previous iterations: {len(context.previous_results)}
- Performance trend: {'Improving' if len(context.previous_results) > 1 and context.current_metrics.get(context.experiment_config.primary_metric, 0) > context.previous_results[-2].metrics.get('overall', {}).get(context.experiment_config.primary_metric, 0) else 'Needs improvement'}

FIELD PERFORMANCE ANALYSIS:
{json.dumps(field_performance, indent=2)}

OPTIMIZATION CONSTRAINTS:
{json.dumps(context.constraints, indent=2)}

Please provide specific recommendations for:
1. Field boost weights (qf parameter values)
2. Analysis chain improvements
3. New fields that should be added
4. Copy field configurations
5. Dynamic field patterns

Focus on changes that will improve search relevance for the corpus type: {context.experiment_config.corpus}
"""
        return prompt

    def _parse_ai_response(self, ai_response: SchemaRecommendation, context: OptimizationContext) -> AgentRecommendation:
        """Parse AI response into standard recommendation format."""
        
        # Calculate confidence based on the specificity and safety of recommendations
        confidence = self._calculate_confidence(ai_response, context)
        
        # Convert schema recommendation to suggested changes
        suggested_changes = {
            "field_boosts": ai_response.field_boosts,
            "analysis_suggestions": ai_response.field_analysis_suggestions,
            "new_fields": ai_response.new_field_suggestions,
            "copy_fields": ai_response.copy_field_suggestions,
            "dynamic_fields": ai_response.dynamic_field_suggestions
        }
        
        # Determine risk level based on the scope of changes
        risk_level = self._assess_risk_level(ai_response)
        
        # Calculate priority based on potential impact
        priority = self._calculate_priority(ai_response, context)
        
        return AgentRecommendation(
            confidence=confidence,
            reasoning=f"Schema analysis identified {len(ai_response.field_boosts)} field boost opportunities, "
                     f"{len(ai_response.field_analysis_suggestions)} analysis improvements, and "
                     f"{len(ai_response.new_field_suggestions)} new field suggestions.",
            suggested_changes=suggested_changes,
            expected_impact=f"Expected to improve {context.experiment_config.primary_metric} through "
                          f"better field weighting and enhanced analysis chains",
            risk_level=risk_level,
            priority=priority
        )

    def _get_fallback_recommendation(self, context: OptimizationContext) -> AgentRecommendation:
        """Provide fallback recommendation when AI fails."""
        # Simple heuristic-based recommendations
        schema_fields = context.schema_info.get("fields", {})
        
        # Basic field boost suggestions based on field names
        field_boosts = {}
        for field_name, field_info in schema_fields.items():
            if "title" in field_name.lower():
                field_boosts[field_name] = 2.0
            elif "description" in field_name.lower() or "content" in field_name.lower():
                field_boosts[field_name] = 1.0
            elif field_info.get("type") == "text_general":
                field_boosts[field_name] = 0.5
        
        return AgentRecommendation(
            confidence=0.3,  # Low confidence for fallback
            reasoning="AI analysis failed, using heuristic-based field boost recommendations",
            suggested_changes={"field_boosts": field_boosts},
            expected_impact="Basic field boosting may provide modest relevance improvements",
            risk_level="low",
            priority=5
        )

    def _analyze_field_performance(self, context: OptimizationContext) -> Dict[str, Any]:
        """Analyze how different fields are performing in current queries."""
        analysis = {}
        
        if context.previous_results:
            # Analyze which fields might be contributing to good/bad results
            latest_result = context.previous_results[-1]
            
            # Look at query results to infer field importance
            for query_id, query_metrics in latest_result.metrics.get("per_query", {}).items():
                metric_value = query_metrics.get(context.experiment_config.primary_metric, 0)
                
                if metric_value > 0.7:
                    analysis[f"high_performing_query_{query_id}"] = {
                        "metric": metric_value,
                        "status": "good"
                    }
                elif metric_value < 0.3:
                    analysis[f"low_performing_query_{query_id}"] = {
                        "metric": metric_value,
                        "status": "needs_improvement"
                    }
        
        return analysis

    def _calculate_confidence(self, recommendation: SchemaRecommendation, context: OptimizationContext) -> float:
        """Calculate confidence score for the recommendation."""
        confidence = 0.8  # Base confidence
        
        # Reduce confidence if too many drastic changes
        if len(recommendation.field_boosts) > 10:
            confidence -= 0.2
            
        # Increase confidence if recommendations are conservative
        boost_values = list(recommendation.field_boosts.values())
        if boost_values and max(boost_values) <= 3.0 and min(boost_values) >= 0.1:
            confidence += 0.1
            
        return min(1.0, max(0.1, confidence))

    def _assess_risk_level(self, recommendation: SchemaRecommendation) -> str:
        """Assess risk level of the recommendation."""
        risk_factors = 0
        
        # Count risky changes
        if len(recommendation.new_field_suggestions) > 2:
            risk_factors += 1
            
        if len(recommendation.field_analysis_suggestions) > 5:
            risk_factors += 1
            
        # Check for extreme boost values
        boost_values = list(recommendation.field_boosts.values())
        if boost_values and (max(boost_values) > 5.0 or min(boost_values) < 0.05):
            risk_factors += 2
            
        if risk_factors >= 3:
            return "high"
        elif risk_factors >= 1:
            return "medium"
        else:
            return "low"

    def _calculate_priority(self, recommendation: SchemaRecommendation, context: OptimizationContext) -> int:
        """Calculate priority score (1-10) for the recommendation."""
        priority = 5  # Base priority
        
        # Higher priority if current performance is poor
        current_score = context.current_metrics.get(context.experiment_config.primary_metric, 0)
        if current_score < 0.3:
            priority += 3
        elif current_score < 0.6:
            priority += 1
            
        # Higher priority if we have many field boost suggestions
        if len(recommendation.field_boosts) > 3:
            priority += 1
            
        # Lower priority if too many complex changes
        if len(recommendation.new_field_suggestions) > 3:
            priority -= 1
            
        return min(10, max(1, priority))
