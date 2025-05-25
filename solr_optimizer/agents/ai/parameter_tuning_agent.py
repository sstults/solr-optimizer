"""
Parameter Tuning Agent using Pydantic AI for query parameter optimization.

This agent optimizes DisMax/eDisMax parameters, tunes minimum_should_match settings,
and adjusts boosting factors and function queries for improved relevance.
"""

from typing import Any, Dict, List, Optional
import json
import re

from pydantic import BaseModel
from pydantic_ai import Agent

from .base_ai_agent import BaseAIAgent, OptimizationContext, AgentRecommendation


class ParameterRecommendation(BaseModel):
    """Parameter-specific recommendation structure."""
    qf_weights: Dict[str, float]  # Query field weights
    pf_weights: Dict[str, float]  # Phrase field weights
    mm_setting: str  # Minimum should match
    tie_parameter: float  # Tie breaker
    boost_functions: List[str]  # Function queries for boosting
    boost_queries: List[str]  # Boost queries
    query_slop: Optional[int]  # Phrase slop
    phrase_slop: Optional[int]  # Phrase slop for pf
    other_params: Dict[str, Any]  # Other edismax parameters


class ParameterTuningAgent(BaseAIAgent):
    """
    AI agent specialized in tuning Solr query parameters for optimal relevance.
    
    This agent focuses on:
    - DisMax/eDisMax parameter optimization (qf, pf, mm, tie)
    - Minimum should match strategies
    - Function query boosting
    - Phrase matching optimization
    - Query slop and proximity settings
    """

    def _create_agent(self) -> Agent:
        """Create the Pydantic AI agent for parameter tuning."""
        return Agent(
            model=self.model,
            result_type=ParameterRecommendation,
            system_prompt=self.get_system_prompt()
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for the parameter tuning agent."""
        return """
You are an expert Apache Solr query parameter optimization specialist with deep knowledge of DisMax and eDisMax query parsers.

Your expertise includes:
- DisMax/eDisMax parameter optimization (qf, pf, mm, tie, qs, ps)
- Minimum should match (mm) strategies for different query types
- Field weight balancing for optimal relevance
- Phrase boosting strategies using pf parameters
- Function query design for document scoring
- Boost query patterns for relevance enhancement
- Query slop and proximity matching optimization

Key parameter knowledge:
- qf (Query Fields): Field weights for term matching
- pf (Phrase Fields): Field weights for phrase proximity matching
- mm (Minimum Match): How many query terms must match
- tie (Tie Breaker): How to combine scores from multiple fields
- qs (Query Slop): Slop for user query phrases
- ps (Phrase Slop): Slop for pf phrase matching
- bf/boost: Function queries for document boosting
- bq: Boost queries for specific conditions

When optimizing parameters:
1. Balance precision vs recall through mm settings
2. Weight important fields higher in qf
3. Use pf to boost phrase matches in key fields
4. Set appropriate tie values for field score combination
5. Design boost functions based on document features
6. Consider query length and complexity in parameter choices

Provide specific, tested parameter values with clear reasoning.
Focus on parameters that will improve the target relevance metric.
Ensure parameter combinations are compatible and well-balanced.
"""

    def analyze_and_recommend(self, context: OptimizationContext) -> AgentRecommendation:
        """Analyze parameters and provide recommendations."""
        return self.get_recommendation(context)

    def _create_user_prompt(self, context: OptimizationContext) -> str:
        """Create user prompt for parameter optimization."""
        
        # Get current query configuration if available
        current_config = self._extract_current_config(context)
        
        # Analyze query performance patterns
        query_analysis = self._analyze_query_patterns(context)
        
        prompt = f"""
Optimize Solr query parameters to improve {context.experiment_config.primary_metric}@{context.experiment_config.metric_depth}.

CURRENT CONFIGURATION:
{json.dumps(current_config, indent=2)}

SCHEMA FIELDS AVAILABLE:
{json.dumps(list(context.schema_info.get('fields', {}).keys()), indent=2)}

FIELD TYPES:
{json.dumps({name: field.get('type', 'unknown') for name, field in context.schema_info.get('fields', {}).items()}, indent=2)}

CURRENT PERFORMANCE:
- Primary metric: {context.current_metrics.get(context.experiment_config.primary_metric, 'Unknown')}
- Previous iterations: {len(context.previous_results)}
- Performance trend: {'Improving' if len(context.previous_results) > 1 and context.current_metrics.get(context.experiment_config.primary_metric, 0) > context.previous_results[-2].metrics.get('overall', {}).get(context.experiment_config.primary_metric, 0) else 'Needs improvement'}

QUERY ANALYSIS:
{json.dumps(query_analysis, indent=2)}

OPTIMIZATION CONSTRAINTS:
{json.dumps(context.constraints, indent=2)}

Please provide optimized parameters for:
1. qf (query field weights) - balance field importance
2. pf (phrase field weights) - boost phrase matches
3. mm (minimum should match) - precision/recall balance
4. tie (tie breaker) - field score combination
5. boost functions (bf/boost) - document-level boosting
6. boost queries (bq) - conditional boosting
7. slop parameters (qs, ps) - proximity matching

Focus on parameter combinations that work well together and address the specific performance issues identified.
Corpus type: {context.experiment_config.corpus}
"""
        return prompt

    def _parse_ai_response(self, ai_response: ParameterRecommendation, context: OptimizationContext) -> AgentRecommendation:
        """Parse AI response into standard recommendation format."""
        
        # Calculate confidence based on parameter reasonableness
        confidence = self._calculate_confidence(ai_response, context)
        
        # Convert parameter recommendation to suggested changes
        suggested_changes = {
            "qf": " ".join([f"{field}^{weight}" for field, weight in ai_response.qf_weights.items()]),
            "pf": " ".join([f"{field}^{weight}" for field, weight in ai_response.pf_weights.items()]) if ai_response.pf_weights else None,
            "mm": ai_response.mm_setting,
            "tie": ai_response.tie_parameter,
            "bf": ai_response.boost_functions,
            "bq": ai_response.boost_queries,
            "qs": ai_response.query_slop,
            "ps": ai_response.phrase_slop
        }
        
        # Add other parameters
        suggested_changes.update(ai_response.other_params)
        
        # Remove None values
        suggested_changes = {k: v for k, v in suggested_changes.items() if v is not None}
        
        # Determine risk level
        risk_level = self._assess_risk_level(ai_response)
        
        # Calculate priority
        priority = self._calculate_priority(ai_response, context)
        
        return AgentRecommendation(
            confidence=confidence,
            reasoning=f"Parameter optimization targeting {len(ai_response.qf_weights)} query fields, "
                     f"mm setting '{ai_response.mm_setting}', and {len(ai_response.boost_functions)} boost functions.",
            suggested_changes=suggested_changes,
            expected_impact=f"Expected to improve {context.experiment_config.primary_metric} through "
                          f"optimized field weighting and matching strategy",
            risk_level=risk_level,
            priority=priority
        )

    def _get_fallback_recommendation(self, context: OptimizationContext) -> AgentRecommendation:
        """Provide fallback recommendation when AI fails."""
        schema_fields = context.schema_info.get("fields", {})
        
        # Simple heuristic-based parameter suggestions
        qf_weights = {}
        for field_name, field_info in schema_fields.items():
            field_type = field_info.get("type", "")
            if "title" in field_name.lower():
                qf_weights[field_name] = 2.0
            elif "content" in field_name.lower() or "description" in field_name.lower():
                qf_weights[field_name] = 1.0
            elif "text" in field_type:
                qf_weights[field_name] = 0.5
        
        # Conservative mm setting
        mm_setting = "2<-1 5<-2 6<90%"
        
        suggested_changes = {
            "qf": " ".join([f"{field}^{weight}" for field, weight in qf_weights.items()]),
            "mm": mm_setting,
            "tie": 0.1
        }
        
        return AgentRecommendation(
            confidence=0.4,  # Low confidence for fallback
            reasoning="AI parameter optimization failed, using conservative heuristic-based settings",
            suggested_changes=suggested_changes,
            expected_impact="Conservative parameter tuning may provide modest improvements",
            risk_level="low",
            priority=6
        )

    def _extract_current_config(self, context: OptimizationContext) -> Dict[str, Any]:
        """Extract current query configuration from context."""
        if context.previous_results:
            latest_result = context.previous_results[-1]
            return latest_result.configuration.__dict__ if hasattr(latest_result, 'configuration') else {}
        return {}

    def _analyze_query_patterns(self, context: OptimizationContext) -> Dict[str, Any]:
        """Analyze query patterns to inform parameter optimization."""
        analysis = {
            "query_count": len(context.experiment_config.queries),
            "avg_query_length": 0,
            "query_types": {"single_term": 0, "multi_term": 0, "phrase": 0},
            "performance_issues": []
        }
        
        # Analyze query characteristics
        total_terms = 0
        for query in context.experiment_config.queries:
            terms = query.split()
            total_terms += len(terms)
            
            if len(terms) == 1:
                analysis["query_types"]["single_term"] += 1
            elif '"' in query:
                analysis["query_types"]["phrase"] += 1
            else:
                analysis["query_types"]["multi_term"] += 1
        
        if analysis["query_count"] > 0:
            analysis["avg_query_length"] = total_terms / analysis["query_count"]
        
        # Identify performance issues based on metrics
        if context.current_metrics:
            primary_metric = context.experiment_config.primary_metric
            current_score = context.current_metrics.get(primary_metric, 0)
            
            if current_score < 0.3:
                analysis["performance_issues"].append("low_overall_relevance")
            
            # Analyze per-query performance if available
            if context.previous_results:
                latest_result = context.previous_results[-1]
                per_query_metrics = latest_result.metrics.get("per_query", {})
                
                poor_queries = [qid for qid, metrics in per_query_metrics.items() 
                              if metrics.get(primary_metric, 0) < 0.2]
                
                if len(poor_queries) > len(per_query_metrics) * 0.3:
                    analysis["performance_issues"].append("many_poor_queries")
        
        return analysis

    def _calculate_confidence(self, recommendation: ParameterRecommendation, context: OptimizationContext) -> float:
        """Calculate confidence score for the recommendation."""
        confidence = 0.8  # Base confidence
        
        # Check for reasonable qf weights
        qf_weights = list(recommendation.qf_weights.values())
        if qf_weights:
            if max(qf_weights) > 10.0 or min(qf_weights) < 0.01:
                confidence -= 0.2
            if max(qf_weights) <= 5.0 and min(qf_weights) >= 0.1:
                confidence += 0.1
        
        # Check mm setting validity
        if self._validate_mm_setting(recommendation.mm_setting):
            confidence += 0.1
        else:
            confidence -= 0.3
            
        # Check tie parameter
        if 0.0 <= recommendation.tie_parameter <= 1.0:
            confidence += 0.05
        else:
            confidence -= 0.2
            
        return min(1.0, max(0.1, confidence))

    def _validate_mm_setting(self, mm_setting: str) -> bool:
        """Validate that mm setting is properly formatted."""
        if not mm_setting:
            return False
            
        # Check for common mm patterns
        valid_patterns = [
            r'^\d+$',  # Simple number
            r'^\d+%$',  # Percentage
            r'^\d+<-?\d+',  # Conditional with count
            r'^\d+<\d+%',  # Conditional with percentage
        ]
        
        for pattern in valid_patterns:
            if re.match(pattern, mm_setting.split()[0]):
                return True
                
        return False

    def _assess_risk_level(self, recommendation: ParameterRecommendation) -> str:
        """Assess risk level of the recommendation."""
        risk_factors = 0
        
        # Check for extreme qf weights
        qf_weights = list(recommendation.qf_weights.values())
        if qf_weights and (max(qf_weights) > 5.0 or min(qf_weights) < 0.05):
            risk_factors += 2
            
        # Check for complex mm settings
        if len(recommendation.mm_setting.split()) > 2:
            risk_factors += 1
            
        # Check for many boost functions
        if len(recommendation.boost_functions) > 3:
            risk_factors += 1
            
        # Check tie parameter
        if recommendation.tie_parameter > 0.5:
            risk_factors += 1
            
        if risk_factors >= 3:
            return "high"
        elif risk_factors >= 1:
            return "medium"
        else:
            return "low"

    def _calculate_priority(self, recommendation: ParameterRecommendation, context: OptimizationContext) -> int:
        """Calculate priority score (1-10) for the recommendation."""
        priority = 7  # Base priority (parameter tuning is important)
        
        # Higher priority if current performance is poor
        current_score = context.current_metrics.get(context.experiment_config.primary_metric, 0)
        if current_score < 0.3:
            priority += 2
        elif current_score < 0.6:
            priority += 1
            
        # Higher priority if we have many field optimizations
        if len(recommendation.qf_weights) > 5:
            priority += 1
            
        # Lower priority if too complex
        total_complexity = (len(recommendation.boost_functions) + 
                          len(recommendation.boost_queries) + 
                          len(recommendation.mm_setting.split()))
        if total_complexity > 10:
            priority -= 2
            
        return min(10, max(1, priority))
