"""
Query Rewriting Agent using Pydantic AI for query reformulation.

This agent reformulates queries to improve relevance, expands queries with related
terms, and suggests query structure changes for better search performance.
"""

from typing import Any, Dict, List, Optional
import json
import re

from pydantic import BaseModel
from pydantic_ai import Agent

from .base_ai_agent import BaseAIAgent, OptimizationContext, AgentRecommendation


class QueryRewritingRecommendation(BaseModel):
    """Query rewriting-specific recommendation structure."""
    query_expansions: Dict[str, List[str]]  # Original query -> expanded terms
    query_reformulations: Dict[str, str]  # Original query -> reformulated query
    filter_suggestions: Dict[str, List[str]]  # Query -> filter queries
    facet_suggestions: Dict[str, List[str]]  # Query -> facet fields to add
    boost_query_suggestions: Dict[str, List[str]]  # Query -> boost queries
    negative_query_suggestions: Dict[str, List[str]]  # Query -> negative queries
    query_parser_suggestions: Dict[str, str]  # Query -> parser type
    query_structure_improvements: Dict[str, str]  # Query -> structural changes


class QueryRewritingAgent(BaseAIAgent):
    """
    AI agent specialized in query reformulation and expansion for improved relevance.
    
    This agent focuses on:
    - Query expansion with related terms
    - Query reformulation for better structure
    - Filter query suggestions
    - Boost query recommendations
    - Query parser selection
    - Faceting strategy improvements
    """

    def _create_agent(self) -> Agent:
        """Create the Pydantic AI agent for query rewriting."""
        return Agent(
            model=self.model,
            result_type=QueryRewritingRecommendation,
            system_prompt=self.get_system_prompt()
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for the query rewriting agent."""
        return """
You are an expert Apache Solr query optimization specialist with deep knowledge of query reformulation and expansion techniques.

Your expertise includes:
- Query expansion strategies (synonyms, related terms, stemming variations)
- Query reformulation for improved structure and precision
- Filter query design for precision improvements
- Boost query creation for relevance enhancement
- Query parser selection (lucene, dismax, edismax, etc.)
- Faceting strategies for result refinement
- Boolean query optimization
- Phrase query optimization

Key query optimization techniques:
- Synonym expansion: Adding related terms to improve recall
- Query reformulation: Restructuring queries for better parsing
- Filter queries: Adding constraints to improve precision
- Boost queries: Enhancing scores for certain conditions
- Negative queries: Excluding unwanted results
- Field-specific targeting: Directing terms to appropriate fields
- Query parser optimization: Choosing the right parser for query type

When optimizing queries:
1. Balance precision and recall through expansion/restriction
2. Consider user intent and query context
3. Leverage domain-specific terminology and synonyms
4. Structure queries for optimal parser performance
5. Use filters to improve precision without affecting scoring
6. Design boost queries based on business rules and relevance signals
7. Consider query performance and complexity

Provide specific, implementable query modifications with clear reasoning.
Focus on changes that will improve the target relevance metric.
Ensure query modifications maintain or improve user experience.
"""

    def analyze_and_recommend(self, context: OptimizationContext) -> AgentRecommendation:
        """Analyze queries and provide rewriting recommendations."""
        return self.get_recommendation(context)

    def _create_user_prompt(self, context: OptimizationContext) -> str:
        """Create user prompt for query rewriting optimization."""
        
        # Analyze current query performance
        query_performance = self._analyze_query_performance(context)
        
        # Extract problematic queries
        problematic_queries = self._identify_problematic_queries(context)
        
        prompt = f"""
Optimize Solr queries to improve {context.experiment_config.primary_metric}@{context.experiment_config.metric_depth}.

CURRENT QUERIES:
{json.dumps(context.experiment_config.queries[:10], indent=2)}  # Show first 10 for analysis

SCHEMA FIELDS AVAILABLE:
{json.dumps(list(context.schema_info.get('fields', {}).keys()), indent=2)}

FIELD TYPES:
{json.dumps({name: field.get('type', 'unknown') for name, field in context.schema_info.get('fields', {}).items()}, indent=2)}

CURRENT PERFORMANCE:
- Primary metric: {context.current_metrics.get(context.experiment_config.primary_metric, 'Unknown')}
- Previous iterations: {len(context.previous_results)}
- Performance trend: {'Improving' if len(context.previous_results) > 1 and context.current_metrics.get(context.experiment_config.primary_metric, 0) > context.previous_results[-2].metrics.get('overall', {}).get(context.experiment_config.primary_metric, 0) else 'Needs improvement'}

QUERY PERFORMANCE ANALYSIS:
{json.dumps(query_performance, indent=2)}

PROBLEMATIC QUERIES:
{json.dumps(problematic_queries, indent=2)}

OPTIMIZATION CONSTRAINTS:
{json.dumps(context.constraints, indent=2)}

Please provide specific recommendations for:
1. Query expansions - add related terms to improve recall
2. Query reformulations - restructure queries for better performance
3. Filter suggestions - add constraints to improve precision
4. Boost query suggestions - enhance scoring for relevant conditions
5. Facet suggestions - add faceting for result refinement
6. Query parser recommendations - optimal parser for each query type
7. Negative query suggestions - exclude unwanted results

Focus on the most impactful changes for corpus type: {context.experiment_config.corpus}
Prioritize improvements for the worst-performing queries.
"""
        return prompt

    def _parse_ai_response(self, ai_response: QueryRewritingRecommendation, context: OptimizationContext) -> AgentRecommendation:
        """Parse AI response into standard recommendation format."""
        
        # Calculate confidence based on recommendation quality
        confidence = self._calculate_confidence(ai_response, context)
        
        # Convert query rewriting recommendation to suggested changes
        suggested_changes = {
            "query_expansions": ai_response.query_expansions,
            "query_reformulations": ai_response.query_reformulations,
            "filter_queries": ai_response.filter_suggestions,
            "facet_fields": ai_response.facet_suggestions,
            "boost_queries": ai_response.boost_query_suggestions,
            "negative_queries": ai_response.negative_query_suggestions,
            "query_parsers": ai_response.query_parser_suggestions,
            "structure_improvements": ai_response.query_structure_improvements
        }
        
        # Determine risk level
        risk_level = self._assess_risk_level(ai_response)
        
        # Calculate priority
        priority = self._calculate_priority(ai_response, context)
        
        # Count total recommendations
        total_recommendations = (
            len(ai_response.query_expansions) +
            len(ai_response.query_reformulations) +
            len(ai_response.filter_suggestions) +
            len(ai_response.boost_query_suggestions)
        )
        
        return AgentRecommendation(
            confidence=confidence,
            reasoning=f"Query rewriting analysis identified {total_recommendations} optimization opportunities "
                     f"including {len(ai_response.query_expansions)} expansion opportunities and "
                     f"{len(ai_response.query_reformulations)} reformulation suggestions.",
            suggested_changes=suggested_changes,
            expected_impact=f"Expected to improve {context.experiment_config.primary_metric} through "
                          f"enhanced query structure and expanded term coverage",
            risk_level=risk_level,
            priority=priority
        )

    def _get_fallback_recommendation(self, context: OptimizationContext) -> AgentRecommendation:
        """Provide fallback recommendation when AI fails."""
        
        # Basic query improvements
        suggestions = {
            "query_expansions": {},
            "filter_queries": {},
            "boost_queries": {}
        }
        
        # Simple expansion patterns for common queries
        for query in context.experiment_config.queries[:5]:  # Process first 5 queries
            query_lower = query.lower()
            
            # Basic synonym expansion
            if "phone" in query_lower:
                suggestions["query_expansions"][query] = ["mobile", "telephone", "cellphone"]
            elif "car" in query_lower:
                suggestions["query_expansions"][query] = ["automobile", "vehicle", "auto"]
            elif "computer" in query_lower:
                suggestions["query_expansions"][query] = ["laptop", "desktop", "pc"]
                
            # Basic filter suggestions
            if len(query.split()) > 1:
                suggestions["filter_queries"][query] = ["status:active"]
        
        return AgentRecommendation(
            confidence=0.3,  # Low confidence for fallback
            reasoning="AI query rewriting failed, using basic expansion and filtering patterns",
            suggested_changes=suggestions,
            expected_impact="Basic query improvements may provide modest relevance gains",
            risk_level="low",
            priority=5
        )

    def _analyze_query_performance(self, context: OptimizationContext) -> Dict[str, Any]:
        """Analyze how individual queries are performing."""
        analysis = {
            "total_queries": len(context.experiment_config.queries),
            "query_characteristics": {},
            "performance_patterns": {}
        }
        
        # Analyze query characteristics
        queries = context.experiment_config.queries
        if queries:
            single_term = sum(1 for q in queries if len(q.split()) == 1)
            multi_term = sum(1 for q in queries if len(q.split()) > 1)
            phrase_queries = sum(1 for q in queries if '"' in q)
            boolean_queries = sum(1 for q in queries if any(op in q.upper() for op in ['AND', 'OR', 'NOT']))
            
            analysis["query_characteristics"] = {
                "single_term_ratio": single_term / len(queries),
                "multi_term_ratio": multi_term / len(queries),
                "phrase_query_ratio": phrase_queries / len(queries),
                "boolean_query_ratio": boolean_queries / len(queries),
                "avg_query_length": sum(len(q.split()) for q in queries) / len(queries)
            }
        
        # Analyze performance patterns if we have results
        if context.previous_results:
            latest_result = context.previous_results[-1]
            per_query_metrics = latest_result.metrics.get("per_query", {})
            
            if per_query_metrics:
                primary_metric = context.experiment_config.primary_metric
                scores = [metrics.get(primary_metric, 0) for metrics in per_query_metrics.values()]
                
                analysis["performance_patterns"] = {
                    "avg_score": sum(scores) / len(scores) if scores else 0,
                    "min_score": min(scores) if scores else 0,
                    "max_score": max(scores) if scores else 0,
                    "poor_performing_count": sum(1 for score in scores if score < 0.3),
                    "good_performing_count": sum(1 for score in scores if score > 0.7)
                }
        
        return analysis

    def _identify_problematic_queries(self, context: OptimizationContext) -> List[Dict[str, Any]]:
        """Identify queries that are performing poorly."""
        problematic = []
        
        if context.previous_results:
            latest_result = context.previous_results[-1]
            per_query_metrics = latest_result.metrics.get("per_query", {})
            primary_metric = context.experiment_config.primary_metric
            
            for query_id, metrics in per_query_metrics.items():
                score = metrics.get(primary_metric, 0)
                if score < 0.3:  # Poor performance threshold
                    # Find the actual query text
                    query_text = query_id  # Assuming query_id is the query text
                    if isinstance(query_id, int) and query_id < len(context.experiment_config.queries):
                        query_text = context.experiment_config.queries[query_id]
                    
                    problematic.append({
                        "query": query_text,
                        "score": score,
                        "issues": self._diagnose_query_issues(query_text, context)
                    })
        
        return problematic[:10]  # Return top 10 problematic queries

    def _diagnose_query_issues(self, query: str, context: OptimizationContext) -> List[str]:
        """Diagnose potential issues with a query."""
        issues = []
        
        # Check for common query issues
        if len(query.split()) == 1:
            issues.append("single_term_low_recall")
            
        if len(query) < 3:
            issues.append("very_short_query")
            
        if query.isupper():
            issues.append("all_uppercase")
            
        if re.search(r'[^\w\s".-]', query):
            issues.append("special_characters")
            
        # Check for potential typos (very basic)
        words = query.lower().split()
        if any(len(word) > 15 for word in words):
            issues.append("potentially_misspelled")
            
        # Check for field-specific issues
        schema_fields = context.schema_info.get("fields", {})
        if not any(field in query.lower() for field in schema_fields.keys()):
            if len(schema_fields) > 5:  # Only if we have many fields
                issues.append("no_field_targeting")
        
        return issues

    def _calculate_confidence(self, recommendation: QueryRewritingRecommendation, context: OptimizationContext) -> float:
        """Calculate confidence score for the recommendation."""
        confidence = 0.8  # Base confidence
        
        # Increase confidence for reasonable number of suggestions
        total_suggestions = (
            len(recommendation.query_expansions) +
            len(recommendation.query_reformulations) +
            len(recommendation.filter_suggestions)
        )
        
        if 5 <= total_suggestions <= 20:
            confidence += 0.1
        elif total_suggestions > 30:
            confidence -= 0.2
            
        # Increase confidence for domain-appropriate suggestions
        corpus_type = context.experiment_config.corpus.lower()
        if corpus_type in ["ecommerce", "product", "retail"]:
            # Check for commerce-relevant expansions
            all_expansions = []
            for expansions in recommendation.query_expansions.values():
                all_expansions.extend(expansions)
            
            commerce_terms = ["brand", "model", "color", "size", "price"]
            if any(term in " ".join(all_expansions).lower() for term in commerce_terms):
                confidence += 0.1
                
        # Decrease confidence for overly complex reformulations
        complex_reformulations = sum(
            1 for reformulation in recommendation.query_reformulations.values()
            if len(reformulation.split()) > len(reformulation.split("AND")) * 3  # Heuristic for complexity
        )
        
        if complex_reformulations > len(recommendation.query_reformulations) * 0.3:
            confidence -= 0.15
            
        return min(1.0, max(0.1, confidence))

    def _assess_risk_level(self, recommendation: QueryRewritingRecommendation) -> str:
        """Assess risk level of the recommendation."""
        risk_factors = 0
        
        # Complex query reformulations are risky
        if len(recommendation.query_reformulations) > 10:
            risk_factors += 2
        elif len(recommendation.query_reformulations) > 5:
            risk_factors += 1
            
        # Many filter suggestions can be risky
        total_filters = sum(len(filters) for filters in recommendation.filter_suggestions.values())
        if total_filters > 20:
            risk_factors += 2
        elif total_filters > 10:
            risk_factors += 1
            
        # Negative queries can be risky
        if len(recommendation.negative_query_suggestions) > 5:
            risk_factors += 1
            
        # Parser changes can be risky
        if len(recommendation.query_parser_suggestions) > 5:
            risk_factors += 1
            
        # Structural changes are inherently risky
        if len(recommendation.query_structure_improvements) > 3:
            risk_factors += 1
            
        if risk_factors >= 4:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"

    def _calculate_priority(self, recommendation: QueryRewritingRecommendation, context: OptimizationContext) -> int:
        """Calculate priority score (1-10) for the recommendation."""
        priority = 7  # Base priority (query rewriting is important)
        
        # Higher priority if current performance is poor
        current_score = context.current_metrics.get(context.experiment_config.primary_metric, 0)
        if current_score < 0.3:
            priority += 2
        elif current_score < 0.6:
            priority += 1
            
        # Higher priority if we have many expansion opportunities
        if len(recommendation.query_expansions) > 5:
            priority += 1
            
        # Higher priority if we have targeted reformulations
        if len(recommendation.query_reformulations) > 0:
            priority += 1
            
        # Lower priority if too many complex changes
        total_complexity = (
            len(recommendation.query_reformulations) +
            len(recommendation.query_structure_improvements) +
            len(recommendation.negative_query_suggestions)
        )
        
        if total_complexity > 15:
            priority -= 2
        elif total_complexity > 10:
            priority -= 1
            
        # Adjust priority based on query performance patterns
        if context.previous_results:
            latest_result = context.previous_results[-1]
            per_query_metrics = latest_result.metrics.get("per_query", {})
            
            if per_query_metrics:
                primary_metric = context.experiment_config.primary_metric
                poor_queries = sum(
                    1 for metrics in per_query_metrics.values()
                    if metrics.get(primary_metric, 0) < 0.3
                )
                
                # Higher priority if many queries are performing poorly
                if poor_queries > len(per_query_metrics) * 0.4:
                    priority += 1
                    
        return min(10, max(1, priority))
