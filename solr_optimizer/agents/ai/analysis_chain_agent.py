"""
Analysis Chain Agent using Pydantic AI for tokenization and analyzer optimization.

This agent evaluates and optimizes tokenization, stemming, and other text analysis
processes, suggests synonym expansions, stopword adjustments, and analyzer configurations.
"""

from typing import Any, Dict, List, Optional
import json

from pydantic import BaseModel
from pydantic_ai import Agent

from .base_ai_agent import BaseAIAgent, OptimizationContext, AgentRecommendation


class AnalysisChainRecommendation(BaseModel):
    """Analysis chain-specific recommendation structure."""
    tokenizer_suggestions: Dict[str, str]  # Field -> tokenizer type
    filter_suggestions: Dict[str, List[str]]  # Field -> list of filters
    synonym_suggestions: Dict[str, List[str]]  # Field -> synonym groups
    stopword_suggestions: Dict[str, List[str]]  # Field -> stopwords to add/remove
    stemmer_suggestions: Dict[str, str]  # Field -> stemmer type
    char_filter_suggestions: Dict[str, List[str]]  # Field -> character filters
    analyzer_type_suggestions: Dict[str, str]  # Field -> analyzer type
    custom_analyzer_configs: List[Dict[str, Any]]  # Complete analyzer definitions


class AnalysisChainAgent(BaseAIAgent):
    """
    AI agent specialized in optimizing Solr text analysis chains.
    
    This agent focuses on:
    - Tokenizer optimization for different content types
    - Filter chain optimization (stemming, synonyms, stopwords)
    - Character filter recommendations
    - Custom analyzer design
    - Language-specific analysis optimization
    """

    def _create_agent(self) -> Agent:
        """Create the Pydantic AI agent for analysis chain optimization."""
        return Agent(
            model=self.model,
            result_type=AnalysisChainRecommendation,
            system_prompt=self.get_system_prompt()
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for the analysis chain agent."""
        return """
You are an expert Apache Solr text analysis specialist with deep knowledge of tokenizers, filters, and analyzers.

Your expertise includes:
- Tokenizer selection (Standard, Keyword, Pattern, WhitespaceTokenizer, etc.)
- Token filter optimization (stemming, synonyms, stopwords, lowercasing, etc.)
- Character filter configuration (HTML stripping, pattern replacement, etc.)
- Language-specific analysis chains
- Custom analyzer design for specific domains
- Analysis chain performance optimization
- Query-time vs index-time analysis considerations

Key analysis components:
- Tokenizers: Break text into tokens (StandardTokenizer, KeywordTokenizer, PatternTokenizer)
- Token Filters: Transform tokens (LowerCaseFilter, StopFilter, SynonymFilter, SnowballPorterFilter)
- Character Filters: Pre-process text before tokenization (HTMLStripCharFilter, PatternReplaceCharFilter)

When optimizing analysis chains:
1. Consider the content type and language of the corpus
2. Balance precision vs recall through stemming and synonym strategies
3. Optimize for the specific query patterns and user expectations
4. Ensure consistency between index-time and query-time analysis
5. Consider performance implications of complex filter chains
6. Design for the target domain (e.g., legal, medical, e-commerce)

Provide specific, implementable analyzer configurations with clear reasoning.
Focus on changes that will improve search relevance for the target metric.
Consider both precision and recall implications of analysis choices.
"""

    def analyze_and_recommend(self, context: OptimizationContext) -> AgentRecommendation:
        """Analyze text analysis chains and provide recommendations."""
        return self.get_recommendation(context)

    def _create_user_prompt(self, context: OptimizationContext) -> str:
        """Create user prompt for analysis chain optimization."""
        
        # Extract current analysis configuration
        current_analysis = self._extract_current_analysis(context)
        
        # Analyze query characteristics for analysis optimization
        query_analysis = self._analyze_query_characteristics(context)
        
        prompt = f"""
Optimize Solr text analysis chains to improve {context.experiment_config.primary_metric}@{context.experiment_config.metric_depth}.

CURRENT ANALYSIS CONFIGURATION:
{json.dumps(current_analysis, indent=2)}

SCHEMA FIELDS AND TYPES:
{json.dumps({name: field.get('type', 'unknown') for name, field in context.schema_info.get('fields', {}).items()}, indent=2)}

CURRENT PERFORMANCE:
- Primary metric: {context.current_metrics.get(context.experiment_config.primary_metric, 'Unknown')}
- Previous iterations: {len(context.previous_results)}
- Performance trend: {'Improving' if len(context.previous_results) > 1 and context.current_metrics.get(context.experiment_config.primary_metric, 0) > context.previous_results[-2].metrics.get('overall', {}).get(context.experiment_config.primary_metric, 0) else 'Needs improvement'}

QUERY CHARACTERISTICS:
{json.dumps(query_analysis, indent=2)}

OPTIMIZATION CONSTRAINTS:
{json.dumps(context.constraints, indent=2)}

Please provide specific recommendations for:
1. Tokenizer selection for each field type
2. Token filter chains (stemming, synonyms, stopwords)
3. Character filter preprocessing
4. Synonym group suggestions based on query patterns
5. Stopword adjustments for the domain
6. Custom analyzer configurations for optimal performance

Consider the corpus type: {context.experiment_config.corpus}
Focus on analysis choices that will improve search matching and relevance.
Balance between precision (exact matching) and recall (broader matching).
"""
        return prompt

    def _parse_ai_response(self, ai_response: AnalysisChainRecommendation, context: OptimizationContext) -> AgentRecommendation:
        """Parse AI response into standard recommendation format."""
        
        # Calculate confidence based on recommendation quality
        confidence = self._calculate_confidence(ai_response, context)
        
        # Convert analysis recommendation to suggested changes
        suggested_changes = {
            "tokenizers": ai_response.tokenizer_suggestions,
            "filters": ai_response.filter_suggestions,
            "synonyms": ai_response.synonym_suggestions,
            "stopwords": ai_response.stopword_suggestions,
            "stemmers": ai_response.stemmer_suggestions,
            "char_filters": ai_response.char_filter_suggestions,
            "analyzer_types": ai_response.analyzer_type_suggestions,
            "custom_analyzers": ai_response.custom_analyzer_configs
        }
        
        # Determine risk level
        risk_level = self._assess_risk_level(ai_response)
        
        # Calculate priority
        priority = self._calculate_priority(ai_response, context)
        
        # Count total recommendations
        total_recommendations = (
            len(ai_response.tokenizer_suggestions) +
            len(ai_response.filter_suggestions) +
            len(ai_response.synonym_suggestions) +
            len(ai_response.custom_analyzer_configs)
        )
        
        return AgentRecommendation(
            confidence=confidence,
            reasoning=f"Analysis chain optimization identified {total_recommendations} improvement opportunities "
                     f"including {len(ai_response.synonym_suggestions)} synonym enhancements and "
                     f"{len(ai_response.custom_analyzer_configs)} custom analyzer configurations.",
            suggested_changes=suggested_changes,
            expected_impact=f"Expected to improve {context.experiment_config.primary_metric} through "
                          f"enhanced text analysis and better query-document matching",
            risk_level=risk_level,
            priority=priority
        )

    def _get_fallback_recommendation(self, context: OptimizationContext) -> AgentRecommendation:
        """Provide fallback recommendation when AI fails."""
        schema_fields = context.schema_info.get("fields", {})
        
        # Basic analysis chain improvements
        suggestions = {
            "filters": {},
            "synonyms": {},
            "analyzer_types": {}
        }
        
        # Add basic recommendations for text fields
        for field_name, field_info in schema_fields.items():
            field_type = field_info.get("type", "")
            if "text" in field_type.lower():
                # Basic filter chain
                suggestions["filters"][field_name] = [
                    "lowercase",
                    "stop",
                    "snowball"
                ]
                
                # Basic synonym suggestions for common fields
                if "title" in field_name.lower():
                    suggestions["synonyms"][field_name] = [
                        "car,automobile,vehicle",
                        "phone,telephone,mobile"
                    ]
                
                # Suggest standard text analyzer
                suggestions["analyzer_types"][field_name] = "text_general"
        
        return AgentRecommendation(
            confidence=0.4,  # Low confidence for fallback
            reasoning="AI analysis chain optimization failed, using basic text analysis improvements",
            suggested_changes=suggestions,
            expected_impact="Basic analysis chain improvements may provide modest relevance gains",
            risk_level="low",
            priority=4
        )

    def _extract_current_analysis(self, context: OptimizationContext) -> Dict[str, Any]:
        """Extract current analysis configuration from schema."""
        analysis_config = {}
        
        # Extract field type analyzers from schema
        schema_fields = context.schema_info.get("fields", {})
        field_types = context.schema_info.get("fieldTypes", {})
        
        for field_name, field_info in schema_fields.items():
            field_type = field_info.get("type", "")
            if field_type in field_types:
                type_config = field_types[field_type]
                analysis_config[field_name] = {
                    "type": field_type,
                    "analyzer": type_config.get("analyzer", {}),
                    "indexAnalyzer": type_config.get("indexAnalyzer", {}),
                    "queryAnalyzer": type_config.get("queryAnalyzer", {})
                }
        
        return analysis_config

    def _analyze_query_characteristics(self, context: OptimizationContext) -> Dict[str, Any]:
        """Analyze query characteristics to inform analysis chain optimization."""
        analysis = {
            "query_count": len(context.experiment_config.queries),
            "language_indicators": {},
            "term_patterns": {},
            "complexity_indicators": {}
        }
        
        # Analyze query patterns
        total_queries = len(context.experiment_config.queries)
        if total_queries > 0:
            # Count different types of queries
            phrase_queries = sum(1 for q in context.experiment_config.queries if '"' in q)
            multi_word_queries = sum(1 for q in context.experiment_config.queries if len(q.split()) > 1)
            single_word_queries = total_queries - multi_word_queries
            
            analysis["complexity_indicators"] = {
                "phrase_query_ratio": phrase_queries / total_queries,
                "multi_word_ratio": multi_word_queries / total_queries,
                "single_word_ratio": single_word_queries / total_queries
            }
            
            # Analyze term patterns
            all_terms = []
            for query in context.experiment_config.queries:
                # Simple tokenization for analysis
                terms = query.lower().replace('"', '').split()
                all_terms.extend(terms)
            
            if all_terms:
                unique_terms = set(all_terms)
                analysis["term_patterns"] = {
                    "total_terms": len(all_terms),
                    "unique_terms": len(unique_terms),
                    "avg_term_length": sum(len(term) for term in all_terms) / len(all_terms),
                    "repetition_ratio": len(all_terms) / len(unique_terms) if unique_terms else 1
                }
                
                # Basic language detection heuristics
                english_indicators = sum(1 for term in unique_terms if term in ['the', 'and', 'or', 'for', 'with', 'in', 'on', 'at'])
                analysis["language_indicators"]["english_likelihood"] = english_indicators / len(unique_terms) if unique_terms else 0
        
        return analysis

    def _calculate_confidence(self, recommendation: AnalysisChainRecommendation, context: OptimizationContext) -> float:
        """Calculate confidence score for the recommendation."""
        confidence = 0.7  # Base confidence
        
        # Increase confidence for well-structured recommendations
        if recommendation.custom_analyzer_configs:
            # Check if custom analyzers are well-defined
            for analyzer in recommendation.custom_analyzer_configs:
                if "tokenizer" in analyzer and "filters" in analyzer:
                    confidence += 0.1
                    break
        
        # Decrease confidence for too many changes
        total_changes = (
            len(recommendation.tokenizer_suggestions) +
            len(recommendation.filter_suggestions) +
            len(recommendation.synonym_suggestions) +
            len(recommendation.custom_analyzer_configs)
        )
        
        if total_changes > 15:
            confidence -= 0.2
        elif total_changes > 10:
            confidence -= 0.1
            
        # Increase confidence for domain-appropriate suggestions
        corpus_type = context.experiment_config.corpus.lower()
        if "synonym" in recommendation.synonym_suggestions:
            if corpus_type in ["ecommerce", "product", "retail"]:
                confidence += 0.1
                
        return min(1.0, max(0.1, confidence))

    def _assess_risk_level(self, recommendation: AnalysisChainRecommendation) -> str:
        """Assess risk level of the recommendation."""
        risk_factors = 0
        
        # Custom analyzers are higher risk
        if len(recommendation.custom_analyzer_configs) > 2:
            risk_factors += 2
        elif len(recommendation.custom_analyzer_configs) > 0:
            risk_factors += 1
            
        # Many tokenizer changes are risky
        if len(recommendation.tokenizer_suggestions) > 3:
            risk_factors += 1
            
        # Complex filter chains are risky
        total_filters = sum(len(filters) for filters in recommendation.filter_suggestions.values())
        if total_filters > 20:
            risk_factors += 2
        elif total_filters > 10:
            risk_factors += 1
            
        # Many synonym changes can be risky
        total_synonyms = sum(len(syns) for syns in recommendation.synonym_suggestions.values())
        if total_synonyms > 50:
            risk_factors += 1
            
        if risk_factors >= 4:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"

    def _calculate_priority(self, recommendation: AnalysisChainRecommendation, context: OptimizationContext) -> int:
        """Calculate priority score (1-10) for the recommendation."""
        priority = 6  # Base priority
        
        # Higher priority if current performance is poor
        current_score = context.current_metrics.get(context.experiment_config.primary_metric, 0)
        if current_score < 0.3:
            priority += 2
        elif current_score < 0.6:
            priority += 1
            
        # Higher priority for synonym improvements (often high impact)
        if len(recommendation.synonym_suggestions) > 0:
            priority += 1
            
        # Higher priority if we have custom analyzer suggestions
        if len(recommendation.custom_analyzer_configs) > 0:
            priority += 1
            
        # Lower priority if too many complex changes (risky)
        if len(recommendation.custom_analyzer_configs) > 3:
            priority -= 1
            
        # Adjust based on query complexity
        if context.experiment_config.queries:
            phrase_queries = sum(1 for q in context.experiment_config.queries if '"' in q)
            if phrase_queries > len(context.experiment_config.queries) * 0.3:
                # Many phrase queries - analysis chains are more important
                priority += 1
                
        return min(10, max(1, priority))
