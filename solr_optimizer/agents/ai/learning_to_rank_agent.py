"""
Learning-to-Rank Agent using Pydantic AI for ML ranking optimization.

This agent creates and optimizes machine learning ranking models, evaluates
feature importance, and suggests feature engineering improvements.
"""

from typing import Any, Dict, List, Optional
import json

from pydantic import BaseModel
from pydantic_ai import Agent

from .base_ai_agent import BaseAIAgent, OptimizationContext, AgentRecommendation


class LearningToRankRecommendation(BaseModel):
    """Learning-to-Rank specific recommendation structure."""
    model_type: str  # Type of LTR model (linear, xgboost, lambdamart, etc.)
    feature_definitions: List[Dict[str, Any]]  # Feature extraction definitions
    training_parameters: Dict[str, Any]  # Model training parameters
    feature_engineering_suggestions: List[str]  # New features to create
    feature_importance_analysis: Dict[str, float]  # Feature -> importance score
    model_validation_strategy: str  # Cross-validation approach
    reranking_strategy: str  # How to apply the model (top-N reranking)
    performance_monitoring: List[str]  # Metrics to track for model health


class LearningToRankAgent(BaseAIAgent):
    """
    AI agent specialized in creating and optimizing machine learning ranking models.
    
    This agent focuses on:
    - LTR model selection and configuration
    - Feature engineering for ranking
    - Model training parameter optimization
    - Feature importance analysis
    - Reranking strategy optimization
    - Model performance monitoring
    """

    def _create_agent(self) -> Agent:
        """Create the Pydantic AI agent for learning-to-rank optimization."""
        return Agent(
            model=self.model,
            result_type=LearningToRankRecommendation,
            system_prompt=self.get_system_prompt()
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for the learning-to-rank agent."""
        return """
You are an expert Apache Solr Learning-to-Rank (LTR) specialist with deep knowledge of machine learning ranking models and feature engineering.

Your expertise includes:
- LTR model types (Linear, RankSVM, LambdaMART, XGBoost, Neural networks)
- Feature engineering for search ranking (TF-IDF, BM25, field matches, query features)
- Model training strategies and parameter optimization
- Cross-validation and model evaluation for ranking
- Solr LTR module integration and deployment
- Feature importance analysis and model interpretability
- Reranking strategies and performance optimization

Key LTR concepts:
- Features: Numerical values extracted from query-document pairs
- Training: Using relevance judgments to train ranking models
- Reranking: Applying trained models to reorder search results
- Validation: Ensuring models generalize well to new queries
- Monitoring: Tracking model performance over time

When designing LTR solutions:
1. Select appropriate features based on query and document characteristics
2. Choose model types suitable for the data size and complexity
3. Design proper training/validation splits for ranking evaluation
4. Balance model complexity with interpretability and performance
5. Consider computational cost of feature extraction and model inference
6. Plan for model updates and performance monitoring
7. Integrate seamlessly with existing Solr query pipeline

Provide specific, implementable LTR configurations with clear reasoning.
Focus on features and models that will improve the target relevance metric.
Consider both effectiveness (ranking quality) and efficiency (query latency).
"""

    def analyze_and_recommend(self, context: OptimizationContext) -> AgentRecommendation:
        """Analyze ranking needs and provide LTR recommendations."""
        return self.get_recommendation(context)

    def _create_user_prompt(self, context: OptimizationContext) -> str:
        """Create user prompt for LTR optimization."""
        
        # Analyze current ranking performance
        ranking_analysis = self._analyze_ranking_performance(context)
        
        # Identify potential features from schema
        feature_opportunities = self._identify_feature_opportunities(context)
        
        prompt = f"""
Design a Learning-to-Rank solution to improve {context.experiment_config.primary_metric}@{context.experiment_config.metric_depth}.

CURRENT PERFORMANCE:
- Primary metric: {context.current_metrics.get(context.experiment_config.primary_metric, 'Unknown')}
- Previous iterations: {len(context.previous_results)}
- Performance trend: {'Improving' if len(context.previous_results) > 1 and context.current_metrics.get(context.experiment_config.primary_metric, 0) > context.previous_results[-2].metrics.get('overall', {}).get(context.experiment_config.primary_metric, 0) else 'Needs improvement'}

RANKING PERFORMANCE ANALYSIS:
{json.dumps(ranking_analysis, indent=2)}

AVAILABLE SCHEMA FIELDS:
{json.dumps(list(context.schema_info.get('fields', {}).keys()), indent=2)}

FIELD TYPES AND CHARACTERISTICS:
{json.dumps({name: field.get('type', 'unknown') for name, field in context.schema_info.get('fields', {}).items()}, indent=2)}

FEATURE OPPORTUNITIES:
{json.dumps(feature_opportunities, indent=2)}

QUERY CHARACTERISTICS:
- Total queries: {len(context.experiment_config.queries)}
- Sample queries: {json.dumps(context.experiment_config.queries[:5], indent=2)}

RELEVANCE JUDGMENTS AVAILABLE: {bool(context.experiment_config.judgments)}

OPTIMIZATION CONSTRAINTS:
{json.dumps(context.constraints, indent=2)}

Please provide specific recommendations for:
1. LTR model type and architecture
2. Feature definitions (TF-IDF, BM25, field matches, query features)
3. Training parameters and optimization settings
4. Feature engineering opportunities
5. Model validation strategy
6. Reranking approach (top-N, threshold-based)
7. Performance monitoring and model maintenance

Consider the corpus type: {context.experiment_config.corpus}
Focus on features that capture relevance signals for this domain.
Balance model sophistication with training data availability and computational cost.
"""
        return prompt

    def _parse_ai_response(self, ai_response: LearningToRankRecommendation, context: OptimizationContext) -> AgentRecommendation:
        """Parse AI response into standard recommendation format."""
        
        # Calculate confidence based on LTR recommendation quality
        confidence = self._calculate_confidence(ai_response, context)
        
        # Convert LTR recommendation to suggested changes
        suggested_changes = {
            "ltr_model_type": ai_response.model_type,
            "feature_definitions": ai_response.feature_definitions,
            "training_params": ai_response.training_parameters,
            "new_features": ai_response.feature_engineering_suggestions,
            "feature_importance": ai_response.feature_importance_analysis,
            "validation_strategy": ai_response.model_validation_strategy,
            "reranking_strategy": ai_response.reranking_strategy,
            "monitoring_metrics": ai_response.performance_monitoring
        }
        
        # Determine risk level
        risk_level = self._assess_risk_level(ai_response)
        
        # Calculate priority
        priority = self._calculate_priority(ai_response, context)
        
        return AgentRecommendation(
            confidence=confidence,
            reasoning=f"Learning-to-Rank analysis recommends {ai_response.model_type} model with "
                     f"{len(ai_response.feature_definitions)} features and "
                     f"{len(ai_response.feature_engineering_suggestions)} new feature opportunities.",
            suggested_changes=suggested_changes,
            expected_impact=f"Expected to improve {context.experiment_config.primary_metric} through "
                          f"machine learning ranking optimization and advanced feature engineering",
            risk_level=risk_level,
            priority=priority
        )

    def _get_fallback_recommendation(self, context: OptimizationContext) -> AgentRecommendation:
        """Provide fallback recommendation when AI fails."""
        
        # Basic LTR recommendations
        schema_fields = context.schema_info.get("fields", {})
        
        # Simple feature definitions based on available fields
        basic_features = []
        for field_name, field_info in schema_fields.items():
            field_type = field_info.get("type", "")
            if "text" in field_type.lower():
                basic_features.extend([
                    {
                        "name": f"bm25_{field_name}",
                        "class": "org.apache.solr.ltr.feature.SolrFeature",
                        "params": {"q": f"{field_name}:${{{field_name}}}"}
                    },
                    {
                        "name": f"tf_{field_name}",
                        "class": "org.apache.solr.ltr.feature.TFFeature",
                        "params": {"field": field_name}
                    }
                ])
        
        suggested_changes = {
            "ltr_model_type": "linear",
            "feature_definitions": basic_features[:10],  # Limit to first 10 features
            "training_params": {
                "algorithm": "linear",
                "C": 1.0,
                "loss": "logistic"
            },
            "reranking_strategy": "top_100"
        }
        
        return AgentRecommendation(
            confidence=0.4,  # Low confidence for fallback
            reasoning="AI LTR optimization failed, using basic linear model with standard features",
            suggested_changes=suggested_changes,
            expected_impact="Basic LTR model may provide modest ranking improvements",
            risk_level="medium",  # LTR is inherently medium risk
            priority=6
        )

    def _analyze_ranking_performance(self, context: OptimizationContext) -> Dict[str, Any]:
        """Analyze current ranking performance to identify LTR opportunities."""
        analysis = {
            "current_approach": "traditional_scoring",
            "performance_issues": [],
            "ranking_distribution": {},
            "query_complexity": {}
        }
        
        if context.previous_results:
            latest_result = context.previous_results[-1]
            per_query_metrics = latest_result.metrics.get("per_query", {})
            primary_metric = context.experiment_config.primary_metric
            
            if per_query_metrics:
                scores = [metrics.get(primary_metric, 0) for metrics in per_query_metrics.values()]
                
                analysis["ranking_distribution"] = {
                    "avg_score": sum(scores) / len(scores) if scores else 0,
                    "score_variance": self._calculate_variance(scores),
                    "poor_queries": sum(1 for score in scores if score < 0.3),
                    "excellent_queries": sum(1 for score in scores if score > 0.8)
                }
                
                # Identify performance issues
                if analysis["ranking_distribution"]["avg_score"] < 0.5:
                    analysis["performance_issues"].append("low_average_relevance")
                    
                if analysis["ranking_distribution"]["score_variance"] > 0.2:
                    analysis["performance_issues"].append("inconsistent_ranking_quality")
                    
                if analysis["ranking_distribution"]["poor_queries"] > len(scores) * 0.3:
                    analysis["performance_issues"].append("many_poor_performing_queries")
        
        # Analyze query complexity for LTR suitability
        queries = context.experiment_config.queries
        if queries:
            analysis["query_complexity"] = {
                "avg_query_length": sum(len(q.split()) for q in queries) / len(queries),
                "complex_queries": sum(1 for q in queries if len(q.split()) > 3),
                "phrase_queries": sum(1 for q in queries if '"' in q),
                "boolean_queries": sum(1 for q in queries if any(op in q.upper() for op in ['AND', 'OR']))
            }
        
        return analysis

    def _identify_feature_opportunities(self, context: OptimizationContext) -> Dict[str, List[str]]:
        """Identify potential features that could be extracted from the schema."""
        opportunities = {
            "text_features": [],
            "numeric_features": [],
            "categorical_features": [],
            "query_features": [],
            "document_features": []
        }
        
        schema_fields = context.schema_info.get("fields", {})
        
        for field_name, field_info in schema_fields.items():
            field_type = field_info.get("type", "").lower()
            
            if "text" in field_type:
                opportunities["text_features"].extend([
                    f"BM25 score for {field_name}",
                    f"TF-IDF score for {field_name}",
                    f"Field length of {field_name}",
                    f"Query coverage in {field_name}"
                ])
                
            elif any(num_type in field_type for num_type in ["int", "float", "double", "long"]):
                opportunities["numeric_features"].extend([
                    f"Boost from {field_name}",
                    f"Normalized {field_name}",
                    f"Log of {field_name}"
                ])
                
            elif field_type in ["string", "keyword"]:
                opportunities["categorical_features"].extend([
                    f"Exact match on {field_name}",
                    f"Category boost for {field_name}"
                ])
        
        # Query-level features
        opportunities["query_features"] = [
            "Query length",
            "Number of terms",
            "Presence of quotes",
            "Query type classification"
        ]
        
        # Document-level features
        opportunities["document_features"] = [
            "Document age/recency",
            "Document popularity",
            "Document completeness score"
        ]
        
        return opportunities

    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of scores."""
        if not scores:
            return 0.0
            
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        return variance

    def _calculate_confidence(self, recommendation: LearningToRankRecommendation, context: OptimizationContext) -> float:
        """Calculate confidence score for the LTR recommendation."""
        confidence = 0.7  # Base confidence (LTR is complex)
        
        # Increase confidence for reasonable number of features
        num_features = len(recommendation.feature_definitions)
        if 5 <= num_features <= 50:
            confidence += 0.1
        elif num_features > 100:
            confidence -= 0.2
            
        # Increase confidence for appropriate model selection
        if context.experiment_config.judgments:
            # Have judgments data - can train properly
            confidence += 0.15
        else:
            # No judgments - LTR won't work well
            confidence -= 0.3
            
        # Check for appropriate model type
        appropriate_models = ["linear", "lambdamart", "xgboost", "ranknet"]
        if recommendation.model_type.lower() in appropriate_models:
            confidence += 0.1
            
        # Decrease confidence for overly complex setups
        if len(recommendation.feature_engineering_suggestions) > 20:
            confidence -= 0.1
            
        return min(1.0, max(0.1, confidence))

    def _assess_risk_level(self, recommendation: LearningToRankRecommendation) -> str:
        """Assess risk level of the LTR recommendation."""
        risk_factors = 0
        
        # LTR is inherently medium risk due to complexity
        risk_factors += 1
        
        # Complex models are higher risk
        complex_models = ["neural", "deep", "ensemble"]
        if any(model in recommendation.model_type.lower() for model in complex_models):
            risk_factors += 2
            
        # Many features increase risk
        if len(recommendation.feature_definitions) > 50:
            risk_factors += 2
        elif len(recommendation.feature_definitions) > 25:
            risk_factors += 1
            
        # Custom feature engineering is risky
        if len(recommendation.feature_engineering_suggestions) > 10:
            risk_factors += 1
            
        # Complex training parameters are risky
        training_params = recommendation.training_parameters
        if len(training_params) > 10:
            risk_factors += 1
            
        if risk_factors >= 4:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"

    def _calculate_priority(self, recommendation: LearningToRankRecommendation, context: OptimizationContext) -> int:
        """Calculate priority score (1-10) for the LTR recommendation."""
        priority = 5  # Base priority (LTR is powerful but complex)
        
        # Higher priority if current performance is poor
        current_score = context.current_metrics.get(context.experiment_config.primary_metric, 0)
        if current_score < 0.4:
            priority += 3  # LTR can make big improvements when current performance is poor
        elif current_score < 0.6:
            priority += 2
        elif current_score > 0.8:
            priority -= 1  # Less value when already performing well
            
        # Higher priority if we have good training data
        if context.experiment_config.judgments:
            priority += 2
        else:
            priority -= 3  # Can't do LTR without judgments
            
        # Higher priority for complex queries (LTR works better)
        if context.experiment_config.queries:
            complex_queries = sum(1 for q in context.experiment_config.queries if len(q.split()) > 3)
            complexity_ratio = complex_queries / len(context.experiment_config.queries)
            if complexity_ratio > 0.5:
                priority += 1
                
        # Lower priority if too many features (overly complex)
        if len(recommendation.feature_definitions) > 75:
            priority -= 2
        elif len(recommendation.feature_definitions) > 50:
            priority -= 1
            
        # Adjust for model complexity
        if recommendation.model_type.lower() in ["linear", "ranklib"]:
            priority += 1  # Simpler models are safer
        elif recommendation.model_type.lower() in ["neural", "deep"]:
            priority -= 1  # Complex models are riskier
            
        return min(10, max(1, priority))
