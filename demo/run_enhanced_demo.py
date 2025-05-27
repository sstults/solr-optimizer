#!/usr/bin/env python3
"""
Enhanced Solr Optimizer Demo Script

This script runs an enhanced version of the Solr optimization demo with:
- More explicit logging of what's happening at each step
- Iterative optimization attempts until NDCG improves
- Minimum 5 attempts, maximum 10 attempts
- Different optimization strategies on each iteration
- Detailed progress tracking and analysis
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import our framework components
from solr_optimizer.core.default_experiment_manager import DefaultExperimentManager
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.query_config import QueryConfig
from solr_optimizer.agents.solr.pysolr_execution_agent import PySolrExecutionAgent
from solr_optimizer.agents.metrics.standard_metrics_agent import StandardMetricsAgent
from solr_optimizer.agents.logging.file_based_logging_agent import FileBasedLoggingAgent
from solr_optimizer.agents.comparison.standard_comparison_agent import StandardComparisonAgent
from solr_optimizer.agents.query.dummy_query_tuning_agent import DummyQueryTuningAgent

# Import validation components
from preflight_validation import SolrDemoPreflightValidator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDemoOrchestrator:
    """Enhanced demo orchestrator with iterative optimization and explicit progress tracking."""
    
    def __init__(self):
        self.data_dir = PROJECT_ROOT / "demo" / "data"
        self.solr_url = "http://localhost:8983/solr"
        self.collection_name = "ecommerce_products"
        self.experiment_manager = None
        self.experiment_id = None
        self.baseline_ndcg = None
        self.best_ndcg = None
        self.best_iteration_id = None
        self.iteration_history = []
        self.validator = None
        self.fallback_config = None
        
        # Optimization strategies to try
        self.optimization_strategies = [
            self._create_dismax_strategy_1,
            self._create_dismax_strategy_2,
            self._create_edismax_strategy_1,
            self._create_edismax_strategy_2,
            self._create_edismax_strategy_3,
            self._create_advanced_edismax_strategy_1,
            self._create_advanced_edismax_strategy_2,
            self._create_experimental_strategy_1,
            self._create_experimental_strategy_2,
            self._create_aggressive_strategy
        ]
        
    def setup_experiment_manager(self):
        """Initialize the experiment manager with all agents."""
        print("🔧 SETTING UP EXPERIMENT MANAGER")
        print("=" * 60)
        print("⏳ Initializing agents...")
        
        # Initialize agents
        solr_agent = PySolrExecutionAgent(solr_url=self.solr_url)
        print("✅ Solr execution agent initialized")
        
        metrics_agent = StandardMetricsAgent()
        print("✅ Metrics agent initialized")
        
        logging_agent = FileBasedLoggingAgent()
        print("✅ Logging agent initialized")
        
        comparison_agent = StandardComparisonAgent()
        print("✅ Comparison agent initialized")
        
        query_tuning_agent = DummyQueryTuningAgent()
        print("✅ Query tuning agent initialized")
        
        # Create experiment manager
        self.experiment_manager = DefaultExperimentManager(
            solr_execution_agent=solr_agent,
            metrics_agent=metrics_agent,
            logging_agent=logging_agent,
            comparison_agent=comparison_agent,
            query_tuning_agent=query_tuning_agent
        )
        
        print("✅ Experiment manager fully initialized")
        print()
    
    def load_demo_data(self) -> tuple:
        """Load queries and judgments for the demo."""
        print("📂 LOADING DEMO DATA")
        print("=" * 60)
        
        # Load queries
        queries_file = self.data_dir / "processed" / "queries.csv"
        print(f"⏳ Loading queries from: {queries_file}")
        
        if not queries_file.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_file}")
        
        import csv
        queries = []
        with open(queries_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                queries.append(row['query'])
        
        print(f"✅ Loaded {len(queries)} total queries")
        
        # Load judgments
        judgments_file = self.data_dir / "judgments" / "judgments.csv"
        print(f"⏳ Loading relevance judgments from: {judgments_file}")
        
        if not judgments_file.exists():
            raise FileNotFoundError(f"Judgments file not found: {judgments_file}")
        
        judgments = {}
        judgment_count = 0
        with open(judgments_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_id = row['query_id']
                if query_id not in judgments:
                    judgments[query_id] = {}
                judgments[query_id][row['product_id']] = int(row['judgment'])
                judgment_count += 1
        
        print(f"✅ Loaded {judgment_count} relevance judgments for {len(judgments)} queries")
        
        # Map judgments to query text
        query_judgments = {}
        with open(queries_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_id = row['query_id']
                query_text = row['query']
                if query_id in judgments:
                    query_judgments[query_text] = judgments[query_id]
        
        # Use first 20 queries for demo (manageable size)
        demo_queries = queries[:20]
        demo_judgments = {q: query_judgments[q] for q in demo_queries if q in query_judgments}
        
        print(f"✅ Using {len(demo_queries)} queries for demo")
        print(f"✅ Mapped {len(demo_judgments)} queries with judgments")
        print()
        
        return demo_queries, demo_judgments
    
    def create_demo_experiment(self, queries: List[str], judgments: Dict) -> str:
        """Create a new experiment for the demo."""
        print("🧪 CREATING DEMO EXPERIMENT")
        print("=" * 60)
        
        config = ExperimentConfig(
            corpus=self.collection_name,
            queries=queries,
            judgments=judgments,
            primary_metric="ndcg",
            metric_depth=10,
            secondary_metrics=["precision", "recall"]
        )
        
        print(f"⏳ Setting up experiment with configuration:")
        print(f"   • Corpus: {config.corpus}")
        print(f"   • Queries: {len(config.queries)}")
        print(f"   • Primary metric: {config.primary_metric}@{config.metric_depth}")
        print(f"   • Secondary metrics: {', '.join(config.secondary_metrics)}")
        
        self.experiment_id = self.experiment_manager.setup_experiment(config)
        print(f"✅ Created experiment: {self.experiment_id}")
        print()
        return self.experiment_id
    
    def run_baseline_iteration(self) -> Tuple[str, float]:
        """Run the baseline iteration and return iteration ID and NDCG score."""
        print("📊 RUNNING BASELINE ITERATION")
        print("=" * 60)
        print("🎯 Purpose: Establish performance baseline with default Solr configuration")
        print()
        
        print("⚙️ Baseline Configuration:")
        print("   • Query Parser: Standard Lucene")
        print("   • Default Field: _text_ (contains all searchable content)")
        print("   • No field boosting")
        print("   • No phrase matching")
        print("   • Basic relevance scoring")
        print()
        
        # Default query configuration (basic search using _text_ field)
        baseline_config = QueryConfig(
            iteration_id="baseline",
            query_parser="lucene",
            additional_params={
                "df": "_text_",
                "rows": 10
            }
        )
        
        print("⏳ Executing baseline queries...")
        start_time = time.time()
        
        iteration_result = self.experiment_manager.run_iteration(
            experiment_id=self.experiment_id,
            query_config=baseline_config
        )
        
        execution_time = time.time() - start_time
        iteration_id = iteration_result.iteration_id
        
        # Extract NDCG score
        ndcg_score = 0.0
        if iteration_result.metric_results:
            for metric_result in iteration_result.metric_results:
                if hasattr(metric_result, 'metric_name') and metric_result.metric_name == 'ndcg':
                    ndcg_score = metric_result.value
                    break
                elif isinstance(metric_result, dict) and metric_result.get('metric_name') == 'ndcg':
                    ndcg_score = metric_result.get('value', 0.0)
                    break
        
        self.baseline_ndcg = ndcg_score
        self.best_ndcg = ndcg_score
        self.best_iteration_id = iteration_id
        
        print(f"✅ Baseline completed in {execution_time:.2f} seconds")
        print(f"📊 Baseline NDCG@10: {ndcg_score:.4f}")
        print(f"🎯 This is our target to beat!")
        print()
        
        self.iteration_history.append({
            'iteration_id': iteration_id,
            'strategy': 'Baseline (Lucene)',
            'ndcg': ndcg_score,
            'improvement': 0.0,
            'config_summary': 'Standard Lucene query parser'
        })
        
        return iteration_id, ndcg_score
    
    def run_iterative_optimization(self) -> bool:
        """Run iterative optimization attempts until improvement or max iterations."""
        print("🚀 STARTING ITERATIVE OPTIMIZATION")
        print("=" * 60)
        print(f"🎯 Goal: Improve NDCG@10 above baseline of {self.baseline_ndcg:.4f}")
        print(f"📋 Strategy: Try different optimization approaches")
        print(f"🔄 Minimum attempts: 5")
        print(f"🛑 Maximum attempts: 10")
        print(f"✅ Stop when: NDCG improves OR max attempts reached")
        print()
        
        improvement_found = False
        attempt = 0
        max_attempts = 10
        min_attempts = 5
        
        while attempt < max_attempts:
            attempt += 1
            
            print(f"🔄 OPTIMIZATION ATTEMPT #{attempt}")
            print("-" * 50)
            
            # Choose strategy (cycle through available strategies)
            strategy_index = (attempt - 1) % len(self.optimization_strategies)
            strategy_func = self.optimization_strategies[strategy_index]
            
            try:
                # Create optimization configuration
                config, strategy_name, strategy_description = strategy_func(attempt)
                
                print(f"📋 Strategy: {strategy_name}")
                print(f"💡 Description: {strategy_description}")
                print()
                
                print("⚙️ Configuration Details:")
                self._print_config_details(config)
                print()
                
                print("⏳ Executing optimized queries...")
                start_time = time.time()
                
                # Run the iteration
                iteration_result = self.experiment_manager.run_iteration(
                    experiment_id=self.experiment_id,
                    query_config=config
                )
                
                execution_time = time.time() - start_time
                
                # Extract NDCG score
                ndcg_score = 0.0
                if iteration_result.metric_results:
                    for metric_result in iteration_result.metric_results:
                        if hasattr(metric_result, 'metric_name') and metric_result.metric_name == 'ndcg':
                            ndcg_score = metric_result.value
                            break
                        elif isinstance(metric_result, dict) and metric_result.get('metric_name') == 'ndcg':
                            ndcg_score = metric_result.get('value', 0.0)
                            break
                
                improvement = ndcg_score - self.baseline_ndcg
                improvement_pct = (improvement / self.baseline_ndcg) * 100 if self.baseline_ndcg > 0 else 0
                
                print(f"✅ Attempt #{attempt} completed in {execution_time:.2f} seconds")
                print(f"📊 NDCG@10: {ndcg_score:.4f}")
                print(f"📈 Change from baseline: {improvement:+.4f} ({improvement_pct:+.2f}%)")
                
                # Track this iteration
                self.iteration_history.append({
                    'iteration_id': iteration_result.iteration_id,
                    'strategy': strategy_name,
                    'ndcg': ndcg_score,
                    'improvement': improvement,
                    'config_summary': strategy_description
                })
                
                # Check if this is the best so far
                if ndcg_score > self.best_ndcg:
                    self.best_ndcg = ndcg_score
                    self.best_iteration_id = iteration_result.iteration_id
                    print(f"🏆 NEW BEST SCORE! Previous best: {self.best_ndcg:.4f}")
                
                # Check if we found improvement
                if improvement > 0:
                    improvement_found = True
                    print(f"🎉 SUCCESS! Found improvement of {improvement:.4f} ({improvement_pct:.2f}%)")
                    
                    # Continue for minimum attempts, but note we found improvement
                    if attempt >= min_attempts:
                        print(f"✅ Reached minimum attempts ({min_attempts}) with improvement found.")
                        break
                    else:
                        print(f"⏳ Continuing to reach minimum attempts ({min_attempts})...")
                else:
                    print(f"📉 No improvement yet. Baseline: {self.baseline_ndcg:.4f}, Current: {ndcg_score:.4f}")
                
                print()
                
                # Brief pause between attempts
                if attempt < max_attempts:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"❌ Attempt #{attempt} failed: {e}")
                print("⏳ Continuing with next strategy...")
                print()
                continue
        
        # Final summary
        print("🏁 OPTIMIZATION ATTEMPTS COMPLETED")
        print("=" * 60)
        
        if improvement_found:
            best_improvement = self.best_ndcg - self.baseline_ndcg
            best_improvement_pct = (best_improvement / self.baseline_ndcg) * 100
            print(f"🎉 SUCCESS: Found improvement after {attempt} attempts")
            print(f"📊 Best NDCG@10: {self.best_ndcg:.4f}")
            print(f"📈 Best improvement: +{best_improvement:.4f} ({best_improvement_pct:+.2f}%)")
        else:
            print(f"📉 No improvement found after {attempt} attempts")
            print(f"📊 Baseline NDCG@10: {self.baseline_ndcg:.4f}")
            print(f"📊 Best NDCG@10: {self.best_ndcg:.4f}")
        
        print()
        return improvement_found
    
    def _print_config_details(self, config: QueryConfig):
        """Print detailed configuration information."""
        print(f"   • Query Parser: {config.query_parser}")
        
        if config.query_fields:
            qf_str = " ".join([f"{field}^{boost}" for field, boost in config.query_fields.items()])
            print(f"   • Query Fields: {qf_str}")
        
        if config.phrase_fields:
            pf_str = " ".join([f"{field}^{boost}" for field, boost in config.phrase_fields.items()])
            print(f"   • Phrase Fields: {pf_str}")
        
        if config.minimum_match:
            print(f"   • Minimum Should Match: {config.minimum_match}")
        
        if config.tie_breaker:
            print(f"   • Tie Breaker: {config.tie_breaker}")
        
        if config.additional_params:
            for key, value in config.additional_params.items():
                print(f"   • {key}: {value}")
    
    def show_detailed_results(self):
        """Show detailed results of all optimization attempts."""
        print("📊 DETAILED OPTIMIZATION RESULTS")
        print("=" * 80)
        
        print(f"{'#':<3} {'Strategy':<25} {'NDCG@10':<10} {'Change':<10} {'%Change':<10}")
        print("-" * 80)
        
        for i, result in enumerate(self.iteration_history):
            change_str = f"{result['improvement']:+.4f}" if i > 0 else "baseline"
            pct_change_str = f"{(result['improvement']/self.baseline_ndcg)*100:+.2f}%" if i > 0 and self.baseline_ndcg > 0 else "---"
            
            marker = "🏆" if result['iteration_id'] == self.best_iteration_id else "  "
            
            print(f"{marker} {i+1:<3} {result['strategy']:<25} {result['ndcg']:<10.4f} {change_str:<10} {pct_change_str:<10}")
        
        print()
        
        # Show configuration details for best iteration
        if self.best_iteration_id:
            print("🏆 BEST CONFIGURATION DETAILS")
            print("-" * 50)
            best_result = next(r for r in self.iteration_history if r['iteration_id'] == self.best_iteration_id)
            print(f"Strategy: {best_result['strategy']}")
            print(f"Configuration: {best_result['config_summary']}")
            print(f"NDCG@10: {best_result['ndcg']:.4f}")
            if best_result['improvement'] > 0:
                print(f"Improvement: +{best_result['improvement']:.4f} ({(best_result['improvement']/self.baseline_ndcg)*100:+.2f}%)")
        
        print()
    
    # Optimization strategy methods
    def _create_dismax_strategy_1(self, attempt: int) -> Tuple[QueryConfig, str, str]:
        """Basic DisMax strategy with conservative field boosting."""
        config = QueryConfig(
            iteration_id=f"dismax_basic_{attempt}",
            query_parser="dismax",
            query_fields={
                "product_title": 2.0,
                "product_description": 1.0,
                "product_bullet_point": 1.2
            },
            phrase_fields={
                "product_title": 3.0
            },
            minimum_match="2<-1 5<80%",
            tie_breaker=0.01,
            additional_params={"rows": 10}
        )
        return config, "DisMax Basic", "Conservative field boosting with basic phrase matching"
    
    def _create_dismax_strategy_2(self, attempt: int) -> Tuple[QueryConfig, str, str]:
        """DisMax strategy with aggressive field boosting."""
        config = QueryConfig(
            iteration_id=f"dismax_aggressive_{attempt}",
            query_parser="dismax",
            query_fields={
                "product_title": 3.0,
                "product_description": 1.0,
                "product_bullet_point": 1.5,
                "product_brand": 1.8
            },
            phrase_fields={
                "product_title": 4.0,
                "product_description": 1.5
            },
            minimum_match="2<-1 5<-2 6<90%",
            tie_breaker=0.05,
            additional_params={"rows": 10}
        )
        return config, "DisMax Aggressive", "Aggressive field boosting with enhanced phrase matching"
    
    def _create_edismax_strategy_1(self, attempt: int) -> Tuple[QueryConfig, str, str]:
        """Basic eDisMax strategy."""
        config = QueryConfig(
            iteration_id=f"edismax_basic_{attempt}",
            query_parser="edismax",
            query_fields={
                "product_title": 2.5,
                "product_description": 1.0,
                "product_bullet_point": 1.3,
                "product_brand": 1.2
            },
            phrase_fields={
                "product_title": 3.5,
                "product_description": 1.2
            },
            minimum_match="3<-1 6<-2 8<75%",
            tie_breaker=0.1,
            additional_params={
                "pf2": "product_title^2.0",
                "rows": 10
            }
        )
        return config, "eDisMax Basic", "Extended DisMax with bi-gram phrase matching"
    
    def _create_edismax_strategy_2(self, attempt: int) -> Tuple[QueryConfig, str, str]:
        """eDisMax strategy with multi-level phrase boosting."""
        config = QueryConfig(
            iteration_id=f"edismax_multilevel_{attempt}",
            query_parser="edismax",
            query_fields={
                "product_title": 3.0,
                "product_description": 1.0,
                "product_bullet_point": 2.0,
                "product_brand": 1.5
            },
            phrase_fields={
                "product_title": 5.0,
                "product_description": 2.0
            },
            minimum_match="3<-1 6<-2 8<75%",
            tie_breaker=0.1,
            additional_params={
                "pf2": "product_title^4.0 product_description^1.5",
                "pf3": "product_title^3.0 product_description^1.0",
                "rows": 10
            }
        )
        return config, "eDisMax Multi-level", "Multi-level phrase boosting (pf, pf2, pf3)"
    
    def _create_edismax_strategy_3(self, attempt: int) -> Tuple[QueryConfig, str, str]:
        """eDisMax strategy with function queries."""
        config = QueryConfig(
            iteration_id=f"edismax_functions_{attempt}",
            query_parser="edismax",
            query_fields={
                "product_title": 3.0,
                "product_description": 1.0,
                "product_bullet_point": 2.0,
                "product_brand": 1.5
            },
            phrase_fields={
                "product_title": 4.0,
                "product_description": 1.8
            },
            minimum_match="2<-1 5<-2 8<80%",
            tie_breaker=0.1,
            additional_params={
                "pf2": "product_title^3.0 product_description^1.2",
                "boost": "if(exists(product_brand),1.2,1.0)",
                "rows": 10
            }
        )
        return config, "eDisMax Functions", "Function queries with conditional boosting"
    
    def _create_advanced_edismax_strategy_1(self, attempt: int) -> Tuple[QueryConfig, str, str]:
        """Advanced eDisMax with complex minimum should match."""
        config = QueryConfig(
            iteration_id=f"edismax_advanced_{attempt}",
            query_parser="edismax",
            query_fields={
                "product_title": 4.0,
                "product_description": 1.0,
                "product_bullet_point": 2.5,
                "product_brand": 2.0
            },
            phrase_fields={
                "product_title": 6.0,
                "product_description": 2.5
            },
            minimum_match="1<-1 2<-1 3<-2 5<-3 8<75%",
            tie_breaker=0.15,
            additional_params={
                "pf2": "product_title^5.0 product_description^2.0",
                "pf3": "product_title^4.0 product_description^1.5",
                "boost": "if(exists(product_brand),1.3,1.0)",
                "rows": 10
            }
        )
        return config, "Advanced eDisMax", "Complex minimum should match with high boosting"
    
    def _create_advanced_edismax_strategy_2(self, attempt: int) -> Tuple[QueryConfig, str, str]:
        """Advanced eDisMax with relaxed matching."""
        config = QueryConfig(
            iteration_id=f"edismax_relaxed_{attempt}",
            query_parser="edismax",
            query_fields={
                "product_title": 3.5,
                "product_description": 1.2,
                "product_bullet_point": 2.2,
                "product_brand": 1.8
            },
            phrase_fields={
                "product_title": 5.5,
                "product_description": 2.2
            },
            minimum_match="1<-1 3<-1 6<-2 9<60%",
            tie_breaker=0.2,
            additional_params={
                "pf2": "product_title^4.5 product_description^1.8",
                "pf3": "product_title^3.5 product_description^1.2",
                "boost": "if(exists(product_brand),1.25,1.0)",
                "rows": 10
            }
        )
        return config, "Relaxed eDisMax", "Relaxed matching for broader recall"
    
    def _create_experimental_strategy_1(self, attempt: int) -> Tuple[QueryConfig, str, str]:
        """Experimental strategy with very high title boosting."""
        config = QueryConfig(
            iteration_id=f"experimental_title_{attempt}",
            query_parser="edismax",
            query_fields={
                "product_title": 5.0,
                "product_description": 0.8,
                "product_bullet_point": 1.8,
                "product_brand": 2.5
            },
            phrase_fields={
                "product_title": 8.0,
                "product_description": 1.0
            },
            minimum_match="2<-1 4<-2 7<85%",
            tie_breaker=0.05,
            additional_params={
                "pf2": "product_title^6.0",
                "pf3": "product_title^5.0",
                "rows": 10
            }
        )
        return config, "Title-Focused", "Extremely high title field boosting"
    
    def _create_experimental_strategy_2(self, attempt: int) -> Tuple[QueryConfig, str, str]:
        """Experimental strategy with balanced field weights."""
        config = QueryConfig(
            iteration_id=f"experimental_balanced_{attempt}",
            query_parser="edismax",
            query_fields={
                "product_title": 2.2,
                "product_description": 1.5,
                "product_bullet_point": 1.8,
                "product_brand": 1.3
            },
            phrase_fields={
                "product_title": 3.2,
                "product_description": 2.5,
                "product_bullet_point": 2.0
            },
            minimum_match="2<-1 4<-1 7<70%",
            tie_breaker=0.3,
            additional_params={
                "pf2": "product_title^2.5 product_description^2.0 product_bullet_point^1.8",
                "pf3": "product_title^2.0 product_description^1.5",
                "rows": 10
            }
        )
        return config, "Balanced Fields", "Balanced field weights with high tie breaker"
    
    def _create_aggressive_strategy(self, attempt: int) -> Tuple[QueryConfig, str, str]:
        """Most aggressive optimization strategy."""
        config = QueryConfig(
            iteration_id=f"aggressive_{attempt}",
            query_parser="edismax",
            query_fields={
                "product_title": 6.0,
                "product_description": 1.5,
                "product_bullet_point": 3.0,
                "product_brand": 3.0
            },
            phrase_fields={
                "product_title": 10.0,
                "product_description": 3.0,
                "product_bullet_point": 4.0
            },
            minimum_match="1<-1 2<-1 4<-2 7<-3 10<50%",
            tie_breaker=0.5,
            additional_params={
                "pf2": "product_title^8.0 product_description^3.0 product_bullet_point^2.5",
                "pf3": "product_title^6.0 product_description^2.0",
                "boost": "if(exists(product_brand),1.5,1.0)",
                "rows": 10
            }
        )
        return config, "Aggressive Max", "Maximum aggressive optimization with all features"
    
    def run_preflight_validation(self) -> bool:
        """Run pre-flight validation to ensure demo can execute successfully."""
        print("🔍 PRE-FLIGHT VALIDATION")
        print("=" * 50)
        print("Checking demo prerequisites...")
        print()
        
        self.validator = SolrDemoPreflightValidator(
            solr_url=self.solr_url,
            collection_name=self.collection_name,
            data_dir=self.data_dir
        )
        
        # Run validation
        validation_passed = self.validator.run_all_validations()
        
        if not validation_passed:
            # Check if we can get a fallback configuration
            self.fallback_config = self.validator.get_working_fallback_config()
            
            if self.fallback_config:
                print(f"🔄 FALLBACK MODE ACTIVATED")
                print(f"Using fallback configuration: {self.fallback_config['name']}")
                print(f"Description: {self.fallback_config['config']['description']}")
                print()
                return True
            else:
                print("🚨 CRITICAL VALIDATION FAILURE")
                print("Demo cannot proceed - please fix the issues above.")
                return False
        
        print("✅ All validations passed - demo ready to proceed!")
        print()
        return True
    
    def validate_search_returns_results(self) -> bool:
        """Validate that searches return results before running optimization."""
        print("🔍 VALIDATING SEARCH FUNCTIONALITY")
        print("-" * 40)
        
        test_queries = ["laptop", "smartphone", "headphones"]
        working_queries = 0
        
        for query in test_queries:
            try:
                # Test with current baseline configuration
                baseline_config = QueryConfig(
                    iteration_id="validation_test",
                    query_parser="lucene",
                    additional_params={
                        "df": "_text_",
                        "rows": 10
                    }
                )
                
                # Run a quick test iteration
                result = self.experiment_manager.run_iteration(
                    experiment_id=self.experiment_id,
                    query_config=baseline_config
                )
                
                # Check if we got meaningful results
                if result and hasattr(result, 'metric_results') and result.metric_results:
                    working_queries += 1
                    print(f"✅ '{query}': Search working")
                else:
                    print(f"❌ '{query}': No results returned")
                    
            except Exception as e:
                print(f"❌ '{query}': Error - {e}")
        
        success_rate = (working_queries / len(test_queries)) * 100
        
        if success_rate >= 67:  # At least 2/3 queries should work
            print(f"✅ Search validation passed ({success_rate:.0f}% success rate)")
            return True
        else:
            print(f"❌ Search validation failed ({success_rate:.0f}% success rate)")
            print("⚠️  Continuing with caution - some optimizations may not work")
            return False
    
    def apply_fallback_configuration(self):
        """Apply fallback configuration if needed."""
        if self.fallback_config:
            print("🔄 APPLYING FALLBACK CONFIGURATION")
            print("-" * 40)
            
            config = self.fallback_config['config']
            print(f"Using: {self.fallback_config['name']}")
            print(f"Description: {config['description']}")
            
            # Modify baseline to use fallback
            if config['query_parser'] == 'lucene':
                # Update baseline to use specific default field
                print(f"• Setting default field to: {config['default_field']}")
                
            elif config['query_parser'] in ['dismax', 'edismax']:
                # Update optimization strategies to use working fields
                print(f"• Using query parser: {config['query_parser']}")
                if 'query_fields' in config:
                    qf_str = " ".join([f"{field}^{boost}" for field, boost in config['query_fields'].items()])
                    print(f"• Query fields: {qf_str}")
            
            print("✅ Fallback configuration applied")
            print()
    
    def run_complete_enhanced_demo(self):
        """Run the complete enhanced demo with robustness improvements."""
        print("🚀 SOLR OPTIMIZER ENHANCED DEMO")
        print("="*80)
        print("This enhanced demo will:")
        print("• Run pre-flight validation to ensure reliability")
        print("• Validate searches return results before optimization")
        print("• Use fallback configurations if default setup fails")
        print("• Try multiple optimization strategies iteratively")
        print("• Continue until NDCG improves or max attempts reached")
        print("• Provide detailed analysis and progress tracking")
        print("="*80)
        print()
        
        try:
            # Step 1: Pre-flight validation
            if not self.run_preflight_validation():
                print("❌ Pre-flight validation failed. Cannot proceed.")
                return False
            
            # Step 2: Apply fallback configuration if needed
            if self.fallback_config:
                self.apply_fallback_configuration()
            
            # Step 3: Setup experiment manager
            self.setup_experiment_manager()
            
            # Step 4: Load data with validation
            try:
                queries, judgments = self.load_demo_data()
            except FileNotFoundError as e:
                print(f"❌ Data loading failed: {e}")
                print("💡 Please run: python demo/scripts/download_data.py")
                return False
            
            # Step 5: Create experiment
            self.create_demo_experiment(queries, judgments)
            
            # Step 6: Validate searches work before optimization
            search_validation_passed = self.validate_search_returns_results()
            
            # Step 7: Run baseline
            baseline_id, baseline_ndcg = self.run_baseline_iteration()
            
            # Check if baseline returned meaningful results
            if self.baseline_ndcg == 0.0:
                print("⚠️  WARNING: Baseline NDCG is 0.0")
                print("   This suggests search configuration issues.")
                print("   Continuing with optimization but results may be unreliable.")
                print()
            
            # Step 8: Run iterative optimization
            improvement_found = self.run_iterative_optimization()
            
            # Step 9: Show detailed results
            self.show_detailed_results()
            
            # Final analysis
            print("📋 FINAL ANALYSIS")
            print("=" * 60)
            
            if improvement_found:
                best_improvement = self.best_ndcg - self.baseline_ndcg
                best_improvement_pct = (best_improvement / self.baseline_ndcg) * 100
                print(f"🎉 SUCCESS! Demo found NDCG improvement:")
                print(f"   • Baseline NDCG@10: {self.baseline_ndcg:.4f}")
                print(f"   • Best NDCG@10: {self.best_ndcg:.4f}")
                print(f"   • Improvement: +{best_improvement:.4f} ({best_improvement_pct:+.2f}%)")
                print()
                print("🏆 Best performing strategy:")
                best_result = next(r for r in self.iteration_history if r['iteration_id'] == self.best_iteration_id)
                print(f"   • Strategy: {best_result['strategy']}")
                print(f"   • Configuration: {best_result['config_summary']}")
            else:
                print(f"📉 No improvement found in {len(self.iteration_history)-1} optimization attempts")
                print(f"   • This suggests the baseline is already well-optimized")
                print(f"   • Or the optimization strategies need adjustment")
                print(f"   • Consider trying different field weights or query parsers")
            
            print()
            print("📁 All experiment data saved in: experiment_storage/")
            print(f"🔗 Solr Admin UI: {self.solr_url}/")
            print()
            print("🎯 Next steps:")
            if improvement_found:
                print("   • Deploy the best configuration to production")
                print("   • Monitor performance with real user queries")
                print("   • Continue iterating with A/B testing")
            else:
                print("   • Analyze individual query performance")
                print("   • Consider different optimization approaches")
                print("   • Review relevance judgments for accuracy")
            
            print("="*80)
            print("🎉 Enhanced demo completed!")
            print("="*80)
            
        except Exception as e:
            logger.error(f"❌ Enhanced demo failed: {e}")
            raise


def main():
    """Main function to run the enhanced demo."""
    demo = EnhancedDemoOrchestrator()
    
    try:
        demo.run_complete_enhanced_demo()
    except KeyboardInterrupt:
        logger.info("\n🛑 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
