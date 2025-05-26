#!/usr/bin/env python3
"""
Complete Solr Optimizer Demo Script

This script runs a complete end-to-end demonstration of the Solr optimization framework,
showcasing AI-powered query optimization with real metrics and explanations.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteDemoOrchestrator:
    """Orchestrates the complete Solr optimization demo."""
    
    def __init__(self):
        self.data_dir = PROJECT_ROOT / "demo" / "data"
        self.solr_url = "http://localhost:8983/solr"
        self.collection_name = "ecommerce_products"
        self.experiment_manager = None
        self.experiment_id = None
        
    def setup_experiment_manager(self):
        """Initialize the experiment manager with all agents."""
        logger.info("🔧 Setting up experiment manager...")
        
        # Initialize agents
        solr_agent = PySolrExecutionAgent(base_url=self.solr_url)
        metrics_agent = StandardMetricsAgent()
        logging_agent = FileBasedLoggingAgent()
        comparison_agent = StandardComparisonAgent()
        query_tuning_agent = DummyQueryTuningAgent()
        
        # Create experiment manager
        self.experiment_manager = DefaultExperimentManager(
            solr_execution_agent=solr_agent,
            metrics_agent=metrics_agent,
            logging_agent=logging_agent,
            comparison_agent=comparison_agent,
            query_tuning_agent=query_tuning_agent
        )
        
        logger.info("✅ Experiment manager initialized")
    
    def load_demo_data(self) -> tuple:
        """Load queries and judgments for the demo."""
        logger.info("📂 Loading demo data...")
        
        # Load queries
        queries_file = self.data_dir / "processed" / "queries.csv"
        if not queries_file.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_file}")
        
        import csv
        queries = []
        with open(queries_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                queries.append(row['query'])
        
        # Load judgments
        judgments_file = self.data_dir / "judgments" / "judgments.csv"
        if not judgments_file.exists():
            raise FileNotFoundError(f"Judgments file not found: {judgments_file}")
        
        judgments = {}
        with open(judgments_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_id = row['query_id']
                if query_id not in judgments:
                    judgments[query_id] = {}
                judgments[query_id][row['product_id']] = int(row['judgment'])
        
        # Map judgments to query text
        query_judgments = {}
        with open(queries_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_id = row['query_id']
                query_text = row['query']
                if query_id in judgments:
                    query_judgments[query_text] = judgments[query_id]
        
        logger.info(f"✅ Loaded {len(queries)} queries and judgments for {len(query_judgments)} queries")
        return queries[:20], query_judgments  # Use first 20 queries for demo
    
    def create_demo_experiment(self, queries: List[str], judgments: Dict) -> str:
        """Create a new experiment for the demo."""
        logger.info("🧪 Creating demo experiment...")
        
        config = ExperimentConfig(
            corpus=self.collection_name,
            queries=queries,
            judgments=judgments,
            primary_metric="ndcg",
            metric_depth=10,
            secondary_metrics=["precision", "recall"]
        )
        
        self.experiment_id = self.experiment_manager.setup_experiment(config)
        logger.info(f"✅ Created experiment: {self.experiment_id}")
        return self.experiment_id
    
    def run_baseline_iteration(self) -> str:
        """Run the baseline iteration with default Solr configuration."""
        logger.info("📊 Running baseline iteration...")
        
        # Default query configuration (basic search)
        baseline_config = QueryConfig(
            request_handler="/select",
            query_params={
                "defType": "lucene",
                "df": "searchable_text",
                "rows": 10
            }
        )
        
        iteration_id = self.experiment_manager.run_iteration(
            experiment_id=self.experiment_id,
            query_config=baseline_config,
            iteration_name="baseline"
        )
        
        logger.info(f"✅ Baseline iteration completed: {iteration_id}")
        return iteration_id
    
    def run_basic_optimization_iteration(self) -> str:
        """Run an iteration with basic DisMax optimization."""
        logger.info("🔧 Running basic optimization iteration...")
        
        # Improved query configuration using DisMax
        optimized_config = QueryConfig(
            request_handler="/dismax",
            query_params={
                "defType": "dismax",
                "qf": "product_title^2.0 product_description^1.0 product_bullet_point^1.5 product_brand^1.2",
                "pf": "product_title^3.0 product_description^1.5",
                "mm": "2<-1 5<-2 6<90%",
                "tie": 0.01,
                "rows": 10
            }
        )
        
        iteration_id = self.experiment_manager.run_iteration(
            experiment_id=self.experiment_id,
            query_config=optimized_config,
            iteration_name="basic_optimization"
        )
        
        logger.info(f"✅ Basic optimization iteration completed: {iteration_id}")
        return iteration_id
    
    def run_advanced_optimization_iteration(self) -> str:
        """Run an iteration with advanced eDisMax optimization."""
        logger.info("🚀 Running advanced optimization iteration...")
        
        # Advanced query configuration using eDisMax
        advanced_config = QueryConfig(
            request_handler="/edismax",
            query_params={
                "defType": "edismax",
                "qf": "product_title^3.0 product_description^1.0 product_bullet_point^2.0 product_brand^1.5 all_text^0.5",
                "pf": "product_title^5.0 product_description^2.0",
                "pf2": "product_title^4.0 product_description^1.5",
                "pf3": "product_title^3.0 product_description^1.0",
                "mm": "3<-1 6<-2 8<75%",
                "tie": 0.1,
                "boost": "if(exists(product_brand),1.2,1.0)",
                "rows": 10
            }
        )
        
        iteration_id = self.experiment_manager.run_iteration(
            experiment_id=self.experiment_id,
            query_config=advanced_config,
            iteration_name="advanced_optimization"
        )
        
        logger.info(f"✅ Advanced optimization iteration completed: {iteration_id}")
        return iteration_id
    
    def compare_iterations(self, iteration1: str, iteration2: str):
        """Compare two iterations and show the results."""
        logger.info(f"📊 Comparing iterations: {iteration1} vs {iteration2}")
        
        comparison = self.experiment_manager.compare_iterations(
            experiment_id=self.experiment_id,
            iteration1=iteration1,
            iteration2=iteration2
        )
        
        self._display_comparison_results(comparison, iteration1, iteration2)
    
    def _display_comparison_results(self, comparison: Dict, iter1: str, iter2: str):
        """Display comparison results in a user-friendly format."""
        print(f"\n{'='*80}")
        print(f"📊 ITERATION COMPARISON: {iter1} → {iter2}")
        print('='*80)
        
        # Overall metrics comparison
        if 'overall_metrics' in comparison:
            print("\n🎯 OVERALL METRICS:")
            overall = comparison['overall_metrics']
            for metric, data in overall.items():
                old_val = data.get('iter1_value', 0)
                new_val = data.get('iter2_value', 0)
                change = data.get('change', 0)
                improvement = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"  {metric.upper()}: {old_val:.3f} → {new_val:.3f} ({change:+.3f}) {improvement}")
        
        # Per-query analysis
        if 'per_query_analysis' in comparison:
            print(f"\n🔍 PER-QUERY ANALYSIS:")
            per_query = comparison['per_query_analysis']
            improved_queries = [q for q, data in per_query.items() if data.get('change', 0) > 0]
            degraded_queries = [q for q, data in per_query.items() if data.get('change', 0) < 0]
            
            print(f"  📈 Improved queries: {len(improved_queries)}")
            print(f"  📉 Degraded queries: {len(degraded_queries)}")
            
            if improved_queries:
                print(f"\n  Top improved queries:")
                sorted_improved = sorted(
                    [(q, per_query[q].get('change', 0)) for q in improved_queries],
                    key=lambda x: x[1], reverse=True
                )[:5]
                for query, change in sorted_improved:
                    print(f"    '{query}': +{change:.3f}")
            
            if degraded_queries:
                print(f"\n  Most degraded queries:")
                sorted_degraded = sorted(
                    [(q, per_query[q].get('change', 0)) for q in degraded_queries],
                    key=lambda x: x[1]
                )[:3]
                for query, change in sorted_degraded:
                    print(f"    '{query}': {change:.3f}")
        
        # Summary
        print(f"\n💡 SUMMARY:")
        if 'overall_metrics' in comparison:
            overall = comparison['overall_metrics']
            primary_metric = next(iter(overall.keys())) if overall else None
            if primary_metric:
                change = overall[primary_metric].get('change', 0)
                if change > 0.01:
                    print(f"  🎉 Significant improvement! {primary_metric.upper()} increased by {change:.3f}")
                elif change > 0:
                    print(f"  ✅ Modest improvement: {primary_metric.upper()} increased by {change:.3f}")
                elif change < -0.01:
                    print(f"  ⚠️ Performance degraded: {primary_metric.upper()} decreased by {abs(change):.3f}")
                else:
                    print(f"  ➡️ No significant change in {primary_metric.upper()}")
        
        print('='*80)
    
    def show_experiment_history(self):
        """Show the complete experiment history."""
        logger.info("📋 Showing experiment history...")
        
        iterations = self.experiment_manager.list_iterations(self.experiment_id)
        
        print(f"\n{'='*60}")
        print(f"📈 EXPERIMENT HISTORY: {self.experiment_id}")
        print('='*60)
        
        for iteration in iterations:
            iteration_data = self.experiment_manager.get_iteration_result(
                self.experiment_id, iteration
            )
            
            metrics = iteration_data.get('metrics', {}).get('overall', {})
            primary_metric = metrics.get('ndcg', 0) if metrics else 0
            
            print(f"\n🔄 {iteration}:")
            print(f"  NDCG@10: {primary_metric:.3f}")
            
            if 'query_config' in iteration_data:
                config = iteration_data['query_config']
                handler = config.get('request_handler', 'unknown')
                print(f"  Handler: {handler}")
                
                if 'query_params' in config:
                    params = config['query_params']
                    if 'qf' in params:
                        print(f"  Query Fields: {params['qf']}")
                    if 'mm' in params:
                        print(f"  Min Should Match: {params['mm']}")
        
        print('='*60)
    
    def run_complete_demo(self):
        """Run the complete end-to-end demo."""
        print("🚀 SOLR OPTIMIZER COMPLETE DEMO")
        print("================================\n")
        
        try:
            # Setup
            self.setup_experiment_manager()
            queries, judgments = self.load_demo_data()
            self.create_demo_experiment(queries, judgments)
            
            print(f"\n🎯 Demo configured with {len(queries)} queries")
            print("Running optimization iterations...\n")
            
            # Run iterations
            baseline_id = self.run_baseline_iteration()
            time.sleep(2)  # Brief pause between iterations
            
            basic_id = self.run_basic_optimization_iteration()
            time.sleep(2)
            
            advanced_id = self.run_advanced_optimization_iteration()
            
            # Show comparisons
            print("\n" + "="*80)
            print("🔍 OPTIMIZATION RESULTS")
            print("="*80)
            
            self.compare_iterations(baseline_id, basic_id)
            print("\n")
            self.compare_iterations(basic_id, advanced_id)
            print("\n")
            self.compare_iterations(baseline_id, advanced_id)
            
            # Show full history
            self.show_experiment_history()
            
            print(f"\n🎉 Demo completed successfully!")
            print(f"📁 Experiment data saved in: experiment_storage/")
            print(f"🔗 Solr Admin UI: {self.solr_url}/")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise


def main():
    """Main function to run the complete demo."""
    demo = CompleteDemoOrchestrator()
    
    try:
        demo.run_complete_demo()
    except KeyboardInterrupt:
        logger.info("\n🛑 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
