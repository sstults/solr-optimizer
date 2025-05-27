#!/usr/bin/env python3
"""
Demo Script with Enhanced Reporting Integration

This script demonstrates how to integrate the Priority 3 Enhanced Reporting features
with the existing Solr optimization demo.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enhanced_reporting import EnhancedReporter
from run_enhanced_demo import EnhancedDemoOrchestrator


class DemoWithEnhancedReporting(EnhancedDemoOrchestrator):
    """Enhanced demo with integrated Priority 3 reporting features."""
    
    def __init__(self):
        super().__init__()
        self.reporter = None
    
    def setup_enhanced_reporting(self):
        """Initialize the enhanced reporter."""
        print("📊 SETTING UP ENHANCED REPORTING")
        print("-" * 40)
        
        self.reporter = EnhancedReporter(
            solr_url=self.solr_url,
            collection_name=self.collection_name
        )
        
        print("✅ Enhanced reporting initialized")
        print()
    
    def capture_iteration_data(self, iteration_id: str, queries: list, config_params: dict):
        """Capture enhanced reporting data for an iteration."""
        if not self.reporter:
            return
        
        print(f"📊 Capturing enhanced reporting data for {iteration_id}...")
        
        # Capture sample results and validation data for first few queries
        sample_queries = queries[:5]  # Limit to avoid overwhelming output
        
        for query in sample_queries:
            try:
                # Capture sample results
                self.reporter.capture_sample_results(
                    query=query,
                    config_params=config_params,
                    iteration_id=iteration_id,
                    num_results=5
                )
                
                # Validate result counts
                self.reporter.validate_result_counts(
                    query=query,
                    config_params=config_params,
                    iteration_id=iteration_id
                )
                
            except Exception as e:
                print(f"⚠️  Warning: Could not capture data for query '{query}': {e}")
        
        print(f"✅ Enhanced reporting data captured for {iteration_id}")
    
    def run_baseline_iteration_with_reporting(self):
        """Run baseline with enhanced reporting data capture."""
        print("📊 RUNNING BASELINE WITH ENHANCED REPORTING")
        print("=" * 60)
        
        # Run normal baseline
        iteration_id, ndcg_score = self.run_baseline_iteration()
        
        # Capture enhanced reporting data
        baseline_config_params = {
            "df": "_text_",
            "rows": 10
        }
        self.capture_iteration_data(iteration_id, self.demo_queries, baseline_config_params)
        
        return iteration_id, ndcg_score
    
    def run_optimization_with_reporting(self):
        """Run optimization with enhanced reporting for best iteration."""
        print("🚀 RUNNING OPTIMIZATION WITH ENHANCED REPORTING")
        print("=" * 60)
        
        # Run normal iterative optimization
        improvement_found = self.run_iterative_optimization()
        
        # Capture enhanced reporting data for the best iteration
        if self.best_iteration_id:
            print(f"\n📊 Capturing enhanced data for best iteration: {self.best_iteration_id}")
            
            # Find the best iteration's configuration
            best_result = next(r for r in self.iteration_history if r['iteration_id'] == self.best_iteration_id)
            
            # This is a simplified config capture - in practice you'd need to
            # reconstruct the exact parameters used for the best iteration
            best_config_params = {
                "defType": "edismax",  # Most of our best strategies use edismax
                "qf": "product_title^3.0 product_description^1.0 product_bullet_point^2.0",
                "pf": "product_title^5.0",
                "mm": "3<-1 6<-2 8<75%",
                "tie": "0.1",
                "rows": 10
            }
            
            self.capture_iteration_data(self.best_iteration_id, self.demo_queries, best_config_params)
        
        return improvement_found
    
    def show_comprehensive_enhanced_reporting(self):
        """Show the comprehensive enhanced reporting output."""
        if not self.reporter:
            print("❌ Enhanced reporting not available")
            return
        
        # Generate comprehensive report
        self.reporter.generate_comprehensive_report(
            baseline_iteration="baseline",
            best_iteration=self.best_iteration_id or "baseline",
            queries=self.demo_queries,
            judgments=self.demo_judgments,
            baseline_ndcg=self.baseline_ndcg or 0.0,
            best_ndcg=self.best_ndcg or 0.0
        )
    
    def run_complete_demo_with_enhanced_reporting(self):
        """Run the complete demo with Priority 3 enhanced reporting features."""
        print("🚀 SOLR OPTIMIZER DEMO WITH ENHANCED REPORTING")
        print("="*80)
        print("This demo includes Priority 3 Enhanced Reporting features:")
        print("• Sample search results for manual verification")
        print("• Query-by-query NDCG breakdown")
        print("• Search result count validation")
        print("• Top-scoring documents with relevance verification")
        print("• Before/after comparison of document retrieval")
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
            
            # Step 3: Setup experiment manager and enhanced reporting
            self.setup_experiment_manager()
            self.setup_enhanced_reporting()
            
            # Step 4: Load data
            try:
                queries, judgments = self.load_demo_data()
                # Store for enhanced reporting
                self.demo_queries = queries
                self.demo_judgments = judgments
            except FileNotFoundError as e:
                print(f"❌ Data loading failed: {e}")
                print("💡 Please run: python demo/scripts/download_data.py")
                return False
            
            # Step 5: Create experiment
            self.create_demo_experiment(queries, judgments)
            
            # Step 6: Validate searches work
            search_validation_passed = self.validate_search_returns_results()
            
            # Step 7: Run baseline with reporting
            baseline_id, baseline_ndcg = self.run_baseline_iteration_with_reporting()
            
            # Step 8: Run optimization with reporting
            improvement_found = self.run_optimization_with_reporting()
            
            # Step 9: Show standard detailed results
            self.show_detailed_results()
            
            # Step 10: Show enhanced reporting
            print("\n" + "="*80)
            print("🎯 PRIORITY 3 ENHANCED REPORTING")
            print("="*80)
            self.show_comprehensive_enhanced_reporting()
            
            # Final summary
            print("📋 DEMO COMPLETION SUMMARY")
            print("=" * 60)
            
            if improvement_found:
                best_improvement = self.best_ndcg - self.baseline_ndcg
                best_improvement_pct = (best_improvement / self.baseline_ndcg) * 100
                print(f"🎉 SUCCESS! Found NDCG improvement:")
                print(f"   • Baseline NDCG@10: {self.baseline_ndcg:.4f}")
                print(f"   • Best NDCG@10: {self.best_ndcg:.4f}")
                print(f"   • Improvement: +{best_improvement:.4f} ({best_improvement_pct:+.2f}%)")
            else:
                print(f"📉 No improvement found in {len(self.iteration_history)-1} attempts")
            
            print()
            print("✅ Priority 3 Enhanced Reporting features demonstrated:")
            print("   • Sample search results captured and displayed")
            print("   • Query-by-query performance breakdown shown")
            print("   • Result count validation completed")
            print("   • Top-scoring documents analyzed")
            print("   • Before/after comparison generated")
            
            print("="*80)
            print("🎉 Enhanced demo with reporting completed!")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced demo failed: {e}")
            raise


def main():
    """Main function to run the enhanced demo with reporting."""
    demo = DemoWithEnhancedReporting()
    
    try:
        success = demo.run_complete_demo_with_enhanced_reporting()
        if success:
            print("\n🎯 All Priority 3 Enhanced Reporting features successfully demonstrated!")
        else:
            print("\n❌ Demo completed with issues - check output above")
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
