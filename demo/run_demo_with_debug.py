#!/usr/bin/env python3
"""
Solr Optimizer Demo with Debug Mode

This script runs the Solr optimizer demo with comprehensive debug logging
and configuration analysis as part of Priority 4 Configuration Debugging tasks.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import debug components
from demo.debug_mode import DebugMode
from demo.schema_validation import SolrSchemaValidator
from demo.run_enhanced_demo import EnhancedDemoOrchestrator


class DebugEnabledDemoOrchestrator(EnhancedDemoOrchestrator):
    """Enhanced demo orchestrator with comprehensive debug capabilities."""
    
    def __init__(self, enable_debug: bool = True, debug_log_file: str = None):
        super().__init__()
        self.enable_debug = enable_debug
        self.debug_log_file = debug_log_file or f"demo_debug_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        # Initialize debug components
        if self.enable_debug:
            self.debug_mode = DebugMode(self.solr_url, self.collection_name, self.debug_log_file)
            self.schema_validator = SolrSchemaValidator(self.solr_url, self.collection_name)
        else:
            self.debug_mode = None
            self.schema_validator = None
    
    def run_debug_preflight_checks(self) -> bool:
        """Run comprehensive debug-enabled preflight checks."""
        print("🐛 DEBUG MODE: COMPREHENSIVE PREFLIGHT CHECKS")
        print("=" * 60)
        
        # Basic preflight validation
        if not super().run_preflight_validation():
            return False
        
        if not self.enable_debug:
            print("Debug mode disabled - skipping advanced checks")
            return True
        
        # Schema validation with detailed reporting
        print("\n🔍 SCHEMA VALIDATION WITH DEBUG ANALYSIS")
        print("-" * 50)
        
        try:
            passed, results = self.schema_validator.validate_all()
            self.schema_validator.print_validation_report()
            
            if not passed:
                print("\n⚠️  Schema validation failed but continuing with debug analysis...")
            
        except Exception as e:
            print(f"❌ Schema validation error: {e}")
            print("Continuing with limited debug capabilities...")
        
        # Configuration inspection
        print("\n🔧 CONFIGURATION INSPECTION")
        print("-" * 50)
        self.debug_mode.inspect_current_configuration()
        
        # Test query execution with debug
        print("\n🔍 DEBUG QUERY EXECUTION TEST")
        print("-" * 50)
        
        test_queries = ["laptop", "smartphone", "headphones"]
        test_configs = [
            {"defType": "lucene", "df": "_text_"},
            {"defType": "dismax", "qf": "product_title^2.0 product_description^1.0"},
            {"defType": "edismax", "qf": "product_title^3.0 product_description^1.0 product_bullet_point^1.5"}
        ]
        
        debug_results = []
        for i, query in enumerate(test_queries[:2]):  # Test first 2 queries
            config = test_configs[i % len(test_configs)]
            try:
                print(f"Testing query: '{query}' with {config.get('defType', 'lucene')} parser")
                debug_info = self.debug_mode.query_interceptor.execute_query_with_debug(
                    query, params=config
                )
                debug_results.append(debug_info)
                print(f"✅ Query successful: {debug_info.num_found} results in {debug_info.execution_time_ms:.2f}ms")
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        print(f"\n✅ Debug preflight checks completed")
        print(f"📁 Debug log: {self.debug_log_file}")
        return True
    
    def run_optimization_with_debug(self) -> bool:
        """Run optimization with debug logging for each iteration."""
        print("\n🚀 OPTIMIZATION WITH DEBUG ANALYSIS")
        print("=" * 60)
        
        if not self.enable_debug:
            return super().run_iterative_optimization()
        
        # Run baseline with debug
        print("🐛 DEBUG MODE: Analyzing baseline iteration")
        baseline_id, baseline_ndcg = self.run_baseline_iteration()
        
        if self.debug_mode:
            # Log baseline query execution
            try:
                baseline_debug = self.debug_mode.query_interceptor.execute_query_with_debug(
                    "laptop",  # Sample query for debug analysis
                    params={"df": "_text_", "rows": 10}
                )
                print(f"🔍 Baseline debug: {baseline_debug.num_found} results, {baseline_debug.execution_time_ms:.2f}ms")
            except Exception as e:
                print(f"⚠️  Baseline debug logging failed: {e}")
        
        # Run iterative optimization with debug logging
        print("\n🐛 DEBUG MODE: Analyzing optimization iterations")
        improvement_found = super().run_iterative_optimization()
        
        # Save comprehensive debug session
        if self.debug_mode:
            try:
                debug_summary = self.debug_mode.debug_logger.get_query_summary()
                print(f"\n📊 DEBUG SESSION SUMMARY:")
                print(f"   Total queries logged: {debug_summary.get('total_queries', 0)}")
                print(f"   Average execution time: {debug_summary.get('average_execution_time_ms', 0):.2f}ms")
                print(f"   Total results found: {debug_summary.get('total_results_found', 0)}")
                
                self.debug_mode.save_debug_session(f"optimization_debug_{time.strftime('%Y%m%d_%H%M%S')}.json")
                print(f"📁 Optimization debug saved")
                
            except Exception as e:
                print(f"⚠️  Debug session save failed: {e}")
        
        return improvement_found
    
    def analyze_query_performance(self, queries: List[str] = None) -> Dict[str, Any]:
        """Analyze query performance with different configurations."""
        if not self.enable_debug or not self.debug_mode:
            print("Debug mode not enabled - skipping performance analysis")
            return {}
        
        print("\n📊 QUERY PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        if queries is None:
            queries = ["laptop computer", "wireless headphones", "gaming mouse", "bluetooth speaker"]
        
        # Test configurations
        test_configs = [
            {"name": "Lucene", "params": {"defType": "lucene", "df": "_text_"}},
            {"name": "DisMax", "params": {"defType": "dismax", "qf": "product_title^2.0 product_description^1.0"}},
            {"name": "eDisMax", "params": {"defType": "edismax", "qf": "product_title^3.0 product_description^1.0 product_bullet_point^1.5", "pf": "product_title^4.0"}},
            {"name": "Advanced eDisMax", "params": {"defType": "edismax", "qf": "product_title^5.0 product_description^1.0 product_bullet_point^2.0 product_brand^1.5", "pf": "product_title^6.0", "pf2": "product_title^4.0", "mm": "2<-1 5<80%"}}
        ]
        
        performance_results = {}
        
        for query in queries:
            print(f"\nAnalyzing query: '{query}'")
            query_results = {}
            
            for config in test_configs:
                try:
                    debug_info = self.debug_mode.query_interceptor.execute_query_with_debug(
                        query, params=config["params"]
                    )
                    
                    query_results[config["name"]] = {
                        "execution_time_ms": debug_info.execution_time_ms,
                        "num_found": debug_info.num_found,
                        "config": config["params"]
                    }
                    
                    print(f"  {config['name']:15} {debug_info.execution_time_ms:6.2f}ms  {debug_info.num_found:6d} results")
                    
                except Exception as e:
                    print(f"  {config['name']:15} ERROR: {e}")
                    query_results[config["name"]] = {"error": str(e)}
            
            performance_results[query] = query_results
        
        # Summary analysis
        print(f"\n📊 PERFORMANCE SUMMARY:")
        config_stats = {}
        for config in test_configs:
            times = []
            results = []
            for query_data in performance_results.values():
                if config["name"] in query_data and "execution_time_ms" in query_data[config["name"]]:
                    times.append(query_data[config["name"]]["execution_time_ms"])
                    results.append(query_data[config["name"]]["num_found"])
            
            if times:
                config_stats[config["name"]] = {
                    "avg_time_ms": sum(times) / len(times),
                    "avg_results": sum(results) / len(results),
                    "queries_tested": len(times)
                }
        
        for config_name, stats in config_stats.items():
            print(f"  {config_name:15} Avg: {stats['avg_time_ms']:6.2f}ms  {stats['avg_results']:6.1f} results  ({stats['queries_tested']} queries)")
        
        return performance_results
    
    def generate_debug_report(self) -> str:
        """Generate a comprehensive debug report."""
        if not self.enable_debug:
            return "Debug mode not enabled"
        
        print("\n📋 GENERATING COMPREHENSIVE DEBUG REPORT")
        print("=" * 50)
        
        report_lines = []
        report_lines.append("# Solr Optimizer Debug Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Solr URL: {self.solr_url}")
        report_lines.append(f"Collection: {self.collection_name}")
        report_lines.append("")
        
        # Schema validation summary
        if self.schema_validator:
            try:
                passed, results = self.schema_validator.validate_all()
                report_lines.append("## Schema Validation")
                report_lines.append(f"Overall Status: {'PASSED' if passed else 'FAILED'}")
                report_lines.append(f"Total Checks: {len(results)}")
                
                critical_failures = [r for r in results if r.level.value == "CRITICAL" and not r.passed]
                warnings = [r for r in results if r.level.value == "WARNING" and not r.passed]
                
                report_lines.append(f"Critical Failures: {len(critical_failures)}")
                report_lines.append(f"Warnings: {len(warnings)}")
                report_lines.append("")
                
                if critical_failures:
                    report_lines.append("### Critical Issues:")
                    for result in critical_failures:
                        report_lines.append(f"- {result.check_name}: {result.message}")
                    report_lines.append("")
                
            except Exception as e:
                report_lines.append(f"## Schema Validation Error: {e}")
                report_lines.append("")
        
        # Query performance summary
        if self.debug_mode:
            try:
                summary = self.debug_mode.debug_logger.get_query_summary()
                report_lines.append("## Query Performance Summary")
                report_lines.append(f"Total Queries: {summary.get('total_queries', 0)}")
                report_lines.append(f"Average Execution Time: {summary.get('average_execution_time_ms', 0):.2f}ms")
                report_lines.append(f"Total Results: {summary.get('total_results_found', 0)}")
                report_lines.append(f"Average Results per Query: {summary.get('average_results_per_query', 0):.1f}")
                report_lines.append("")
                
                if summary.get('query_parsers_used'):
                    report_lines.append("### Query Parsers Used:")
                    for parser, count in summary['query_parsers_used'].items():
                        report_lines.append(f"- {parser}: {count} queries")
                    report_lines.append("")
                
            except Exception as e:
                report_lines.append(f"## Query Performance Error: {e}")
                report_lines.append("")
        
        # Optimization results summary
        if self.iteration_history:
            report_lines.append("## Optimization Results")
            report_lines.append(f"Baseline NDCG: {self.baseline_ndcg:.4f}")
            report_lines.append(f"Best NDCG: {self.best_ndcg:.4f}")
            improvement = self.best_ndcg - self.baseline_ndcg if self.baseline_ndcg else 0
            report_lines.append(f"Improvement: {improvement:+.4f}")
            report_lines.append("")
            
            report_lines.append("### Iteration History:")
            for i, result in enumerate(self.iteration_history):
                report_lines.append(f"{i+1}. {result['strategy']}: {result['ndcg']:.4f} ({result['improvement']:+.4f})")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        if self.schema_validator and hasattr(self, '_last_validation_results'):
            critical_failures = [r for r in self._last_validation_results if r.level.value == "CRITICAL" and not r.passed]
            if critical_failures:
                report_lines.append("### Critical Schema Issues to Fix:")
                for result in critical_failures:
                    if result.recommendation:
                        report_lines.append(f"- {result.recommendation}")
        
        if self.debug_mode:
            summary = self.debug_mode.debug_logger.get_query_summary()
            avg_time = summary.get('average_execution_time_ms', 0)
            if avg_time > 1000:  # More than 1 second
                report_lines.append("- Consider optimizing query performance (average >1s)")
            
            avg_results = summary.get('average_results_per_query', 0)
            if avg_results < 10:
                report_lines.append("- Low result counts may indicate search configuration issues")
        
        if improvement <= 0:
            report_lines.append("- No optimization improvement found - review relevance judgments and field weights")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = f"debug_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
        try:
            with open(report_file, 'w') as f:
                f.write(report_content)
            print(f"📁 Debug report saved: {report_file}")
        except Exception as e:
            print(f"⚠️  Failed to save report: {e}")
        
        return report_content
    
    def run_complete_debug_demo(self):
        """Run the complete demo with comprehensive debug analysis."""
        print("🐛 SOLR OPTIMIZER DEMO - DEBUG MODE")
        print("="*80)
        print("This debug-enabled demo will:")
        print("• Run comprehensive schema validation")
        print("• Log all Solr queries with detailed debug information")
        print("• Analyze query performance across different configurations")
        print("• Generate detailed debug reports and recommendations")
        print("• Save all debug data for further analysis")
        print("="*80)
        print()
        
        try:
            # Debug-enabled preflight validation
            if not self.run_debug_preflight_checks():
                print("❌ Debug preflight validation failed. Cannot proceed.")
                return False
            
            # Setup experiment manager
            self.setup_experiment_manager()
            
            # Load data with validation
            try:
                queries, judgments = self.load_demo_data()
            except FileNotFoundError as e:
                print(f"❌ Data loading failed: {e}")
                print("💡 Please run: python demo/scripts/download_data.py")
                return False
            
            # Create experiment
            self.create_demo_experiment(queries, judgments)
            
            # Run optimization with debug
            improvement_found = self.run_optimization_with_debug()
            
            # Performance analysis
            if self.enable_debug:
                self.analyze_query_performance(queries[:4])  # Analyze first 4 queries
            
            # Show detailed results
            self.show_detailed_results()
            
            # Generate comprehensive debug report
            self.generate_debug_report()
            
            print("\n🎉 DEBUG DEMO COMPLETED")
            print("=" * 60)
            print(f"Debug Log: {self.debug_log_file}")
            
            if self.enable_debug:
                print("Generated Files:")
                print(f"  • Debug log: {self.debug_log_file}")
                print(f"  • Optimization debug: optimization_debug_*.json")
                print(f"  • Debug report: debug_report_*.md")
            
            print("\n📋 Debug Analysis Summary:")
            if improvement_found:
                print(f"✅ Optimization successful: {self.best_ndcg:.4f} NDCG")
            else:
                print(f"📉 No improvement: {self.baseline_ndcg:.4f} NDCG baseline")
            
            if self.enable_debug and self.debug_mode:
                summary = self.debug_mode.debug_logger.get_query_summary()
                print(f"🔍 Debug queries: {summary.get('total_queries', 0)} logged")
                print(f"⏱️  Average query time: {summary.get('average_execution_time_ms', 0):.2f}ms")
            
            print("\n💡 Next Steps:")
            print("  • Review debug reports for configuration issues")
            print("  • Analyze query performance patterns")
            print("  • Use troubleshooting guide for specific problems")
            print("  • Consider schema optimizations based on validation results")
            
            return True
            
        except Exception as e:
            print(f"❌ Debug demo failed: {e}")
            if self.enable_debug:
                print(f"📁 Check debug log: {self.debug_log_file}")
            raise


def main():
    """Main function to run the debug-enabled demo."""
    parser = argparse.ArgumentParser(description="Run Solr Optimizer Demo with Debug Mode")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    parser.add_argument("--debug-log", type=str, help="Debug log file path")
    
    args = parser.parse_args()
    
    demo = DebugEnabledDemoOrchestrator(
        enable_debug=not args.no_debug,
        debug_log_file=args.debug_log
    )
    
    try:
        demo.run_complete_debug_demo()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
