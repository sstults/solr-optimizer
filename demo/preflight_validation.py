#!/usr/bin/env python3
"""
Pre-flight Validation for Solr Optimizer Demo

This script performs comprehensive validation to ensure the demo will run successfully:
- Validates Solr connection and configuration
- Ensures searches return results
- Checks data quality
- Validates schema configuration
- Provides fallback configurations if needed
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None

@dataclass
class SearchValidationResult:
    """Result of search validation."""
    query: str
    results_count: int
    response_time_ms: float
    passed: bool
    error: Optional[str] = None

class SolrDemoPreflightValidator:
    """Comprehensive pre-flight validation for the Solr demo."""
    
    def __init__(self, 
                 solr_url: str = "http://localhost:8983/solr",
                 collection_name: str = "ecommerce_products",
                 data_dir: Path = None):
        self.solr_url = solr_url
        self.collection_name = collection_name
        self.data_dir = data_dir or PROJECT_ROOT / "demo" / "data"
        self.validation_results = []
        self.search_validation_results = []
        
        # Test queries for validation
        self.test_queries = [
            "laptop",
            "smartphone", 
            "headphones",
            "camera",
            "tablet",
            "wireless",
            "bluetooth",
            "gaming",
            "apple",
            "samsung"
        ]
        
        # Fallback configurations
        self.fallback_configs = {
            "basic_lucene": {
                "query_parser": "lucene",
                "default_field": "_text_",
                "description": "Basic Lucene query parser with _text_ field"
            },
            "simple_dismax": {
                "query_parser": "dismax",
                "query_fields": {"_text_": 1.0},
                "description": "Simple DisMax with single field"
            },
            "multi_field_dismax": {
                "query_parser": "dismax", 
                "query_fields": {
                    "product_title": 2.0,
                    "product_description": 1.0,
                    "_text_": 0.5
                },
                "description": "Multi-field DisMax with field boosting"
            }
        }
    
    def validate_solr_connection(self) -> ValidationResult:
        """Validate Solr server connection."""
        try:
            start_time = time.time()
            response = requests.get(f"{self.solr_url}/admin/info/system", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                system_info = response.json()
                return ValidationResult(
                    check_name="Solr Connection",
                    passed=True,
                    message=f"Successfully connected to Solr in {response_time:.1f}ms",
                    details={
                        "solr_version": system_info.get("lucene", {}).get("solr-spec-version", "unknown"),
                        "response_time_ms": response_time
                    }
                )
            else:
                return ValidationResult(
                    check_name="Solr Connection",
                    passed=False,
                    message=f"Solr returned status {response.status_code}",
                    fix_suggestion="Check if Solr is running: cd demo/docker-setup && ./setup.sh"
                )
                
        except requests.exceptions.ConnectionError:
            return ValidationResult(
                check_name="Solr Connection",
                passed=False,
                message=f"Cannot connect to Solr at {self.solr_url}",
                fix_suggestion="Start Solr: cd demo/docker-setup && ./setup.sh"
            )
        except Exception as e:
            return ValidationResult(
                check_name="Solr Connection",
                passed=False,
                message=f"Connection error: {e}",
                fix_suggestion="Check Solr URL and network connectivity"
            )
    
    def validate_collection_exists(self) -> ValidationResult:
        """Validate that the target collection exists and is accessible."""
        try:
            response = requests.get(
                f"{self.solr_url}/admin/collections",
                params={"action": "LIST", "wt": "json"},
                timeout=10
            )
            
            if response.status_code == 200:
                collections = response.json().get("collections", [])
                if self.collection_name in collections:
                    return ValidationResult(
                        check_name="Collection Existence",
                        passed=True,
                        message=f"Collection '{self.collection_name}' exists",
                        details={"available_collections": collections}
                    )
                else:
                    return ValidationResult(
                        check_name="Collection Existence",
                        passed=False,
                        message=f"Collection '{self.collection_name}' not found",
                        details={"available_collections": collections},
                        fix_suggestion="Create collection: cd demo/docker-setup && ./setup.sh"
                    )
            else:
                return ValidationResult(
                    check_name="Collection Existence",
                    passed=False,
                    message=f"Error checking collections: HTTP {response.status_code}",
                    fix_suggestion="Check Solr admin API accessibility"
                )
                
        except Exception as e:
            return ValidationResult(
                check_name="Collection Existence",
                passed=False,
                message=f"Error checking collections: {e}",
                fix_suggestion="Verify Solr is running and accessible"
            )
    
    def validate_data_loaded(self) -> ValidationResult:
        """Validate that data has been loaded into the collection."""
        try:
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/select",
                params={"q": "*:*", "rows": "0", "wt": "json"},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                doc_count = result["response"]["numFound"]
                
                if doc_count > 0:
                    return ValidationResult(
                        check_name="Data Loaded",
                        passed=True,
                        message=f"Found {doc_count:,} documents in collection",
                        details={"document_count": doc_count}
                    )
                else:
                    return ValidationResult(
                        check_name="Data Loaded",
                        passed=False,
                        message="No documents found in collection",
                        fix_suggestion="Load data: python demo/scripts/load_data.py"
                    )
            else:
                return ValidationResult(
                    check_name="Data Loaded",
                    passed=False,
                    message=f"Error querying collection: HTTP {response.status_code}",
                    fix_suggestion="Check collection configuration and accessibility"
                )
                
        except Exception as e:
            return ValidationResult(
                check_name="Data Loaded",
                passed=False,
                message=f"Error checking data: {e}",
                fix_suggestion="Verify collection exists and is accessible"
            )
    
    def validate_search_returns_results(self) -> ValidationResult:
        """Validate that searches return meaningful results."""
        self.search_validation_results = []
        successful_queries = 0
        total_queries = len(self.test_queries)
        
        for query in self.test_queries:
            try:
                start_time = time.time()
                response = requests.get(
                    f"{self.solr_url}/{self.collection_name}/select",
                    params={
                        "q": query,
                        "df": "_text_",
                        "rows": "10",
                        "wt": "json"
                    },
                    timeout=10
                )
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    results_count = result["response"]["numFound"]
                    
                    search_result = SearchValidationResult(
                        query=query,
                        results_count=results_count,
                        response_time_ms=response_time,
                        passed=results_count > 0
                    )
                    
                    if results_count > 0:
                        successful_queries += 1
                else:
                    search_result = SearchValidationResult(
                        query=query,
                        results_count=0,
                        response_time_ms=response_time,
                        passed=False,
                        error=f"HTTP {response.status_code}"
                    )
                
                self.search_validation_results.append(search_result)
                
            except Exception as e:
                search_result = SearchValidationResult(
                    query=query,
                    results_count=0,
                    response_time_ms=0,
                    passed=False,
                    error=str(e)
                )
                self.search_validation_results.append(search_result)
        
        success_rate = (successful_queries / total_queries) * 100
        
        if success_rate >= 70:  # At least 70% of queries should return results
            return ValidationResult(
                check_name="Search Results",
                passed=True,
                message=f"{successful_queries}/{total_queries} queries returned results ({success_rate:.1f}%)",
                details={
                    "success_rate": success_rate,
                    "successful_queries": successful_queries,
                    "total_queries": total_queries
                }
            )
        else:
            return ValidationResult(
                check_name="Search Results",
                passed=False,
                message=f"Only {successful_queries}/{total_queries} queries returned results ({success_rate:.1f}%)",
                details={
                    "success_rate": success_rate,
                    "successful_queries": successful_queries,
                    "total_queries": total_queries
                },
                fix_suggestion="Check data quality and field configuration"
            )
    
    def validate_data_quality(self) -> ValidationResult:
        """Validate data quality by checking field population and content."""
        try:
            # Check field population
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/select",
                params={
                    "q": "*:*",
                    "rows": "0",
                    "wt": "json",
                    "facet": "true",
                    "facet.field": ["product_brand", "product_locale"],
                    "stats": "true",
                    "stats.field": ["product_title", "product_description"]
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check field statistics
                stats = result.get("stats", {}).get("stats_fields", {})
                title_stats = stats.get("product_title", {})
                desc_stats = stats.get("product_description", {})
                
                # Check facet distribution
                facets = result.get("facet_counts", {}).get("facet_fields", {})
                brand_facets = facets.get("product_brand", [])
                locale_facets = facets.get("product_locale", [])
                
                # Calculate metrics
                total_docs = result["response"]["numFound"]
                title_count = title_stats.get("count", 0)
                desc_count = desc_stats.get("count", 0)
                
                title_coverage = (title_count / total_docs) * 100 if total_docs > 0 else 0
                desc_coverage = (desc_count / total_docs) * 100 if total_docs > 0 else 0
                
                unique_brands = len(brand_facets) // 2 if brand_facets else 0
                unique_locales = len(locale_facets) // 2 if locale_facets else 0
                
                # Quality thresholds
                min_title_coverage = 90  # 90% of docs should have titles
                min_desc_coverage = 70   # 70% of docs should have descriptions
                min_brands = 5           # At least 5 different brands
                
                quality_issues = []
                if title_coverage < min_title_coverage:
                    quality_issues.append(f"Low title coverage: {title_coverage:.1f}%")
                if desc_coverage < min_desc_coverage:
                    quality_issues.append(f"Low description coverage: {desc_coverage:.1f}%")
                if unique_brands < min_brands:
                    quality_issues.append(f"Few unique brands: {unique_brands}")
                
                if not quality_issues:
                    return ValidationResult(
                        check_name="Data Quality",
                        passed=True,
                        message=f"Data quality good: {title_coverage:.1f}% title coverage, {unique_brands} brands",
                        details={
                            "title_coverage": title_coverage,
                            "description_coverage": desc_coverage,
                            "unique_brands": unique_brands,
                            "unique_locales": unique_locales,
                            "total_documents": total_docs
                        }
                    )
                else:
                    return ValidationResult(
                        check_name="Data Quality",
                        passed=False,
                        message=f"Data quality issues: {'; '.join(quality_issues)}",
                        details={
                            "title_coverage": title_coverage,
                            "description_coverage": desc_coverage,
                            "unique_brands": unique_brands,
                            "issues": quality_issues
                        },
                        fix_suggestion="Regenerate data with better quality: python demo/scripts/download_data.py"
                    )
            else:
                return ValidationResult(
                    check_name="Data Quality",
                    passed=False,
                    message=f"Error checking data quality: HTTP {response.status_code}",
                    fix_suggestion="Check collection accessibility"
                )
                
        except Exception as e:
            return ValidationResult(
                check_name="Data Quality",
                passed=False,
                message=f"Error checking data quality: {e}",
                fix_suggestion="Verify collection and data are accessible"
            )
    
    def test_fallback_configurations(self) -> ValidationResult:
        """Test fallback search configurations to ensure at least one works."""
        working_configs = []
        
        for config_name, config in self.fallback_configs.items():
            try:
                params = {
                    "q": "laptop",
                    "rows": "5",
                    "wt": "json"
                }
                
                if config["query_parser"] == "lucene":
                    params["df"] = config["default_field"]
                elif config["query_parser"] in ["dismax", "edismax"]:
                    params["defType"] = config["query_parser"]
                    if "query_fields" in config:
                        qf_parts = [f"{field}^{boost}" for field, boost in config["query_fields"].items()]
                        params["qf"] = " ".join(qf_parts)
                
                response = requests.get(
                    f"{self.solr_url}/{self.collection_name}/select",
                    params=params,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result["response"]["numFound"] > 0:
                        working_configs.append(config_name)
                        
            except Exception:
                continue
        
        if working_configs:
            return ValidationResult(
                check_name="Fallback Configurations",
                passed=True,
                message=f"Working fallback configs: {', '.join(working_configs)}",
                details={"working_configs": working_configs}
            )
        else:
            return ValidationResult(
                check_name="Fallback Configurations",
                passed=False,
                message="No fallback configurations working",
                fix_suggestion="Check schema and data configuration"
            )
    
    def run_all_validations(self) -> bool:
        """Run all validation checks and return overall success."""
        print("🔍 SOLR DEMO PRE-FLIGHT VALIDATION")
        print("=" * 50)
        print()
        
        # Define validation sequence
        validations = [
            self.validate_solr_connection,
            self.validate_collection_exists,
            self.validate_data_loaded,
            self.validate_search_returns_results,
            self.validate_data_quality,
            self.test_fallback_configurations
        ]
        
        # Run validations
        for validation_func in validations:
            result = validation_func()
            self.validation_results.append(result)
            
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} {result.check_name}: {result.message}")
            
            if not result.passed and result.fix_suggestion:
                print(f"    💡 Fix: {result.fix_suggestion}")
            
            print()
        
        # Show search validation details if available
        if self.search_validation_results:
            print("🔍 SEARCH VALIDATION DETAILS")
            print("-" * 30)
            for search_result in self.search_validation_results:
                status = "✅" if search_result.passed else "❌"
                if search_result.passed:
                    print(f"{status} '{search_result.query}': {search_result.results_count} results ({search_result.response_time_ms:.1f}ms)")
                else:
                    error_msg = search_result.error or "No results"
                    print(f"{status} '{search_result.query}': {error_msg}")
            print()
        
        # Summary
        passed_count = sum(1 for result in self.validation_results if result.passed)
        total_count = len(self.validation_results)
        
        print("📊 VALIDATION SUMMARY")
        print("-" * 20)
        print(f"Passed: {passed_count}/{total_count} checks")
        
        if passed_count == total_count:
            print("🎉 All validations passed! Demo is ready to run.")
            return True
        else:
            print("⚠️  Some validations failed. Address issues before running demo.")
            critical_failures = [r for r in self.validation_results if not r.passed and r.check_name in ["Solr Connection", "Collection Existence", "Data Loaded"]]
            if critical_failures:
                print("🚨 Critical issues found - demo cannot run until resolved.")
            else:
                print("⚠️  Non-critical issues found - demo may run with reduced functionality.")
            return False
    
    def get_working_fallback_config(self) -> Optional[Dict[str, Any]]:
        """Get a working fallback configuration for demo use."""
        fallback_result = next((r for r in self.validation_results if r.check_name == "Fallback Configurations"), None)
        
        if fallback_result and fallback_result.passed:
            working_configs = fallback_result.details.get("working_configs", [])
            if working_configs:
                config_name = working_configs[0]  # Use first working config
                return {
                    "name": config_name,
                    "config": self.fallback_configs[config_name]
                }
        
        return None


def main():
    """Main validation function."""
    validator = SolrDemoPreflightValidator()
    
    if validator.run_all_validations():
        print("\n✅ Pre-flight validation successful!")
        sys.exit(0)
    else:
        print("\n❌ Pre-flight validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
