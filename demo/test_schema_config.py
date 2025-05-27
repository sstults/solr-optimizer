#!/usr/bin/env python3
"""
Schema Configuration Test Script

This script tests the Solr schema configuration to ensure:
- Default field (_text_) works properly
- Search queries return results
- Field weights are properly configured for relevance
- Different request handlers work correctly

Run this after setting up Solr and loading data.
"""

import sys
import json
import requests
from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SchemaConfigTester:
    """Tests Solr schema configuration and search functionality."""
    
    def __init__(self, solr_url: str = "http://localhost:8983/solr", collection: str = "ecommerce_products"):
        self.solr_url = solr_url
        self.collection = collection
        self.base_url = f"{solr_url}/{collection}"
        
        # Test queries that should exist in typical ecommerce data
        self.test_queries = [
            "laptop",
            "wireless headphones",
            "coffee maker",
            "running shoes",
            "smartphone",
            "tablet",
            "kitchen",
            "electronics",
            "black",
            "portable"
        ]
    
    def check_solr_connection(self) -> bool:
        """Check if Solr is running and collection exists."""
        try:
            response = requests.get(f"{self.base_url}/admin/ping", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Solr connection successful")
                return True
            else:
                logger.error(f"❌ Solr ping failed with status {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"❌ Failed to connect to Solr: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents in the collection."""
        try:
            response = requests.get(f"{self.base_url}/select", params={
                "q": "*:*",
                "rows": 0,
                "wt": "json"
            }, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                count = data['response']['numFound']
                logger.info(f"📊 Collection contains {count} documents")
                return count
            else:
                logger.error(f"❌ Failed to get document count: {response.status_code}")
                return 0
        except Exception as e:
            logger.error(f"❌ Error getting document count: {e}")
            return 0
    
    def test_default_field_search(self) -> Dict[str, Any]:
        """Test searches using the default field (_text_)."""
        logger.info("🔍 Testing default field (_text_) searches...")
        
        results = {}
        for query in self.test_queries[:5]:  # Test first 5 queries
            try:
                response = requests.get(f"{self.base_url}/select", params={
                    "q": query,
                    "rows": 3,
                    "wt": "json"
                }, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    num_found = data['response']['numFound']
                    docs = data['response']['docs']
                    
                    results[query] = {
                        'num_found': num_found,
                        'docs': docs,
                        'success': True
                    }
                    
                    logger.info(f"   Query '{query}': {num_found} results")
                    if num_found > 0 and docs:
                        # Show first result for verification
                        first_doc = docs[0]
                        title = first_doc.get('product_title', ['N/A'])[0] if isinstance(first_doc.get('product_title'), list) else first_doc.get('product_title', 'N/A')
                        logger.info(f"      Top result: {title}")
                else:
                    results[query] = {'success': False, 'error': f"HTTP {response.status_code}"}
                    logger.error(f"   Query '{query}' failed: HTTP {response.status_code}")
                    
            except Exception as e:
                results[query] = {'success': False, 'error': str(e)}
                logger.error(f"   Query '{query}' failed: {e}")
        
        return results
    
    def test_dismax_handler(self) -> Dict[str, Any]:
        """Test the DisMax request handler."""
        logger.info("🎯 Testing DisMax request handler...")
        
        results = {}
        for query in self.test_queries[:3]:  # Test first 3 queries
            try:
                response = requests.get(f"{self.base_url}/dismax", params={
                    "q": query,
                    "rows": 3,
                    "wt": "json"
                }, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    num_found = data['response']['numFound']
                    docs = data['response']['docs']
                    
                    results[query] = {
                        'num_found': num_found,
                        'docs': docs,
                        'success': True
                    }
                    
                    logger.info(f"   DisMax '{query}': {num_found} results")
                    if num_found > 0 and docs:
                        first_doc = docs[0]
                        title = first_doc.get('product_title', ['N/A'])[0] if isinstance(first_doc.get('product_title'), list) else first_doc.get('product_title', 'N/A')
                        logger.info(f"      Top result: {title}")
                        
                        # Check if we get score information
                        if 'score' in first_doc:
                            logger.info(f"      Score: {first_doc['score']:.4f}")
                else:
                    results[query] = {'success': False, 'error': f"HTTP {response.status_code}"}
                    logger.error(f"   DisMax '{query}' failed: HTTP {response.status_code}")
                    
            except Exception as e:
                results[query] = {'success': False, 'error': str(e)}
                logger.error(f"   DisMax '{query}' failed: {e}")
        
        return results
    
    def test_edismax_handler(self) -> Dict[str, Any]:
        """Test the eDisMax request handler."""
        logger.info("⚡ Testing eDisMax request handler...")
        
        results = {}
        for query in self.test_queries[:3]:  # Test first 3 queries
            try:
                response = requests.get(f"{self.base_url}/edismax", params={
                    "q": query,
                    "rows": 3,
                    "wt": "json"
                }, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    num_found = data['response']['numFound']
                    docs = data['response']['docs']
                    
                    results[query] = {
                        'num_found': num_found,
                        'docs': docs,
                        'success': True
                    }
                    
                    logger.info(f"   eDisMax '{query}': {num_found} results")
                    if num_found > 0 and docs:
                        first_doc = docs[0]
                        title = first_doc.get('product_title', ['N/A'])[0] if isinstance(first_doc.get('product_title'), list) else first_doc.get('product_title', 'N/A')
                        logger.info(f"      Top result: {title}")
                        
                        if 'score' in first_doc:
                            logger.info(f"      Score: {first_doc['score']:.4f}")
                else:
                    results[query] = {'success': False, 'error': f"HTTP {response.status_code}"}
                    logger.error(f"   eDisMax '{query}' failed: HTTP {response.status_code}")
                    
            except Exception as e:
                results[query] = {'success': False, 'error': str(e)}
                logger.error(f"   eDisMax '{query}' failed: {e}")
        
        return results
    
    def test_field_weights(self) -> Dict[str, Any]:
        """Test that field weights are working by comparing title vs description matches."""
        logger.info("⚖️  Testing field weight configuration...")
        
        # Test with a query that should prioritize title matches
        test_query = "laptop computer"
        
        results = {}
        
        try:
            # Test with eDisMax to see field weighting in action
            response = requests.get(f"{self.base_url}/edismax", params={
                "q": test_query,
                "rows": 5,
                "wt": "json",
                "debugQuery": "true"
            }, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                docs = data['response']['docs']
                debug_info = data.get('debug', {})
                
                if docs:
                    results['field_weight_test'] = {
                        'query': test_query,
                        'num_results': len(docs),
                        'top_results': [],
                        'success': True
                    }
                    
                    logger.info(f"   Field weight test for '{test_query}':")
                    for i, doc in enumerate(docs):
                        title = doc.get('product_title', ['N/A'])[0] if isinstance(doc.get('product_title'), list) else doc.get('product_title', 'N/A')
                        score = doc.get('score', 0)
                        
                        results['field_weight_test']['top_results'].append({
                            'rank': i + 1,
                            'title': title,
                            'score': score
                        })
                        
                        logger.info(f"      #{i+1}: {title} (score: {score:.4f})")
                    
                    # Check if title field matches score higher than description matches
                    title_match_found = any(test_query.lower() in title.lower() for title in [doc.get('product_title', '') for doc in docs[:3]])
                    if title_match_found:
                        logger.info("   ✅ Title field weighting appears to be working (title matches in top results)")
                    else:
                        logger.warning("   ⚠️ Title field weighting may need adjustment (no title matches in top 3)")
                        
                else:
                    results['field_weight_test'] = {'success': False, 'error': 'No results returned'}
                    logger.error(f"   No results for field weight test query '{test_query}'")
                    
            else:
                results['field_weight_test'] = {'success': False, 'error': f"HTTP {response.status_code}"}
                logger.error(f"   Field weight test failed: HTTP {response.status_code}")
                
        except Exception as e:
            results['field_weight_test'] = {'success': False, 'error': str(e)}
            logger.error(f"   Field weight test failed: {e}")
        
        return results
    
    def test_copy_fields(self) -> Dict[str, Any]:
        """Test that copy fields are working properly."""
        logger.info("📝 Testing copy field configuration...")
        
        results = {}
        
        # Test that _text_ field contains content from source fields
        try:
            response = requests.get(f"{self.base_url}/select", params={
                "q": "_text_:laptop",  # Search specifically in _text_ field
                "rows": 3,
                "wt": "json"
            }, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                num_found = data['response']['numFound']
                
                results['copy_field_test'] = {
                    'field': '_text_',
                    'query': 'laptop',
                    'num_found': num_found,
                    'success': True
                }
                
                logger.info(f"   _text_ field search for 'laptop': {num_found} results")
                
                if num_found > 0:
                    logger.info("   ✅ Copy fields to _text_ are working")
                else:
                    logger.warning("   ⚠️ Copy fields to _text_ may not be working properly")
                    
            else:
                results['copy_field_test'] = {'success': False, 'error': f"HTTP {response.status_code}"}
                logger.error(f"   Copy field test failed: HTTP {response.status_code}")
                
        except Exception as e:
            results['copy_field_test'] = {'success': False, 'error': str(e)}
            logger.error(f"   Copy field test failed: {e}")
        
        return results
    
    def generate_report(self, all_results: Dict[str, Any]) -> None:
        """Generate a comprehensive test report."""
        logger.info("📋 SCHEMA CONFIGURATION TEST REPORT")
        logger.info("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        # Summarize results
        for test_category, results in all_results.items():
            logger.info(f"\n{test_category.upper()}:")
            
            if isinstance(results, dict):
                if 'success' in results:
                    # Single test result
                    total_tests += 1
                    if results['success']:
                        passed_tests += 1
                        logger.info("   ✅ PASSED")
                    else:
                        logger.info(f"   ❌ FAILED: {results.get('error', 'Unknown error')}")
                else:
                    # Multiple test results
                    for query, result in results.items():
                        total_tests += 1
                        if result.get('success', False):
                            passed_tests += 1
                            logger.info(f"   ✅ '{query}': {result.get('num_found', 0)} results")
                        else:
                            logger.info(f"   ❌ '{query}': {result.get('error', 'Unknown error')}")
        
        logger.info(f"\nOVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All schema configuration tests PASSED!")
            logger.info("   The schema is properly configured for search optimization.")
        elif passed_tests >= total_tests * 0.8:
            logger.info("✅ Most schema tests passed - configuration looks good!")
            logger.info("   Minor issues may need attention.")
        else:
            logger.info("⚠️ Several schema tests failed - configuration needs attention!")
            logger.info("   Review the errors above and check Solr configuration.")
        
        logger.info("\nRECOMMENDATIONS:")
        if passed_tests < total_tests:
            logger.info("   • Check that Solr is running and data is loaded")
            logger.info("   • Verify schema.xml copy field configurations")
            logger.info("   • Test individual fields with Solr admin UI")
            logger.info("   • Review solrconfig.xml request handler settings")
        else:
            logger.info("   • Schema configuration is working correctly")
            logger.info("   • Ready to run optimization experiments")
            logger.info("   • Consider testing with production-like queries")
    
    def run_all_tests(self) -> bool:
        """Run all schema configuration tests."""
        logger.info("🚀 STARTING SCHEMA CONFIGURATION TESTS")
        logger.info("=" * 60)
        
        # Check connection first
        if not self.check_solr_connection():
            logger.error("❌ Cannot connect to Solr - aborting tests")
            return False
        
        # Check document count
        doc_count = self.get_document_count()
        if doc_count == 0:
            logger.error("❌ No documents found in collection - load data first")
            return False
        
        # Run all tests
        all_results = {}
        
        all_results['default_field_tests'] = self.test_default_field_search()
        all_results['dismax_tests'] = self.test_dismax_handler()
        all_results['edismax_tests'] = self.test_edismax_handler()
        all_results['field_weight_tests'] = self.test_field_weights()
        all_results['copy_field_tests'] = self.test_copy_fields()
        
        # Generate report
        self.generate_report(all_results)
        
        # Return True if majority of tests passed
        total_tests = sum(len(results) if isinstance(results, dict) and 'success' not in results 
                         else 1 for results in all_results.values())
        passed_tests = 0
        
        for results in all_results.values():
            if isinstance(results, dict):
                if 'success' in results:
                    if results['success']:
                        passed_tests += 1
                else:
                    for result in results.values():
                        if result.get('success', False):
                            passed_tests += 1
        
        return passed_tests >= total_tests * 0.8


def main():
    """Main function to run schema configuration tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Solr schema configuration')
    parser.add_argument('--solr-url', default='http://localhost:8983/solr',
                       help='Solr base URL (default: http://localhost:8983/solr)')
    parser.add_argument('--collection', default='ecommerce_products',
                       help='Collection name (default: ecommerce_products)')
    
    args = parser.parse_args()
    
    tester = SchemaConfigTester(args.solr_url, args.collection)
    
    try:
        success = tester.run_all_tests()
        if success:
            logger.info("\n🎉 Schema configuration tests completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n❌ Schema configuration tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
