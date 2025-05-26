#!/usr/bin/env python3
"""
End-to-end demo validation tests.

Tests the complete demo workflow to ensure all components work together.
"""

import unittest
import sys
import time
import requests
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestEndToEndDemo(unittest.TestCase):
    """Test the complete demo workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.solr_url = "http://localhost:8983/solr"
        cls.collection_name = "ecommerce_products"
        cls.data_dir = PROJECT_ROOT / "demo" / "data"
    
    def test_solr_connectivity(self):
        """Test that Solr is accessible."""
        try:
            response = requests.get(f"{self.solr_url}/admin/info/system", timeout=10)
            self.assertEqual(response.status_code, 200)
            print("✅ Solr connectivity test passed")
        except requests.exceptions.RequestException:
            self.skipTest("Solr is not running - start with: cd demo/docker-setup && ./setup.sh")
    
    def test_collection_exists(self):
        """Test that the demo collection exists."""
        try:
            response = requests.get(
                f"{self.solr_url}/admin/collections",
                params={"action": "LIST", "wt": "json"},
                timeout=10
            )
            response.raise_for_status()
            collections = response.json().get("collections", [])
            self.assertIn(self.collection_name, collections)
            print(f"✅ Collection '{self.collection_name}' exists")
        except requests.exceptions.RequestException:
            self.skipTest("Cannot connect to Solr")
    
    def test_data_files_exist(self):
        """Test that required data files exist."""
        required_files = [
            self.data_dir / "processed" / "products.json",
            self.data_dir / "processed" / "queries.csv", 
            self.data_dir / "judgments" / "judgments.csv"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            self.skipTest(f"Demo data files not found. Run: python demo/scripts/download_data.py\nMissing files: {', '.join(missing_files)}")
        
        print("✅ All required data files exist")
    
    def test_solr_has_data(self):
        """Test that Solr collection has data loaded."""
        try:
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/select",
                params={"q": "*:*", "rows": "0", "wt": "json"},
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            doc_count = result["response"]["numFound"]
            
            self.assertGreater(doc_count, 0, "Collection should have documents")
            self.assertGreater(doc_count, 1000, "Collection should have substantial data for demo")
            
            print(f"✅ Collection contains {doc_count:,} documents")
            
        except requests.exceptions.RequestException:
            self.skipTest("Cannot query Solr collection")
    
    def test_sample_queries_work(self):
        """Test that sample queries return results."""
        test_queries = ["laptop", "shirt", "camera", "nike shoes", "apple"]
        
        for query in test_queries:
            with self.subTest(query=query):
                try:
                    response = requests.get(
                        f"{self.solr_url}/{self.collection_name}/select",
                        params={
                            "q": query,
                            "rows": "5",
                            "wt": "json",
                            "fl": "id,product_title,score"
                        },
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    found_docs = result["response"]["numFound"]
                    
                    # Not all queries will have results, but most should
                    if found_docs > 0:
                        docs = result["response"]["docs"]
                        self.assertGreater(len(docs), 0)
                        
                        # Check that results have required fields
                        first_doc = docs[0]
                        self.assertIn("id", first_doc)
                        self.assertIn("product_title", first_doc)
                        
                        print(f"✅ Query '{query}' returned {found_docs} results")
                    else:
                        print(f"ℹ️ Query '{query}' returned no results (this may be normal)")
                        
                except requests.exceptions.RequestException:
                    self.skipTest("Cannot connect to Solr - start with: cd demo/docker-setup && ./setup.sh")
    
    def test_dismax_handler_works(self):
        """Test that the DisMax handler is configured correctly."""
        try:
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/dismax",
                params={
                    "q": "laptop computer",
                    "rows": "3",
                    "wt": "json",
                    "fl": "id,product_title,score"
                },
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            found_docs = result["response"]["numFound"]
            
            if found_docs > 0:
                print(f"✅ DisMax handler works - found {found_docs} results")
            else:
                print("ℹ️ DisMax handler works but no results for test query")
                
        except requests.exceptions.RequestException:
            self.skipTest("Cannot connect to Solr - start with: cd demo/docker-setup && ./setup.sh")
    
    def test_edismax_handler_works(self):
        """Test that the eDisMax handler is configured correctly."""
        try:
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/edismax",
                params={
                    "q": "smartphone phone",
                    "rows": "3", 
                    "wt": "json",
                    "fl": "id,product_title,score"
                },
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            found_docs = result["response"]["numFound"]
            
            if found_docs > 0:
                print(f"✅ eDisMax handler works - found {found_docs} results")
            else:
                print("ℹ️ eDisMax handler works but no results for test query")
                
        except requests.exceptions.RequestException:
            self.skipTest("Cannot connect to Solr - start with: cd demo/docker-setup && ./setup.sh")
    
    def test_framework_components_importable(self):
        """Test that key framework components can be imported."""
        try:
            from solr_optimizer.core.default_experiment_manager import DefaultExperimentManager
            from solr_optimizer.models.experiment_config import ExperimentConfig
            from solr_optimizer.agents.solr.pysolr_execution_agent import PySolrExecutionAgent
            from solr_optimizer.agents.metrics.standard_metrics_agent import StandardMetricsAgent
            
            print("✅ Framework components import successfully")
            
        except ImportError as e:
            self.fail(f"Failed to import framework components: {e}")
    
    def test_demo_readiness_summary(self):
        """Provide a summary of demo readiness."""
        print("\n" + "="*60)
        print("🎯 DEMO READINESS SUMMARY")
        print("="*60)
        
        checks = [
            ("Solr connectivity", self._check_solr_connectivity()),
            ("Collection exists", self._check_collection_exists()),
            ("Data files exist", self._check_data_files_exist()),
            ("Collection has data", self._check_collection_has_data()),
            ("Sample queries work", self._check_sample_queries_work()),
            ("DisMax handler", self._check_dismax_handler()),
            ("eDisMax handler", self._check_edismax_handler()),
        ]
        
        passed = 0
        for check_name, check_result in checks:
            status = "✅ PASS" if check_result else "❌ FAIL"
            print(f"  {check_name:<25} {status}")
            if check_result:
                passed += 1
        
        print(f"\nOverall: {passed}/{len(checks)} checks passed")
        
        if passed == len(checks):
            print("🎉 Demo is ready to run!")
            print("\nTo run the demo:")
            print("  python demo/run_complete_demo.py")
        else:
            print("⚠️ Demo is not ready. Address failing checks first.")
            print("\nSetup steps:")
            print("  1. cd demo/docker-setup && ./setup.sh")
            print("  2. python demo/scripts/download_data.py")
            print("  3. python demo/scripts/load_data.py")
        
        print("="*60)
    
    def _check_solr_connectivity(self) -> bool:
        """Helper to check Solr connectivity."""
        try:
            response = requests.get(f"{self.solr_url}/admin/info/system", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_collection_exists(self) -> bool:
        """Helper to check collection exists."""
        try:
            response = requests.get(
                f"{self.solr_url}/admin/collections",
                params={"action": "LIST", "wt": "json"}, 
                timeout=5
            )
            collections = response.json().get("collections", [])
            return self.collection_name in collections
        except:
            return False
    
    def _check_data_files_exist(self) -> bool:
        """Helper to check data files exist."""
        required_files = [
            self.data_dir / "processed" / "products.json",
            self.data_dir / "processed" / "queries.csv",
            self.data_dir / "judgments" / "judgments.csv"
        ]
        return all(f.exists() for f in required_files)
    
    def _check_collection_has_data(self) -> bool:
        """Helper to check collection has data."""
        try:
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/select",
                params={"q": "*:*", "rows": "0", "wt": "json"},
                timeout=5
            )
            result = response.json()
            return result["response"]["numFound"] > 1000
        except:
            return False
    
    def _check_sample_queries_work(self) -> bool:
        """Helper to check sample queries work."""
        try:
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/select",
                params={"q": "laptop", "rows": "1", "wt": "json"},
                timeout=5
            )
            result = response.json()
            return result["response"]["numFound"] > 0
        except:
            return False
    
    def _check_dismax_handler(self) -> bool:
        """Helper to check DisMax handler."""
        try:
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/dismax",
                params={"q": "test", "rows": "1", "wt": "json"},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def _check_edismax_handler(self) -> bool:
        """Helper to check eDisMax handler."""
        try:
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/edismax", 
                params={"q": "test", "rows": "1", "wt": "json"},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


if __name__ == "__main__":
    unittest.main(verbosity=2)
