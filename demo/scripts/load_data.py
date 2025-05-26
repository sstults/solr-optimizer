#!/usr/bin/env python3
"""
Load processed data into Solr for the optimization demo.

This script loads the product data and sets up the Solr collection
for testing and optimization.
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
from typing import Dict, List
import logging

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolrDataLoader:
    """Loads processed data into Solr collection."""
    
    def __init__(self, 
                 solr_url: str = "http://localhost:8983/solr",
                 collection_name: str = "ecommerce_products",
                 data_dir: Path = None):
        self.solr_url = solr_url
        self.collection_name = collection_name
        self.data_dir = data_dir or PROJECT_ROOT / "demo" / "data"
        self.processed_dir = self.data_dir / "processed"
        
    def check_solr_connection(self) -> bool:
        """Check if Solr is running and accessible."""
        try:
            response = requests.get(f"{self.solr_url}/admin/info/system", timeout=10)
            response.raise_for_status()
            logger.info(f"✅ Solr is accessible at {self.solr_url}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to Solr at {self.solr_url}: {e}")
            logger.error("   Make sure Solr is running. Try: cd demo/docker-setup && ./setup.sh")
            return False
    
    def check_collection_exists(self) -> bool:
        """Check if the target collection exists."""
        try:
            response = requests.get(
                f"{self.solr_url}/admin/collections",
                params={"action": "LIST", "wt": "json"},
                timeout=10
            )
            response.raise_for_status()
            collections = response.json().get("collections", [])
            
            if self.collection_name in collections:
                logger.info(f"✅ Collection '{self.collection_name}' exists")
                return True
            else:
                logger.error(f"❌ Collection '{self.collection_name}' not found")
                logger.error(f"   Available collections: {collections}")
                logger.error("   Run Docker setup first: cd demo/docker-setup && ./setup.sh")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error checking collections: {e}")
            return False
    
    def clear_existing_data(self) -> bool:
        """Clear existing data from the collection."""
        try:
            logger.info(f"🧹 Clearing existing data from '{self.collection_name}'...")
            
            # Delete all documents
            delete_data = {"delete": {"query": "*:*"}}
            response = requests.post(
                f"{self.solr_url}/{self.collection_name}/update",
                json=delete_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            # Commit the deletion
            commit_data = {"commit": {}}
            response = requests.post(
                f"{self.solr_url}/{self.collection_name}/update",
                json=commit_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            logger.info("✅ Existing data cleared")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error clearing data: {e}")
            return False
    
    def load_products(self, batch_size: int = 100) -> bool:
        """Load product data into Solr in batches."""
        products_file = self.processed_dir / "products.json"
        
        if not products_file.exists():
            logger.error(f"❌ Products file not found: {products_file}")
            logger.error("   Run data download first: python demo/scripts/download_data.py")
            return False
        
        logger.info(f"📥 Loading products from {products_file}...")
        
        try:
            with open(products_file) as f:
                products = json.load(f)
            
            logger.info(f"   Found {len(products):,} products to load")
            
            # Load in batches
            total_loaded = 0
            for i in range(0, len(products), batch_size):
                batch = products[i:i + batch_size]
                
                if self._load_product_batch(batch):
                    total_loaded += len(batch)
                    if total_loaded % 1000 == 0:
                        logger.info(f"   Loaded {total_loaded:,} / {len(products):,} products...")
                else:
                    logger.error(f"❌ Failed to load batch starting at index {i}")
                    return False
            
            # Final commit
            if self._commit_changes():
                logger.info(f"✅ Successfully loaded {total_loaded:,} products")
                return True
            else:
                logger.error("❌ Failed final commit")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading products: {e}")
            return False
    
    def _load_product_batch(self, products: List[Dict]) -> bool:
        """Load a batch of products into Solr."""
        try:
            # Prepare documents for Solr
            solr_docs = []
            for product in products:
                # Convert product to Solr document format
                doc = {
                    "id": product["id"],
                    "product_title": product["product_title"],
                    "product_description": product["product_description"],
                    "product_bullet_point": product["product_bullet_point"] if isinstance(product["product_bullet_point"], list) else [product["product_bullet_point"]],
                    "product_brand": product["product_brand"],
                    "product_color": product["product_color"],
                    "product_locale": product["product_locale"],
                    "category": product.get("category", ""),
                    "price": product.get("price", 0.0)
                }
                solr_docs.append(doc)
            
            # Send to Solr
            response = requests.post(
                f"{self.solr_url}/{self.collection_name}/update",
                json=solr_docs,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error loading batch: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"   Response: {e.response.text}")
            return False
    
    def _commit_changes(self) -> bool:
        """Commit changes to make them visible."""
        try:
            commit_data = {"commit": {}}
            response = requests.post(
                f"{self.solr_url}/{self.collection_name}/update",
                json=commit_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error committing changes: {e}")
            return False
    
    def verify_data_loaded(self) -> bool:
        """Verify that data was loaded correctly."""
        try:
            logger.info("🔍 Verifying loaded data...")
            
            # Check document count
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/select",
                params={"q": "*:*", "rows": "0", "wt": "json"},
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            doc_count = result["response"]["numFound"]
            
            logger.info(f"   📊 Total documents in collection: {doc_count:,}")
            
            if doc_count == 0:
                logger.error("❌ No documents found in collection")
                return False
            
            # Test a sample query
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/select",
                params={
                    "q": "laptop",
                    "rows": "5",
                    "wt": "json",
                    "fl": "id,product_title,product_brand"
                },
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            found_docs = result["response"]["numFound"]
            
            logger.info(f"   🔍 Sample query 'laptop' found {found_docs} results")
            
            if found_docs > 0:
                sample_doc = result["response"]["docs"][0]
                logger.info(f"   📝 Sample result: {sample_doc.get('product_title', 'N/A')}")
            
            logger.info("✅ Data verification completed successfully")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error verifying data: {e}")
            return False
    
    def create_demo_queries_file(self) -> bool:
        """Create a demo queries file that works with the loaded data."""
        try:
            logger.info("📝 Creating demo queries file...")
            
            # Load the generated queries
            queries_file = self.processed_dir / "queries.csv"
            if not queries_file.exists():
                logger.error(f"❌ Queries file not found: {queries_file}")
                return False
            
            import csv
            
            # Read queries and create a demo-friendly format
            demo_queries = []
            with open(queries_file, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i < 20:  # Take first 20 queries for demo
                        demo_queries.append({
                            "query_id": row["query_id"],
                            "query_text": row["query"],
                            "description": f"Demo query: {row['query']}"
                        })
            
            # Save demo queries
            demo_queries_file = self.data_dir / "demo_queries.json"
            with open(demo_queries_file, 'w') as f:
                json.dump(demo_queries, f, indent=2)
            
            logger.info(f"✅ Created demo queries file: {demo_queries_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating demo queries: {e}")
            return False
    
    def show_loading_summary(self):
        """Show a summary of what was loaded."""
        try:
            # Get collection stats
            response = requests.get(
                f"{self.solr_url}/{self.collection_name}/select",
                params={"q": "*:*", "rows": "0", "wt": "json", "facet": "true", "facet.field": "product_brand"},
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            doc_count = result["response"]["numFound"]
            
            # Get brand distribution
            brand_facets = result.get("facet_counts", {}).get("facet_fields", {}).get("product_brand", [])
            brand_counts = {}
            for i in range(0, len(brand_facets), 2):
                if i + 1 < len(brand_facets):
                    brand_counts[brand_facets[i]] = brand_facets[i + 1]
            
            print("\n" + "="*60)
            print("🎉 DATA LOADING SUMMARY")
            print("="*60)
            print(f"Collection: {self.collection_name}")
            print(f"Total Products: {doc_count:,}")
            print(f"Solr URL: {self.solr_url}")
            print(f"\nTop Brands:")
            for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {brand}: {count:,}")
            print("\nNext steps:")
            print("  1. Run demo: python demo/run_complete_demo.py")
            print("  2. Or try manual queries:")
            print(f"     curl '{self.solr_url}/{self.collection_name}/select?q=laptop&rows=5'")
            print("="*60)
            
        except Exception as e:
            logger.error(f"❌ Error generating summary: {e}")


def main():
    """Main function to load data into Solr."""
    print("🚀 Solr Data Loader")
    print("==================\n")
    
    loader = SolrDataLoader()
    
    try:
        # Pre-flight checks
        if not loader.check_solr_connection():
            sys.exit(1)
        
        if not loader.check_collection_exists():
            sys.exit(1)
        
        # Clear existing data
        if not loader.clear_existing_data():
            sys.exit(1)
        
        # Load products
        if not loader.load_products():
            sys.exit(1)
        
        # Verify loading
        if not loader.verify_data_loaded():
            sys.exit(1)
        
        # Create demo files
        loader.create_demo_queries_file()
        
        # Show summary
        loader.show_loading_summary()
        
        print("\n✅ Data loading completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Loading interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
