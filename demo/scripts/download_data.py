#!/usr/bin/env python3
"""
Download and prepare the Amazon ESCI dataset for Solr optimization demo.

This script downloads the Amazon "Exact-Small-Complete-Irrelevant" dataset
which provides real e-commerce product data with relevance judgments.
"""

import os
import sys
import json
import csv
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile
import tempfile
import shutil
import logging

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESCIDataDownloader:
    """Downloads and processes Amazon ESCI dataset for demo purposes."""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or PROJECT_ROOT / "demo" / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.judgments_dir = self.data_dir / "judgments"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.judgments_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_esci_dataset(self) -> bool:
        """
        Download Amazon ESCI dataset from official sources.
        Returns True if successful, False otherwise.
        """
        logger.info("🔍 Attempting to download Amazon ESCI dataset...")
        
        # Primary source: Amazon's official dataset release
        urls = [
            # Try GitHub first (if available)
            "https://raw.githubusercontent.com/amazon-science/esci-data/main/small_version/shopping_queries_dataset_products.parquet",
            "https://raw.githubusercontent.com/amazon-science/esci-data/main/small_version/shopping_queries_dataset_examples.parquet",
            "https://raw.githubusercontent.com/amazon-science/esci-data/main/small_version/shopping_queries_dataset_sources.csv",
        ]
        
        # Alternative: Use synthetic/sample data if real dataset unavailable
        return self._download_sample_data()

    def _download_sample_data(self) -> bool:
        """Generate sample e-commerce data for demonstration purposes."""
        logger.info("📦 Generating sample e-commerce dataset...")
        
        # Generate sample products
        products = self._generate_sample_products(10000)
        
        # Generate sample queries and judgments
        queries, judgments = self._generate_sample_queries_and_judgments(products, 75)
        
        # Save to files
        self._save_products(products)
        self._save_queries_and_judgments(queries, judgments)
        
        logger.info("✅ Sample dataset generated successfully")
        return True
    
    def _generate_sample_products(self, count: int = 10000) -> List[Dict]:
        """Generate realistic sample product data."""
        import random
        
        # Product categories and their typical items
        categories = {
            "Electronics": {
                "items": ["laptop", "smartphone", "tablet", "headphones", "speaker", "camera", "smartwatch", "charger"],
                "brands": ["Apple", "Samsung", "Sony", "HP", "Dell", "Lenovo", "Bose", "Canon"],
                "price_range": (50, 2000)
            },
            "Clothing": {
                "items": ["shirt", "pants", "dress", "jacket", "shoes", "hat", "sweater", "jeans"],
                "brands": ["Nike", "Adidas", "Levi's", "Gap", "H&M", "Zara", "Uniqlo", "Under Armour"],
                "price_range": (15, 300)
            },
            "Home & Kitchen": {
                "items": ["blender", "coffee maker", "vacuum", "air fryer", "knife set", "cutting board", "cookware", "storage"],
                "brands": ["KitchenAid", "Cuisinart", "Ninja", "Dyson", "Black+Decker", "Hamilton Beach", "Instant Pot", "OXO"],
                "price_range": (25, 500)
            },
            "Sports & Outdoors": {
                "items": ["running shoes", "yoga mat", "water bottle", "backpack", "tent", "sleeping bag", "bike", "weights"],
                "brands": ["Nike", "Adidas", "Patagonia", "The North Face", "REI", "Coleman", "Hydro Flask", "Yeti"],
                "price_range": (20, 800)
            },
            "Books": {
                "items": ["novel", "cookbook", "textbook", "biography", "mystery", "romance", "sci-fi", "self-help"],
                "brands": ["Penguin", "Random House", "Harper", "Simon & Schuster", "Macmillan", "Wiley", "O'Reilly", "Academic Press"],
                "price_range": (10, 150)
            }
        }
        
        colors = ["black", "white", "red", "blue", "green", "gray", "brown", "silver", "gold", "pink"]
        adjectives = ["premium", "lightweight", "durable", "professional", "comfortable", "stylish", "compact", "wireless", "waterproof", "ergonomic"]
        
        products = []
        
        for i in range(count):
            category = random.choice(list(categories.keys()))
            cat_data = categories[category]
            
            item = random.choice(cat_data["items"])
            brand = random.choice(cat_data["brands"])
            color = random.choice(colors) if random.random() > 0.3 else ""
            adjective = random.choice(adjectives) if random.random() > 0.4 else ""
            
            # Build product title
            title_parts = [adjective, color, brand, item]
            title_parts = [part for part in title_parts if part]  # Remove empty strings
            product_title = " ".join(title_parts).title()
            
            # Generate price
            price_min, price_max = cat_data["price_range"]
            price = round(random.uniform(price_min, price_max), 2)
            
            # Generate description
            description = f"High-quality {item.lower()} from {brand}. "
            if adjective:
                description += f"Features {adjective.lower()} design. "
            if color:
                description += f"Available in {color.lower()}. "
            description += f"Perfect for your {category.lower().replace(' & ', ' and ')} needs."
            
            # Generate bullet points
            bullet_points = []
            if adjective:
                bullet_points.append(f"{adjective.title()} construction")
            if color:
                bullet_points.append(f"{color.title()} color option")
            bullet_points.append(f"Brand: {brand}")
            bullet_points.append(f"Category: {category}")
            if random.random() > 0.5:
                bullet_points.append("Free shipping available")
            
            product = {
                "id": f"prod_{i+1:06d}",
                "product_title": product_title,
                "product_description": description,
                "product_bullet_point": bullet_points,
                "product_brand": brand,
                "product_color": color if color else "N/A",
                "product_locale": "US",
                "category": category,
                "price": price
            }
            
            products.append(product)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"   Generated {i+1:,} products...")
        
        return products
    
    def _generate_sample_queries_and_judgments(self, products: List[Dict], query_count: int = 75) -> Tuple[List[Dict], List[Dict]]:
        """Generate sample queries with relevance judgments."""
        import random
        
        # Query templates based on common e-commerce search patterns
        query_patterns = [
            # Brand + item queries
            ("{brand} {item}", lambda p: p["product_brand"].lower()),
            # Item queries
            ("{item}", lambda p: any(word in p["product_title"].lower() for word in p["product_title"].lower().split())),
            # Color + item queries  
            ("{color} {item}", lambda p: p["product_color"].lower()),
            # Adjective + item queries
            ("{adjective} {item}", lambda p: any(word in p["product_description"].lower() for word in p["product_description"].lower().split())),
            # Category queries
            ("{category}", lambda p: p["category"].lower()),
            # Price-based queries
            ("cheap {item}", lambda p: p["price"] < 100),
            ("expensive {item}", lambda p: p["price"] > 200),
        ]
        
        # Extract terms from products
        brands = list(set(p["product_brand"] for p in products))
        items = ["laptop", "phone", "shirt", "shoes", "camera", "headphones", "jacket", "tablet", "watch", "bag"]
        colors = ["black", "white", "red", "blue", "green"]
        adjectives = ["wireless", "premium", "lightweight", "professional", "comfortable"]
        categories = list(set(p["category"] for p in products))
        
        queries = []
        judgments = []
        
        for i in range(query_count):
            # Select a query pattern
            pattern, relevance_func = random.choice(query_patterns)
            
            # Fill in the pattern
            query_text = pattern.format(
                brand=random.choice(brands),
                item=random.choice(items),
                color=random.choice(colors),
                adjective=random.choice(adjectives),
                category=random.choice(categories)
            ).lower()
            
            query_id = f"query_{i+1:03d}"
            
            query = {
                "query_id": query_id,
                "query": query_text,
                "category": "demo"
            }
            queries.append(query)
            
            # Generate relevance judgments for this query
            query_terms = set(query_text.lower().split())
            
            # Find relevant products
            relevant_products = []
            for product in products:
                score = self._calculate_relevance_score(query_terms, product)
                if score > 0:
                    relevant_products.append((product["id"], score))
            
            # Sort by relevance and take top 30 for judgment
            relevant_products.sort(key=lambda x: x[1], reverse=True)
            
            # Create judgments (mix of relevant and irrelevant)
            for j, (product_id, score) in enumerate(relevant_products[:30]):
                if score >= 0.8:
                    judgment = 2  # Highly relevant
                elif score >= 0.4:
                    judgment = 1  # Somewhat relevant
                else:
                    judgment = 0  # Not relevant
                
                judgments.append({
                    "query_id": query_id,
                    "product_id": product_id,
                    "judgment": judgment
                })
            
            # Add some random irrelevant products
            irrelevant_products = random.sample([p for p in products if p["id"] not in [pid for pid, _ in relevant_products[:30]]], min(10, len(products) - 30))
            for product in irrelevant_products:
                judgments.append({
                    "query_id": query_id,
                    "product_id": product["id"],
                    "judgment": 0
                })
        
        return queries, judgments
    
    def _calculate_relevance_score(self, query_terms: set, product: Dict) -> float:
        """Calculate a relevance score between query terms and product."""
        # Combine all product text
        product_text = " ".join([
            product["product_title"],
            product["product_description"],
            product["product_brand"],
            product["category"],
            " ".join(product["product_bullet_point"])
        ]).lower()
        
        product_words = set(product_text.split())
        
        # Calculate overlap
        overlap = len(query_terms.intersection(product_words))
        if overlap == 0:
            return 0.0
        
        # Basic relevance score
        title_matches = len(query_terms.intersection(set(product["product_title"].lower().split())))
        brand_matches = len(query_terms.intersection(set(product["product_brand"].lower().split())))
        
        score = (overlap / len(query_terms)) * 0.5  # Base score
        score += (title_matches / len(query_terms)) * 0.3  # Title bonus
        score += (brand_matches / len(query_terms)) * 0.2  # Brand bonus
        
        return min(1.0, score)
    
    def _save_products(self, products: List[Dict]):
        """Save products to JSON file."""
        output_file = self.processed_dir / "products.json"
        with open(output_file, 'w') as f:
            json.dump(products, f, indent=2)
        logger.info(f"💾 Saved {len(products):,} products to {output_file}")
    
    def _save_queries_and_judgments(self, queries: List[Dict], judgments: List[Dict]):
        """Save queries and judgments to CSV files."""
        # Save queries
        queries_file = self.processed_dir / "queries.csv"
        with open(queries_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "query", "category"])
            writer.writeheader()
            writer.writerows(queries)
        logger.info(f"💾 Saved {len(queries):,} queries to {queries_file}")
        
        # Save judgments
        judgments_file = self.judgments_dir / "judgments.csv"
        with open(judgments_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "product_id", "judgment"])
            writer.writeheader()
            writer.writerows(judgments)
        logger.info(f"💾 Saved {len(judgments):,} judgments to {judgments_file}")
        
        # Also save in TREC format
        trec_file = self.judgments_dir / "judgments.trec"
        with open(trec_file, 'w') as f:
            for judgment in judgments:
                f.write(f"{judgment['query_id']} 0 {judgment['product_id']} {judgment['judgment']}\n")
        logger.info(f"💾 Saved judgments in TREC format to {trec_file}")

    def generate_summary_report(self):
        """Generate a summary report of the downloaded data."""
        logger.info("📊 Generating data summary report...")
        
        # Load data
        products_file = self.processed_dir / "products.json"
        queries_file = self.processed_dir / "queries.csv"
        judgments_file = self.judgments_dir / "judgments.csv"
        
        if not all(f.exists() for f in [products_file, queries_file, judgments_file]):
            logger.error("❌ Required data files not found. Please run download first.")
            return
        
        with open(products_file) as f:
            products = json.load(f)
        
        queries_df = pd.read_csv(queries_file)
        judgments_df = pd.read_csv(judgments_file)
        
        # Create summary
        summary = {
            "dataset_info": {
                "total_products": len(products),
                "total_queries": len(queries_df),
                "total_judgments": len(judgments_df),
                "avg_judgments_per_query": len(judgments_df) / len(queries_df)
            },
            "product_categories": pd.Series([p["category"] for p in products]).value_counts().to_dict(),
            "judgment_distribution": judgments_df["judgment"].value_counts().to_dict(),
            "price_statistics": {
                "min_price": min(p["price"] for p in products),
                "max_price": max(p["price"] for p in products),
                "avg_price": sum(p["price"] for p in products) / len(products)
            }
        }
        
        # Save summary
        summary_file = self.data_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("📊 DATASET SUMMARY")
        print("="*50)
        print(f"Total Products: {summary['dataset_info']['total_products']:,}")
        print(f"Total Queries: {summary['dataset_info']['total_queries']:,}")
        print(f"Total Judgments: {summary['dataset_info']['total_judgments']:,}")
        print(f"Avg Judgments per Query: {summary['dataset_info']['avg_judgments_per_query']:.1f}")
        print(f"\nProduct Categories:")
        for category, count in summary['product_categories'].items():
            print(f"  {category}: {count:,}")
        print(f"\nJudgment Distribution:")
        for judgment, count in summary['judgment_distribution'].items():
            label = {0: "Not Relevant", 1: "Somewhat Relevant", 2: "Highly Relevant"}[judgment]
            print(f"  {label}: {count:,}")
        print(f"\nPrice Range: ${summary['price_statistics']['min_price']:.2f} - ${summary['price_statistics']['max_price']:.2f}")
        print(f"Average Price: ${summary['price_statistics']['avg_price']:.2f}")
        print("="*50)


def main():
    """Main function to download and process data."""
    print("🚀 Amazon ESCI Dataset Downloader")
    print("=================================\n")
    
    downloader = ESCIDataDownloader()
    
    try:
        # Download/generate data
        if downloader.download_esci_dataset():
            downloader.generate_summary_report()
            print("\n✅ Data download and processing completed successfully!")
            print(f"📁 Data location: {downloader.data_dir}")
            print("\nNext steps:")
            print("  1. Start Solr: cd demo/docker-setup && ./setup.sh")
            print("  2. Load data: python demo/scripts/load_data.py")
            print("  3. Run demo: python demo/run_complete_demo.py")
        else:
            print("\n❌ Failed to download data")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error during data download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
