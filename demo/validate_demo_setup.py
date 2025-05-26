#!/usr/bin/env python3
"""
Demo Setup Validation Script

This script validates the complete demo setup workflow without requiring
Docker to be running. It checks all the components and provides guidance.
"""

import os
import sys
import json
from pathlib import Path
import subprocess
import logging

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoSetupValidator:
    """Validates demo setup components and provides guidance."""
    
    def __init__(self):
        self.demo_dir = Path(__file__).parent
        self.project_root = self.demo_dir.parent
        self.docker_dir = self.demo_dir / "docker-setup"
        self.scripts_dir = self.demo_dir / "scripts"
        self.data_dir = self.demo_dir / "data"
        
    def validate_file_structure(self):
        """Validate that all required demo files exist."""
        logger.info("🔍 Validating demo file structure...")
        
        required_files = [
            # Docker setup
            self.docker_dir / "docker-compose.yml",
            self.docker_dir / "setup.sh",
            self.docker_dir / "teardown.sh",
            self.docker_dir / "solr-init" / "configsets" / "ecommerce" / "conf" / "schema.xml",
            self.docker_dir / "solr-init" / "configsets" / "ecommerce" / "conf" / "solrconfig.xml",
            self.docker_dir / "solr-init" / "configsets" / "ecommerce" / "conf" / "synonyms.txt",
            self.docker_dir / "solr-init" / "configsets" / "ecommerce" / "conf" / "stopwords.txt",
            
            # Scripts
            self.scripts_dir / "download_data.py",
            self.scripts_dir / "load_data.py",
            
            # Main demo files
            self.demo_dir / "run_complete_demo.py",
            self.demo_dir / "README.md",
            
            # Tests
            self.project_root / "tests" / "demo" / "test_end_to_end_demo.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            logger.error(f"❌ Missing required files:")
            for file_path in missing_files:
                logger.error(f"   - {file_path}")
            return False
        
        logger.info("✅ All required demo files exist")
        return True
    
    def validate_docker_setup(self):
        """Validate Docker setup configuration."""
        logger.info("🐳 Validating Docker setup...")
        
        # Check if Docker is available
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ Docker available: {result.stdout.strip()}")
            else:
                logger.warning("⚠️ Docker not available or not working")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("⚠️ Docker not found in PATH")
            return False
        
        # Check if Docker Compose is available
        compose_cmd = None
        for cmd in [["docker-compose", "--version"], ["docker", "compose", "version"]]:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    compose_cmd = cmd[0] if len(cmd) == 2 else "docker compose"
                    logger.info(f"✅ Docker Compose available: {compose_cmd}")
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        if not compose_cmd:
            logger.warning("⚠️ Docker Compose not available")
            return False
        
        # Validate docker-compose.yml
        compose_file = self.docker_dir / "docker-compose.yml"
        if compose_file.exists():
            logger.info("✅ docker-compose.yml exists and is readable")
        else:
            logger.error("❌ docker-compose.yml not found")
            return False
        
        return True
    
    def validate_framework_imports(self):
        """Validate that framework components can be imported."""
        logger.info("🔧 Validating framework imports...")
        
        try:
            # Test core imports
            from solr_optimizer.core.default_experiment_manager import DefaultExperimentManager
            from solr_optimizer.models.experiment_config import ExperimentConfig
            from solr_optimizer.agents.solr.pysolr_execution_agent import PySolrExecutionAgent
            from solr_optimizer.agents.metrics.standard_metrics_agent import StandardMetricsAgent
            from solr_optimizer.agents.logging.file_based_logging_agent import FileBasedLoggingAgent
            from solr_optimizer.agents.comparison.standard_comparison_agent import StandardComparisonAgent
            from solr_optimizer.agents.query.dummy_query_tuning_agent import DummyQueryTuningAgent
            
            logger.info("✅ All framework components import successfully")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Framework import error: {e}")
            return False
    
    def validate_python_dependencies(self):
        """Validate that required Python dependencies are available."""
        logger.info("🏗️ Validating Python dependencies...")
        
        required_modules = [
            "requests",
            "pandas", 
            "pysolr",
            "pytest"
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            logger.error(f"❌ Missing required Python modules: {', '.join(missing_modules)}")
            logger.error("   Install with: pip install -e .")
            return False
        
        logger.info("✅ All required Python dependencies available")
        return True
    
    def validate_scripts_executable(self):
        """Validate that setup scripts are executable."""
        logger.info("📜 Validating script permissions...")
        
        scripts = [
            self.docker_dir / "setup.sh",
            self.docker_dir / "teardown.sh"
        ]
        
        for script in scripts:
            if not script.exists():
                logger.error(f"❌ Script not found: {script}")
                return False
            
            if not os.access(script, os.X_OK):
                logger.warning(f"⚠️ Script not executable: {script}")
                logger.info(f"   Fix with: chmod +x {script}")
            else:
                logger.info(f"✅ Script executable: {script.name}")
        
        return True
    
    def generate_setup_guide(self):
        """Generate a setup guide based on validation results."""
        logger.info("📋 Generating setup guide...")
        
        guide = """
🚀 SOLR OPTIMIZER DEMO SETUP GUIDE
===================================

The demo implementation is complete! Follow these steps to run the demo:

1. SETUP ENVIRONMENT
   cd demo/docker-setup
   ./setup.sh
   
   This will:
   - Start 3 Solr nodes + 3 Zookeeper nodes
   - Create the 'ecommerce_products' collection
   - Configure schema and request handlers
   - Take ~2-3 minutes to complete

2. GENERATE DATA
   python demo/scripts/download_data.py
   
   This will:
   - Generate 10,000 realistic product records
   - Create 75 test queries with relevance judgments
   - Prepare data in multiple formats

3. LOAD DATA INTO SOLR
   python demo/scripts/load_data.py
   
   This will:
   - Load all product data into Solr
   - Verify data loading and query functionality
   - Show summary statistics

4. RUN THE DEMO
   python demo/run_complete_demo.py
   
   This will:
   - Run baseline iteration (basic Lucene)
   - Run basic optimization (DisMax)
   - Run advanced optimization (eDisMax)
   - Compare results and show improvements
   - Display detailed analysis

5. VALIDATE SETUP (Optional)
   python -m pytest tests/demo/test_end_to_end_demo.py -v
   
   This will run comprehensive validation tests.

EXPECTED RESULTS:
- Baseline NDCG@10: ~0.30-0.40
- Basic optimization: +15-25% improvement
- Advanced optimization: +10-15% additional improvement

TROUBLESHOOTING:
- If Docker issues: Restart Docker, check available memory (4GB+ needed)
- If port conflicts: Check ports 8983-8985, 2181-2183 are free
- If import errors: Run 'pip install -e .' from project root
- For help: Check demo/README.md for detailed instructions

🎉 Ready to run! The demo showcases real-world Solr optimization with 
   AI-powered analysis and detailed performance comparisons.
"""
        
        print(guide)
    
    def run_validation(self):
        """Run complete validation and provide guidance."""
        print("🔍 SOLR OPTIMIZER DEMO VALIDATION")
        print("==================================\n")
        
        checks = [
            ("File structure", self.validate_file_structure()),
            ("Docker setup", self.validate_docker_setup()),
            ("Framework imports", self.validate_framework_imports()),
            ("Python dependencies", self.validate_python_dependencies()),
            ("Script permissions", self.validate_scripts_executable())
        ]
        
        passed = 0
        for check_name, result in checks:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {check_name:<20} {status}")
            if result:
                passed += 1
        
        print(f"\nValidation Results: {passed}/{len(checks)} checks passed\n")
        
        if passed == len(checks):
            print("🎉 All validation checks passed!")
            print("Demo is ready to run - follow the setup guide below.\n")
        else:
            print("⚠️ Some validation checks failed.")
            print("Address the issues above before running the demo.\n")
        
        self.generate_setup_guide()


def main():
    """Main validation function."""
    validator = DemoSetupValidator()
    validator.run_validation()


if __name__ == "__main__":
    main()
