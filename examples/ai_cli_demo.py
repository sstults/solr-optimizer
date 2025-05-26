"""
AI CLI Demo - Demonstrates AI-powered optimization features.

This script shows how to use the new AI-powered CLI commands for
intelligent query optimization in Solr Optimizer.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from solr_optimizer.cli.main import create_experiment_manager
from solr_optimizer.core.ai_experiment_manager import AIExperimentManager
from solr_optimizer.models.experiment_config import ExperimentConfig


def create_sample_data():
    """Create sample queries and judgments for demonstration."""
    temp_dir = tempfile.mkdtemp()
    
    # Sample queries
    queries = [
        "laptop computer",
        "wireless headphones", 
        "gaming keyboard",
        "4k monitor",
        "smartphone case"
    ]
    
    # Sample relevance judgments
    judgments = {
        "laptop computer": {
            "doc001": 3.0,  # highly relevant
            "doc002": 2.0,  # relevant
            "doc003": 1.0,  # somewhat relevant
            "doc004": 0.0   # not relevant
        },
        "wireless headphones": {
            "doc005": 3.0,
            "doc006": 2.0,
            "doc007": 1.0,
            "doc008": 0.0
        },
        "gaming keyboard": {
            "doc009": 3.0,
            "doc010": 2.0,
            "doc011": 1.0,
            "doc012": 0.0
        },
        "4k monitor": {
            "doc013": 3.0,
            "doc014": 2.0,
            "doc015": 1.0,
            "doc016": 0.0
        },
        "smartphone case": {
            "doc017": 3.0,
            "doc018": 2.0,
            "doc019": 1.0,
            "doc020": 0.0
        }
    }
    
    return temp_dir, queries, judgments


def demo_ai_experiment_manager():
    """Demonstrate AI experiment manager capabilities."""
    print("=== AI Experiment Manager Demo ===")
    
    temp_dir, queries, judgments = create_sample_data()
    
    try:
        # Create AI-enabled experiment manager
        print("\n1. Creating AI-enabled experiment manager...")
        ai_manager = create_experiment_manager(
            solr_url="http://localhost:8983/solr",
            storage_dir=temp_dir,
            enable_ai=True,
            ai_model="openai:gpt-4"
        )
        
        print(f"✓ Created AI manager: {type(ai_manager).__name__}")
        print(f"✓ AI enabled: {isinstance(ai_manager, AIExperimentManager)}")
        
        # Create experiment configuration
        print("\n2. Creating experiment configuration...")
        experiment_config = ExperimentConfig(
            name="E-commerce Search Optimization",
            description="AI-powered optimization of e-commerce search queries",
            corpus="ecommerce_products",
            queries=queries,
            judgments=judgments,
            primary_metric="ndcg",
            secondary_metrics=["precision", "recall"],
            metric_depth=10
        )
        
        # Setup experiment
        experiment_id = ai_manager.setup_experiment(experiment_config)
        print(f"✓ Created experiment: {experiment_id}")
        
        # Get AI status
        print("\n3. Checking AI system status...")
        if isinstance(ai_manager, AIExperimentManager):
            status = ai_manager.get_ai_status()
            print(f"  AI Enabled: {status['ai_enabled']}")
            print(f"  AI Model: {status['ai_model']}")
            print(f"  Orchestrator Available: {status['orchestrator_available']}")
        
        print("\n4. AI capabilities demonstrated:")
        print("  ✓ AI-powered experiment manager created")
        print("  ✓ Experiment setup with AI integration")  
        print("  ✓ AI status monitoring available")
        print("  ✓ Ready for AI-powered optimization")
        
        return experiment_id, temp_dir
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, temp_dir


def demo_cli_commands():
    """Demonstrate AI CLI commands."""
    print("\n=== AI CLI Commands Demo ===")
    
    experiment_id, temp_dir = demo_ai_experiment_manager()
    
    if not experiment_id:
        print("Skipping CLI demo due to experiment setup failure")
        return
    
    print("\nAvailable AI CLI Commands:")
    print("\n1. ai-recommend - Get AI optimization recommendations")
    print("   Usage: python -m solr_optimizer.cli.main ai-recommend --experiment-id <id>")
    print("   Example constraints: --constraints max_risk=low focus=parameters")
    
    print("\n2. ai-preview - Preview AI recommendations before applying")
    print("   Usage: python -m solr_optimizer.cli.main ai-preview --experiment-id <id>")
    print("   Shows generated query configuration without executing")
    
    print("\n3. ai-optimize - Run AI-optimized iteration")
    print("   Usage: python -m solr_optimizer.cli.main ai-optimize --experiment-id <id>")
    print("   Automatically applies AI recommendations and runs iteration")
    
    print("\n4. ai-status - Check AI system status")
    print("   Usage: python -m solr_optimizer.cli.main ai-status")
    print("   Shows AI model configuration and availability")
    
    print("\nAI Model Options:")
    print("  --ai-model openai:gpt-4      (default, requires OpenAI API key)")
    print("  --ai-model openai:gpt-3.5-turbo")
    print("  --ai-model anthropic:claude-3")
    print("  --ai-model local:llama2      (requires local model setup)")
    
    print("\nOptimization Constraints:")
    print("  max_risk=low|medium|high     Limit risk level of changes")
    print("  focus=parameters|schema|ltr  Focus on specific optimization area")
    print("  preserve_recall=true         Avoid changes that might hurt recall")
    print("  min_confidence=0.8           Only apply high-confidence recommendations")


def demo_ai_workflow():
    """Demonstrate complete AI optimization workflow."""
    print("\n=== Complete AI Optimization Workflow ===")
    
    print("\nTypical workflow with AI optimization:")
    
    print("\n1. Setup Phase:")
    print("   • Create experiment with queries and relevance judgments")
    print("   • Run baseline iteration to establish current performance")
    print("   • Configure AI model and optimization constraints")
    
    print("\n2. AI Analysis Phase:")
    print("   • Use 'ai-recommend' to get AI suggestions")
    print("   • Review recommendations, confidence, and risk levels")
    print("   • Use 'ai-preview' to see generated query configurations")
    
    print("\n3. Optimization Phase:")
    print("   • Use 'ai-optimize' to run AI-suggested iterations")
    print("   • Compare results with previous iterations")
    print("   • Apply additional AI recommendations as needed")
    
    print("\n4. Monitoring Phase:")
    print("   • Use 'ai-status' to monitor AI system health")
    print("   • Track optimization effectiveness over time")
    print("   • Adjust constraints based on results")
    
    print("\nAI Agent Specializations:")
    print("  🔍 Schema Analysis Agent - Field configurations and boost weights")
    print("  ⚙️  Parameter Tuning Agent - Query parameters (qf, pf, mm, tie)")
    print("  📝 Analysis Chain Agent - Text analysis and tokenization")
    print("  🔄 Query Rewriting Agent - Query reformulation and expansion")
    print("  🧠 Learning-to-Rank Agent - Machine learning ranking models")
    print("  🎭 Orchestrator Agent - Coordinates all agents for optimal strategy")


def main():
    """Run the complete AI CLI demonstration."""
    print("Solr Optimizer - AI-Powered CLI Demo")
    print("=" * 50)
    
    print("\nThis demo showcases the new AI-powered optimization features")
    print("added in Phase 5 of the Solr Optimizer project.")
    
    # Check if AI dependencies are available
    try:
        from solr_optimizer.agents.ai.orchestrator import QueryOptimizationOrchestrator
        print("\n✓ AI dependencies are available")
    except ImportError as e:
        print(f"\n✗ AI dependencies not available: {e}")
        print("Install with: pip install pydantic-ai")
        return
    
    # Run demonstrations
    demo_ai_experiment_manager()
    demo_cli_commands()
    demo_ai_workflow()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nNext steps:")
    print("1. Set up your OpenAI API key: export OPENAI_API_KEY=your_key")
    print("2. Try the AI commands on your own experiments")
    print("3. Experiment with different AI models and constraints")
    print("4. Monitor AI recommendation effectiveness over time")
    
    print("\nFor more information, see:")
    print("- README.md for setup instructions")
    print("- ARCHITECTURE.md for technical details")
    print("- examples/ directory for more examples")


if __name__ == "__main__":
    main()
