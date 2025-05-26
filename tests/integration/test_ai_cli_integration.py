"""
Integration tests for AI CLI commands.

This module tests the integration between the CLI and the AI-powered
experiment management system.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from solr_optimizer.cli.main import (
    create_experiment_manager,
    cmd_ai_recommend,
    cmd_ai_preview,
    cmd_ai_optimize,
    cmd_ai_status,
)
from solr_optimizer.core.ai_experiment_manager import AIExperimentManager
from solr_optimizer.agents.ai.base_ai_agent import AgentRecommendation


class TestAICLIIntegration(unittest.TestCase):
    """Test AI CLI integration functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.solr_url = "http://localhost:8983/solr"

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_ai_experiment_manager(self):
        """Test creating AI-enabled experiment manager."""
        # Test without AI
        manager = create_experiment_manager(
            self.solr_url, self.temp_dir, enable_ai=False
        )
        self.assertNotIsInstance(manager, AIExperimentManager)

        # Test with AI
        manager = create_experiment_manager(
            self.solr_url, self.temp_dir, enable_ai=True
        )
        self.assertIsInstance(manager, AIExperimentManager)

    @patch('solr_optimizer.core.ai_experiment_manager.AIExperimentManager.get_ai_recommendation')
    def test_cmd_ai_recommend(self, mock_get_recommendation):
        """Test AI recommendation command."""
        # Mock recommendation
        mock_recommendation = AgentRecommendation(
            confidence=0.85,
            reasoning="Test recommendation for improved performance",
            suggested_changes={"qf": "title^2.0 content^1.0", "mm": "75%"},
            expected_impact="Expected 15% improvement in nDCG@10",
            risk_level="low",
            priority=8
        )
        mock_get_recommendation.return_value = mock_recommendation

        # Mock arguments
        args = MagicMock()
        args.verbose = False
        args.experiment_id = "test-exp"
        args.ai_model = "openai:gpt-4"
        args.constraints = ["max_risk=low", "focus=parameters"]
        args.solr_url = self.solr_url
        args.storage_dir = self.temp_dir

        # Capture output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            cmd_ai_recommend(args)
            output = captured_output.getvalue()
            
            # Verify output contains recommendation details
            self.assertIn("AI Optimization Recommendation", output)
            self.assertIn("Confidence: 0.85", output)
            self.assertIn("Risk Level: low", output)
            self.assertIn("Priority: 8/10", output)
            
        finally:
            sys.stdout = sys.__stdout__

    @patch('solr_optimizer.core.ai_experiment_manager.AIExperimentManager.preview_ai_recommendation')
    def test_cmd_ai_preview(self, mock_preview):
        """Test AI preview command."""
        # Mock preview data
        mock_preview_data = {
            "confidence": 0.85,
            "risk_level": "low",
            "priority": 8,
            "reasoning": "Test preview reasoning",
            "expected_impact": "Expected improvement",
            "preview_query_config": MagicMock()
        }
        mock_preview_data["preview_query_config"].iteration_id = "ai-test123"
        mock_preview_data["preview_query_config"].description = "AI generated config"
        mock_preview_data["preview_query_config"].query_parser = "edismax"
        mock_preview_data["preview_query_config"].qf = "title^2.0 content^1.0"
        mock_preview_data["preview_query_config"].pf = None
        mock_preview_data["preview_query_config"].mm = "75%"
        mock_preview_data["preview_query_config"].boost = None
        mock_preview_data["preview_query_config"].additional_params = {}

        mock_preview.return_value = mock_preview_data

        # Mock arguments
        args = MagicMock()
        args.verbose = False
        args.experiment_id = "test-exp"
        args.ai_model = "openai:gpt-4"
        args.constraints = []
        args.solr_url = self.solr_url
        args.storage_dir = self.temp_dir

        # Capture output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            cmd_ai_preview(args)
            output = captured_output.getvalue()
            
            # Verify output contains preview details
            self.assertIn("AI Optimization Preview", output)
            self.assertIn("Generated Query Configuration", output)
            self.assertIn("Query Fields (qf): title^2.0 content^1.0", output)
            
        finally:
            sys.stdout = sys.__stdout__

    @patch('solr_optimizer.core.ai_experiment_manager.AIExperimentManager.get_ai_status')
    def test_cmd_ai_status(self, mock_get_status):
        """Test AI status command."""
        # Mock status
        mock_status = {
            "ai_enabled": True,
            "ai_model": "openai:gpt-4",
            "orchestrator_available": True,
            "ai_config": {"temperature": 0.7}
        }
        mock_get_status.return_value = mock_status

        # Mock arguments
        args = MagicMock()
        args.verbose = False
        args.ai_model = "openai:gpt-4"
        args.solr_url = self.solr_url
        args.storage_dir = self.temp_dir

        # Capture output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            cmd_ai_status(args)
            output = captured_output.getvalue()
            
            # Verify output contains status details
            self.assertIn("AI System Status", output)
            self.assertIn("AI Enabled: True", output)
            self.assertIn("AI Model: openai:gpt-4", output)
            self.assertIn("Orchestrator Available: True", output)
            
        finally:
            sys.stdout = sys.__stdout__

    def test_constraint_parsing(self):
        """Test constraint parsing functionality."""
        args = MagicMock()
        args.constraints = ["max_risk=low", "focus=parameters", "temperature=0.7"]
        
        # Test constraint parsing logic
        constraints = {}
        for constraint in args.constraints:
            key, value = constraint.split('=', 1)
            constraints[key] = value
            
        expected_constraints = {
            "max_risk": "low",
            "focus": "parameters", 
            "temperature": "0.7"
        }
        
        self.assertEqual(constraints, expected_constraints)

    def test_ai_manager_error_handling(self):
        """Test error handling when AI functionality is not available."""
        # Create a non-AI manager
        non_ai_manager = create_experiment_manager(
            self.solr_url, self.temp_dir, enable_ai=False
        )
        self.assertNotIsInstance(non_ai_manager, AIExperimentManager)
        
        # Test that the non-AI manager doesn't have AI-specific methods
        self.assertFalse(hasattr(non_ai_manager, 'get_ai_recommendation'))
        
        # Test creation of AI manager (this should work)
        ai_manager = create_experiment_manager(
            self.solr_url, self.temp_dir, enable_ai=True
        )
        self.assertIsInstance(ai_manager, AIExperimentManager)
        self.assertTrue(hasattr(ai_manager, 'get_ai_recommendation'))


if __name__ == '__main__':
    unittest.main()
