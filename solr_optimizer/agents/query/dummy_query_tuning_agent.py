"""
Dummy Query Tuning Agent - Basic implementation for testing purposes.

This module provides a concrete implementation of the QueryTuningAgent interface
with minimal functionality for testing and development purposes.
"""

import random
from typing import Any, Dict, List

from solr_optimizer.agents.query.query_tuning_agent import QueryTuningAgent
from solr_optimizer.models.experiment_config import ExperimentConfig
from solr_optimizer.models.iteration_result import IterationResult
from solr_optimizer.models.query_config import QueryConfig


class DummyQueryTuningAgent(QueryTuningAgent):
    """
    A basic implementation of QueryTuningAgent for testing purposes.

    This agent provides realistic implementations of all required methods
    that allow the system to run with actual query optimization logic for testing.
    """

    def __init__(self):
        """Initialize the dummy agent."""
        # Text field types that typically contain searchable text
        self.text_field_types = {
            "text_general", "text", "text_en", "text_ws", 
            "TextField", "solr.TextField"
        }

    def analyze_schema(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Solr schema to identify potentially useful fields.

        Args:
            schema_info: Schema information from Solr

        Returns:
            Dictionary with analysis results including text_fields, indexed_fields, stored_fields
        """
        analysis = {
            "text_fields": [],
            "indexed_fields": [],
            "stored_fields": []
        }

        if not schema_info or "schema" not in schema_info:
            return analysis

        schema = schema_info["schema"]
        fields = schema.get("fields", [])
        field_types = schema.get("fieldTypes", [])

        # Create a mapping of field type names to their classes
        type_class_map = {}
        for field_type in field_types:
            type_class_map[field_type.get("name", "")] = field_type.get("class", "")

        # Analyze each field
        for field in fields:
            field_name = field.get("name", "")
            field_type = field.get("type", "")
            is_indexed = field.get("indexed", False)
            is_stored = field.get("stored", False)

            # Check if field is indexed
            if is_indexed:
                analysis["indexed_fields"].append(field_name)

            # Check if field is stored
            if is_stored:
                analysis["stored_fields"].append(field_name)

            # Check if field is a text field
            field_class = type_class_map.get(field_type, "")
            if (field_type in self.text_field_types or 
                field_class in self.text_field_types or
                "text" in field_type.lower()):
                analysis["text_fields"].append(field_name)

        return analysis

    def generate_initial_config(self, experiment_config: ExperimentConfig, schema_info: Dict[str, Any]) -> QueryConfig:
        """
        Generate an initial query configuration.

        Args:
            experiment_config: The experiment configuration
            schema_info: Schema information from Solr

        Returns:
            QueryConfig object with realistic initial configuration
        """
        # Analyze schema to find text fields
        analysis = self.analyze_schema(schema_info)
        text_fields = analysis.get("text_fields", [])

        # Create query fields mapping with random boosts
        query_fields = {}
        if text_fields:
            for field in text_fields:
                # Give title-like fields higher boosts
                if "title" in field.lower():
                    boost = random.uniform(1.5, 3.0)
                elif "content" in field.lower() or "body" in field.lower():
                    boost = random.uniform(0.8, 1.5)
                else:
                    boost = random.uniform(0.5, 2.0)
                query_fields[field] = round(boost, 1)

        # Create phrase fields (subset of query fields with lower boosts)
        phrase_fields = {}
        if text_fields:
            for field in text_fields[:2]:  # Use first 2 text fields for phrase matching
                phrase_fields[field] = round(random.uniform(0.1, 0.5), 1)

        # Random tie breaker
        tie_breaker = round(random.uniform(0.0, 1.0), 2)

        # Random minimum match
        minimum_match = None
        if random.random() < 0.5:  # 50% chance of setting MM
            if random.random() < 0.7:  # 70% chance of percentage
                minimum_match = f"{random.randint(50, 90)}%"
            else:  # 30% chance of number
                minimum_match = str(random.randint(1, 3))

        return QueryConfig(
            query_parser="edismax",
            query_fields=query_fields,
            phrase_fields=phrase_fields,
            tie_breaker=tie_breaker,
            minimum_match=minimum_match
        )

    def suggest_next_config(self, previous_result: IterationResult, schema_info: Dict[str, Any]) -> QueryConfig:
        """
        Suggest a new query configuration based on previous results.

        Args:
            previous_result: The results of the previous iteration
            schema_info: Schema information from Solr

        Returns:
            Modified QueryConfig object
        """
        # Start with the previous config
        prev_config = previous_result.query_config
        
        # Analyze schema to get available fields
        analysis = self.analyze_schema(schema_info)
        text_fields = analysis.get("text_fields", [])

        # Make random modifications to the previous config
        new_query_fields = dict(prev_config.query_fields)
        new_phrase_fields = dict(prev_config.phrase_fields)

        # Randomly adjust field boosts
        for field in new_query_fields:
            if random.random() < 0.3:  # 30% chance to modify each field
                new_query_fields[field] = round(random.uniform(0.5, 3.0), 1)

        # Randomly add or remove fields
        if text_fields:
            # Maybe add a new field
            if random.random() < 0.2:  # 20% chance
                new_field = random.choice(text_fields)
                if new_field not in new_query_fields:
                    new_query_fields[new_field] = round(random.uniform(0.5, 2.0), 1)

            # Maybe remove a field
            if len(new_query_fields) > 1 and random.random() < 0.1:  # 10% chance
                field_to_remove = random.choice(list(new_query_fields.keys()))
                del new_query_fields[field_to_remove]

        # Adjust tie breaker
        new_tie_breaker = prev_config.tie_breaker
        if random.random() < 0.3:  # 30% chance to change tie breaker
            new_tie_breaker = round(random.uniform(0.0, 1.0), 2)

        # Adjust minimum match
        new_minimum_match = prev_config.minimum_match
        if random.random() < 0.2:  # 20% chance to change MM
            if random.random() < 0.7:
                new_minimum_match = f"{random.randint(50, 90)}%"
            else:
                new_minimum_match = str(random.randint(1, 3))

        return QueryConfig(
            query_parser=prev_config.query_parser,
            query_fields=new_query_fields,
            phrase_fields=new_phrase_fields,
            boost_queries=list(prev_config.boost_queries),
            boost_functions=list(prev_config.boost_functions),
            minimum_match=new_minimum_match,
            tie_breaker=new_tie_breaker
        )

    def adjust_parameters(self, result: IterationResult, target_metric: str, direction: str) -> QueryConfig:
        """
        Adjust specific parameters to improve a target metric.

        Args:
            result: The results of the previous iteration
            target_metric: The metric to optimize (e.g., 'ndcg@10')
            direction: Either 'increase' or 'decrease'

        Returns:
            Modified QueryConfig object
        """
        # Start with the previous config
        prev_config = result.query_config

        # Make random adjustments based on direction
        new_query_fields = dict(prev_config.query_fields)
        
        # Adjust field boosts
        for field in new_query_fields:
            current_boost = new_query_fields[field]
            if direction == "increase":
                # Increase boosts to try to improve metric
                adjustment = random.uniform(0.1, 0.5)
                new_query_fields[field] = min(10.0, round(current_boost + adjustment, 1))
            elif direction == "decrease":
                # Decrease boosts 
                adjustment = random.uniform(0.1, 0.3)
                new_query_fields[field] = max(0.1, round(current_boost - adjustment, 1))
            else:
                # Unknown direction, make random adjustment
                adjustment = random.uniform(-0.3, 0.3)
                new_query_fields[field] = max(0.1, min(10.0, round(current_boost + adjustment, 1)))

        # Adjust tie breaker
        new_tie_breaker = prev_config.tie_breaker
        if direction == "increase":
            new_tie_breaker = min(1.0, round(new_tie_breaker + random.uniform(0.1, 0.2), 2))
        elif direction == "decrease":
            new_tie_breaker = max(0.0, round(new_tie_breaker - random.uniform(0.1, 0.2), 2))

        return QueryConfig(
            query_parser=prev_config.query_parser,
            query_fields=new_query_fields,
            phrase_fields=dict(prev_config.phrase_fields),
            boost_queries=list(prev_config.boost_queries),
            boost_functions=list(prev_config.boost_functions),
            minimum_match=prev_config.minimum_match,
            tie_breaker=new_tie_breaker
        )
