#!/usr/bin/env python3
"""
Solr Schema Validation

This module provides comprehensive validation of Solr schema configuration
to ensure proper setup for the optimizer demo.
"""

import requests
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation severity levels."""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    level: ValidationLevel
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None


class SolrSchemaValidator:
    """Comprehensive Solr schema validator."""
    
    def __init__(self, solr_url: str, collection: str):
        self.solr_url = solr_url.rstrip('/')
        self.collection = collection
        self.logger = logging.getLogger("SolrSchemaValidator")
        self.validation_results = []
        
        # Expected fields for ecommerce demo
        self.expected_fields = [
            "product_id", "product_title", "product_description", 
            "product_bullet_point", "product_brand", "product_color",
            "category", "query", "_text_"
        ]
        
        # Required copy fields for proper search
        self.expected_copy_fields = {
            "_text_": ["product_title", "product_description", "product_bullet_point", 
                      "product_brand", "product_color", "category", "query"]
        }
    
    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """Run all validation checks and return overall result."""
        self.validation_results = []
        
        # Basic connectivity
        if not self._validate_solr_connectivity():
            return False, self.validation_results
        
        # Schema validation
        schema_data = self._get_schema_data()
        if not schema_data:
            return False, self.validation_results
        
        # Run schema checks
        self._validate_required_fields(schema_data)
        self._validate_copy_fields(schema_data)
        self._validate_field_types(schema_data)
        self._validate_text_analysis(schema_data)
        self._validate_unique_key(schema_data)
        
        # Configuration validation
        config_data = self._get_config_data()
        if config_data:
            self._validate_search_handlers(config_data)
            self._validate_default_search_field(config_data)
        
        # Data validation
        self._validate_collection_data()
        
        # Determine overall pass/fail
        critical_failures = [r for r in self.validation_results 
                           if r.level == ValidationLevel.CRITICAL and not r.passed]
        overall_pass = len(critical_failures) == 0
        
        return overall_pass, self.validation_results
    
    def _validate_solr_connectivity(self) -> bool:
        """Validate basic Solr connectivity."""
        try:
            response = requests.get(f"{self.solr_url}/admin/ping", timeout=10)
            response.raise_for_status()
            
            self.validation_results.append(ValidationResult(
                check_name="Solr Connectivity",
                level=ValidationLevel.CRITICAL,
                passed=True,
                message="Successfully connected to Solr"
            ))
            return True
            
        except Exception as e:
            self.validation_results.append(ValidationResult(
                check_name="Solr Connectivity",
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"Failed to connect to Solr: {e}",
                recommendation="Ensure Solr is running and accessible at the configured URL"
            ))
            return False
    
    def _get_schema_data(self) -> Optional[Dict[str, Any]]:
        """Retrieve schema data from Solr."""
        try:
            response = requests.get(f"{self.solr_url}/{self.collection}/schema")
            response.raise_for_status()
            
            schema_data = response.json()
            
            self.validation_results.append(ValidationResult(
                check_name="Schema Access",
                level=ValidationLevel.CRITICAL,
                passed=True,
                message="Successfully retrieved schema configuration"
            ))
            
            return schema_data.get('schema', {})
            
        except Exception as e:
            self.validation_results.append(ValidationResult(
                check_name="Schema Access",
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"Failed to retrieve schema: {e}",
                recommendation="Ensure collection exists and is accessible"
            ))
            return None
    
    def _get_config_data(self) -> Optional[Dict[str, Any]]:
        """Retrieve configuration data from Solr."""
        try:
            response = requests.get(f"{self.solr_url}/{self.collection}/config")
            response.raise_for_status()
            
            config_data = response.json()
            
            self.validation_results.append(ValidationResult(
                check_name="Config Access",
                level=ValidationLevel.INFO,
                passed=True,
                message="Successfully retrieved configuration"
            ))
            
            return config_data.get('config', {})
            
        except Exception as e:
            self.validation_results.append(ValidationResult(
                check_name="Config Access",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Failed to retrieve config: {e}",
                recommendation="Config access not critical but recommended for full validation"
            ))
            return None
    
    def _validate_required_fields(self, schema_data: Dict[str, Any]):
        """Validate that all required fields are present."""
        fields = schema_data.get('fields', [])
        field_names = [field.get('name') for field in fields]
        
        missing_fields = []
        present_fields = []
        
        for expected_field in self.expected_fields:
            if expected_field in field_names:
                present_fields.append(expected_field)
            else:
                missing_fields.append(expected_field)
        
        if missing_fields:
            self.validation_results.append(ValidationResult(
                check_name="Required Fields",
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"Missing required fields: {', '.join(missing_fields)}",
                details={
                    "missing_fields": missing_fields,
                    "present_fields": present_fields,
                    "total_fields": len(field_names)
                },
                recommendation="Add missing fields to schema.xml or update data loading to include these fields"
            ))
        else:
            self.validation_results.append(ValidationResult(
                check_name="Required Fields",
                level=ValidationLevel.CRITICAL,
                passed=True,
                message="All required fields are present",
                details={
                    "present_fields": present_fields,
                    "total_fields": len(field_names)
                }
            ))
    
    def _validate_copy_fields(self, schema_data: Dict[str, Any]):
        """Validate copy field configuration."""
        copy_fields = schema_data.get('copyFields', [])
        
        # Build copy field mapping
        copy_mapping = {}
        for copy_field in copy_fields:
            source = copy_field.get('source')
            dest = copy_field.get('dest')
            if dest not in copy_mapping:
                copy_mapping[dest] = []
            copy_mapping[dest].append(source)
        
        # Check expected copy fields
        for dest_field, expected_sources in self.expected_copy_fields.items():
            if dest_field not in copy_mapping:
                self.validation_results.append(ValidationResult(
                    check_name=f"Copy Field {dest_field}",
                    level=ValidationLevel.CRITICAL,
                    passed=False,
                    message=f"Copy field destination '{dest_field}' not found",
                    recommendation=f"Add copy fields to populate '{dest_field}' from source fields"
                ))
                continue
            
            actual_sources = copy_mapping[dest_field]
            missing_sources = [src for src in expected_sources if src not in actual_sources]
            
            if missing_sources:
                self.validation_results.append(ValidationResult(
                    check_name=f"Copy Field {dest_field}",
                    level=ValidationLevel.WARNING,
                    passed=False,
                    message=f"Copy field '{dest_field}' missing sources: {', '.join(missing_sources)}",
                    details={
                        "expected_sources": expected_sources,
                        "actual_sources": actual_sources,
                        "missing_sources": missing_sources
                    },
                    recommendation="Add missing copy field sources for better search coverage"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    check_name=f"Copy Field {dest_field}",
                    level=ValidationLevel.INFO,
                    passed=True,
                    message=f"Copy field '{dest_field}' properly configured",
                    details={
                        "sources": actual_sources
                    }
                ))
    
    def _validate_field_types(self, schema_data: Dict[str, Any]):
        """Validate field type configuration."""
        field_types = schema_data.get('fieldTypes', [])
        fields = schema_data.get('fields', [])
        
        # Check for text field types
        text_field_types = []
        for field_type in field_types:
            name = field_type.get('name', '')
            if 'text' in name.lower():
                text_field_types.append(name)
        
        if not text_field_types:
            self.validation_results.append(ValidationResult(
                check_name="Text Field Types",
                level=ValidationLevel.WARNING,
                passed=False,
                message="No text field types found",
                recommendation="Ensure text field types are defined for proper text analysis"
            ))
        else:
            self.validation_results.append(ValidationResult(
                check_name="Text Field Types",
                level=ValidationLevel.INFO,
                passed=True,
                message=f"Found text field types: {', '.join(text_field_types)}"
            ))
        
        # Check field type usage
        field_type_usage = {}
        for field in fields:
            field_type = field.get('type', '')
            if field_type not in field_type_usage:
                field_type_usage[field_type] = []
            field_type_usage[field_type].append(field.get('name', ''))
        
        self.validation_results.append(ValidationResult(
            check_name="Field Type Usage",
            level=ValidationLevel.INFO,
            passed=True,
            message=f"Field types in use: {len(field_type_usage)}",
            details={"field_type_usage": field_type_usage}
        ))
    
    def _validate_text_analysis(self, schema_data: Dict[str, Any]):
        """Validate text analysis configuration."""
        field_types = schema_data.get('fieldTypes', [])
        
        # Look for text analysis components
        tokenizers_found = []
        filters_found = []
        
        for field_type in field_types:
            name = field_type.get('name', '')
            if 'text' in name.lower():
                analyzer = field_type.get('analyzer', {})
                
                # Check tokenizer
                tokenizer = analyzer.get('tokenizer', {})
                if tokenizer:
                    tokenizer_class = tokenizer.get('class', '')
                    if tokenizer_class:
                        tokenizers_found.append(tokenizer_class)
                
                # Check filters
                filters = analyzer.get('filters', [])
                for f in filters:
                    filter_class = f.get('class', '')
                    if filter_class:
                        filters_found.append(filter_class)
        
        # Validate essential components
        essential_components = [
            'solr.StandardTokenizerFactory',
            'solr.LowerCaseFilterFactory',
            'solr.StopFilterFactory'
        ]
        
        missing_components = []
        for component in essential_components:
            if component not in tokenizers_found and component not in filters_found:
                missing_components.append(component)
        
        if missing_components:
            self.validation_results.append(ValidationResult(
                check_name="Text Analysis",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Missing essential analysis components: {', '.join(missing_components)}",
                details={
                    "tokenizers_found": tokenizers_found,
                    "filters_found": filters_found,
                    "missing_components": missing_components
                },
                recommendation="Add missing text analysis components for better search quality"
            ))
        else:
            self.validation_results.append(ValidationResult(
                check_name="Text Analysis",
                level=ValidationLevel.INFO,
                passed=True,
                message="Essential text analysis components found",
                details={
                    "tokenizers_found": tokenizers_found,
                    "filters_found": filters_found
                }
            ))
    
    def _validate_unique_key(self, schema_data: Dict[str, Any]):
        """Validate unique key configuration."""
        unique_key = schema_data.get('uniqueKey')
        
        if not unique_key:
            self.validation_results.append(ValidationResult(
                check_name="Unique Key",
                level=ValidationLevel.CRITICAL,
                passed=False,
                message="No unique key field defined",
                recommendation="Define a unique key field in schema.xml"
            ))
        else:
            # Check if unique key field exists
            fields = schema_data.get('fields', [])
            field_names = [field.get('name') for field in fields]
            
            if unique_key in field_names:
                self.validation_results.append(ValidationResult(
                    check_name="Unique Key",
                    level=ValidationLevel.INFO,
                    passed=True,
                    message=f"Unique key '{unique_key}' properly defined"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    check_name="Unique Key",
                    level=ValidationLevel.CRITICAL,
                    passed=False,
                    message=f"Unique key field '{unique_key}' not found in schema",
                    recommendation="Add the unique key field to schema or update unique key definition"
                ))
    
    def _validate_search_handlers(self, config_data: Dict[str, Any]):
        """Validate search handler configuration."""
        request_handlers = config_data.get('requestHandler', {})
        
        # Check for essential search handlers
        essential_handlers = ['/select', '/query']
        found_handlers = []
        
        for handler_name in request_handlers.keys():
            if any(essential in handler_name for essential in essential_handlers):
                found_handlers.append(handler_name)
        
        if not found_handlers:
            self.validation_results.append(ValidationResult(
                check_name="Search Handlers",
                level=ValidationLevel.WARNING,
                passed=False,
                message="No search handlers found",
                recommendation="Ensure search handlers are properly configured in solrconfig.xml"
            ))
        else:
            self.validation_results.append(ValidationResult(
                check_name="Search Handlers",
                level=ValidationLevel.INFO,
                passed=True,
                message=f"Found search handlers: {', '.join(found_handlers)}"
            ))
    
    def _validate_default_search_field(self, config_data: Dict[str, Any]):
        """Validate default search field configuration."""
        request_handlers = config_data.get('requestHandler', {})
        
        default_fields = []
        for handler_name, handler_config in request_handlers.items():
            if '/select' in handler_name or '/query' in handler_name:
                defaults = handler_config.get('defaults', {})
                df = defaults.get('df')
                if df:
                    default_fields.append((handler_name, df))
        
        if not default_fields:
            self.validation_results.append(ValidationResult(
                check_name="Default Search Field",
                level=ValidationLevel.WARNING,
                passed=False,
                message="No default search field (df) configured",
                recommendation="Configure default search field in search handlers"
            ))
        else:
            # Check if default field is _text_ (recommended)
            text_field_configured = any(df == '_text_' for _, df in default_fields)
            
            if text_field_configured:
                self.validation_results.append(ValidationResult(
                    check_name="Default Search Field",
                    level=ValidationLevel.INFO,
                    passed=True,
                    message="Default search field properly configured to '_text_'",
                    details={"configured_fields": default_fields}
                ))
            else:
                self.validation_results.append(ValidationResult(
                    check_name="Default Search Field",
                    level=ValidationLevel.WARNING,
                    passed=False,
                    message=f"Default search field not set to '_text_': {default_fields}",
                    details={"configured_fields": default_fields},
                    recommendation="Set default search field (df) to '_text_' for best results"
                ))
    
    def _validate_collection_data(self):
        """Validate that collection has data."""
        try:
            response = requests.get(
                f"{self.solr_url}/{self.collection}/select",
                params={"q": "*:*", "rows": 0, "wt": "json"}
            )
            response.raise_for_status()
            
            result = response.json()
            num_docs = result.get('response', {}).get('numFound', 0)
            
            if num_docs == 0:
                self.validation_results.append(ValidationResult(
                    check_name="Collection Data",
                    level=ValidationLevel.CRITICAL,
                    passed=False,
                    message="Collection contains no documents",
                    recommendation="Load data into the collection before running optimization"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    check_name="Collection Data",
                    level=ValidationLevel.INFO,
                    passed=True,
                    message=f"Collection contains {num_docs} documents"
                ))
                
        except Exception as e:
            self.validation_results.append(ValidationResult(
                check_name="Collection Data",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Failed to check collection data: {e}",
                recommendation="Ensure collection is accessible and contains data"
            ))
    
    def print_validation_report(self):
        """Print a detailed validation report."""
        print("🔍 SOLR SCHEMA VALIDATION REPORT")
        print("=" * 60)
        
        # Summary counts
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r.passed)
        critical_failures = sum(1 for r in self.validation_results 
                              if r.level == ValidationLevel.CRITICAL and not r.passed)
        warnings = sum(1 for r in self.validation_results 
                      if r.level == ValidationLevel.WARNING and not r.passed)
        
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Critical Failures: {critical_failures}")
        print(f"Warnings: {warnings}")
        print()
        
        # Group results by level
        for level in [ValidationLevel.CRITICAL, ValidationLevel.WARNING, ValidationLevel.INFO]:
            level_results = [r for r in self.validation_results if r.level == level]
            if not level_results:
                continue
            
            print(f"{level.value} CHECKS:")
            print("-" * 30)
            
            for result in level_results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"{status} {result.check_name}: {result.message}")
                
                if not result.passed and result.recommendation:
                    print(f"    💡 Recommendation: {result.recommendation}")
                
                if result.details:
                    for key, value in result.details.items():
                        if isinstance(value, list) and len(value) > 5:
                            print(f"    📋 {key}: {len(value)} items")
                        else:
                            print(f"    📋 {key}: {value}")
                print()
        
        # Overall assessment
        if critical_failures == 0:
            print("🎉 OVERALL: VALIDATION PASSED")
            if warnings > 0:
                print(f"⚠️  Note: {warnings} warnings should be addressed for optimal performance")
        else:
            print("🚨 OVERALL: VALIDATION FAILED")
            print(f"❌ {critical_failures} critical issues must be fixed before proceeding")


def validate_schema_setup(solr_url: str = "http://localhost:8983/solr",
                         collection: str = "ecommerce_products") -> bool:
    """Convenience function to validate schema setup."""
    validator = SolrSchemaValidator(solr_url, collection)
    passed, results = validator.validate_all()
    validator.print_validation_report()
    return passed


if __name__ == "__main__":
    # Run validation
    success = validate_schema_setup()
    exit(0 if success else 1)
