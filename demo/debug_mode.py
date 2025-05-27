#!/usr/bin/env python3
"""
Solr Optimizer Debug Mode

This module provides debugging capabilities for the Solr Optimizer demo,
including query logging, parameter inspection, and detailed configuration analysis.
"""

import json
import logging
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass, asdict


@dataclass
class SolrQueryDebugInfo:
    """Container for Solr query debug information."""
    query_string: str
    solr_url: str
    collection: str
    handler: str
    parameters: Dict[str, Any]
    execution_time_ms: float
    num_found: int
    raw_response: Dict[str, Any]
    debug_info: Dict[str, Any]
    timestamp: str


class SolrDebugLogger:
    """Enhanced logging for Solr query debugging."""
    
    def __init__(self, log_level: str = "DEBUG", log_file: Optional[str] = None):
        self.logger = logging.getLogger("SolrDebugLogger")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.query_log = []
    
    def log_query_execution(self, debug_info: SolrQueryDebugInfo):
        """Log detailed query execution information."""
        self.query_log.append(debug_info)
        
        self.logger.info("="*80)
        self.logger.info(f"SOLR QUERY EXECUTION #{len(self.query_log)}")
        self.logger.info("="*80)
        self.logger.info(f"Query: {debug_info.query_string}")
        self.logger.info(f"Collection: {debug_info.collection}")
        self.logger.info(f"Handler: {debug_info.handler}")
        self.logger.info(f"Execution Time: {debug_info.execution_time_ms:.2f}ms")
        self.logger.info(f"Results Found: {debug_info.num_found}")
        self.logger.info("-"*50)
        
        # Log parameters
        self.logger.info("QUERY PARAMETERS:")
        for key, value in sorted(debug_info.parameters.items()):
            self.logger.info(f"  {key}: {value}")
        
        # Log debug information if available
        if debug_info.debug_info:
            self.logger.info("-"*50)
            self.logger.info("SOLR DEBUG INFO:")
            self._log_debug_details(debug_info.debug_info)
        
        self.logger.info("="*80)
        self.logger.info("")
    
    def _log_debug_details(self, debug_info: Dict[str, Any], indent: int = 2):
        """Recursively log debug information with proper indentation."""
        spaces = " " * indent
        
        for key, value in debug_info.items():
            if isinstance(value, dict):
                self.logger.info(f"{spaces}{key}:")
                self._log_debug_details(value, indent + 2)
            elif isinstance(value, list):
                self.logger.info(f"{spaces}{key}: [{len(value)} items]")
                for i, item in enumerate(value[:3]):  # Show first 3 items
                    if isinstance(item, dict):
                        self.logger.info(f"{spaces}  [{i}]:")
                        self._log_debug_details(item, indent + 4)
                    else:
                        self.logger.info(f"{spaces}  [{i}]: {item}")
                if len(value) > 3:
                    self.logger.info(f"{spaces}  ... ({len(value)-3} more items)")
            else:
                self.logger.info(f"{spaces}{key}: {value}")
    
    def save_query_log(self, filename: str):
        """Save all logged queries to a JSON file."""
        log_data = [asdict(query) for query in self.query_log]
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.logger.info(f"Query log saved to: {filename}")
    
    def get_query_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged queries."""
        if not self.query_log:
            return {"message": "No queries logged"}
        
        total_queries = len(self.query_log)
        total_time = sum(q.execution_time_ms for q in self.query_log)
        avg_time = total_time / total_queries
        total_results = sum(q.num_found for q in self.query_log)
        
        handlers_used = {}
        parsers_used = {}
        
        for query in self.query_log:
            handler = query.handler
            handlers_used[handler] = handlers_used.get(handler, 0) + 1
            
            parser = query.parameters.get('defType', 'lucene')
            parsers_used[parser] = parsers_used.get(parser, 0) + 1
        
        return {
            "total_queries": total_queries,
            "total_execution_time_ms": total_time,
            "average_execution_time_ms": avg_time,
            "total_results_found": total_results,
            "average_results_per_query": total_results / total_queries,
            "handlers_used": handlers_used,
            "query_parsers_used": parsers_used
        }


class SolrConfigurationInspector:
    """Inspector for Solr configuration details."""
    
    def __init__(self, solr_url: str, collection: str):
        self.solr_url = solr_url.rstrip('/')
        self.collection = collection
        self.logger = logging.getLogger("SolrConfigInspector")
    
    def inspect_schema(self) -> Dict[str, Any]:
        """Inspect and return detailed schema information."""
        try:
            schema_url = f"{self.solr_url}/{self.collection}/schema"
            response = requests.get(schema_url)
            response.raise_for_status()
            
            schema_data = response.json()
            schema = schema_data.get('schema', {})
            
            # Extract key information
            fields = schema.get('fields', [])
            copy_fields = schema.get('copyFields', [])
            field_types = schema.get('fieldTypes', [])
            
            # Analyze field configuration
            field_analysis = self._analyze_fields(fields)
            copy_field_analysis = self._analyze_copy_fields(copy_fields)
            field_type_analysis = self._analyze_field_types(field_types)
            
            return {
                "schema_version": schema.get('version', 'unknown'),
                "unique_key": schema.get('uniqueKey', 'unknown'),
                "default_search_field": schema.get('defaultSearchField'),
                "field_count": len(fields),
                "copy_field_count": len(copy_fields),
                "field_type_count": len(field_types),
                "field_analysis": field_analysis,
                "copy_field_analysis": copy_field_analysis,
                "field_type_analysis": field_type_analysis,
                "raw_schema": schema
            }
            
        except Exception as e:
            self.logger.error(f"Failed to inspect schema: {e}")
            return {"error": str(e)}
    
    def inspect_config(self) -> Dict[str, Any]:
        """Inspect and return detailed configuration information."""
        try:
            config_url = f"{self.solr_url}/{self.collection}/config"
            response = requests.get(config_url)
            response.raise_for_status()
            
            config_data = response.json()
            config = config_data.get('config', {})
            
            # Extract request handlers
            request_handlers = config.get('requestHandler', {})
            search_handlers = {
                name: handler for name, handler in request_handlers.items()
                if name.startswith('/select') or name.startswith('/query') or 'SearchHandler' in str(handler)
            }
            
            # Analyze search handler configurations
            handler_analysis = self._analyze_search_handlers(search_handlers)
            
            return {
                "solr_version": config.get('luceneMatchVersion', 'unknown'),
                "data_dir": config.get('dataDir', 'unknown'),
                "search_handlers": list(search_handlers.keys()),
                "handler_analysis": handler_analysis,
                "raw_config": config
            }
            
        except Exception as e:
            self.logger.error(f"Failed to inspect config: {e}")
            return {"error": str(e)}
    
    def _analyze_fields(self, fields: List[Dict]) -> Dict[str, Any]:
        """Analyze field configuration."""
        indexed_fields = []
        stored_fields = []
        multivalue_fields = []
        text_fields = []
        
        for field in fields:
            name = field.get('name', '')
            field_type = field.get('type', '')
            
            if field.get('indexed', False):
                indexed_fields.append(name)
            if field.get('stored', False):
                stored_fields.append(name)
            if field.get('multiValued', False):
                multivalue_fields.append(name)
            if 'text' in field_type.lower():
                text_fields.append(name)
        
        return {
            "indexed_fields": indexed_fields,
            "stored_fields": stored_fields,
            "multivalue_fields": multivalue_fields,
            "text_fields": text_fields,
            "indexed_count": len(indexed_fields),
            "stored_count": len(stored_fields),
            "text_field_count": len(text_fields)
        }
    
    def _analyze_copy_fields(self, copy_fields: List[Dict]) -> Dict[str, Any]:
        """Analyze copy field configuration."""
        copy_targets = {}
        copy_sources = {}
        
        for copy_field in copy_fields:
            source = copy_field.get('source', '')
            dest = copy_field.get('dest', '')
            
            if dest not in copy_targets:
                copy_targets[dest] = []
            copy_targets[dest].append(source)
            
            if source not in copy_sources:
                copy_sources[source] = []
            copy_sources[source].append(dest)
        
        return {
            "copy_targets": copy_targets,
            "copy_sources": copy_sources,
            "target_count": len(copy_targets),
            "source_count": len(copy_sources)
        }
    
    def _analyze_field_types(self, field_types: List[Dict]) -> Dict[str, Any]:
        """Analyze field type configuration."""
        analyzers = {}
        tokenizers = {}
        filters = {}
        
        for field_type in field_types:
            name = field_type.get('name', '')
            analyzer_config = field_type.get('analyzer', {})
            
            if analyzer_config:
                analyzers[name] = analyzer_config
                
                # Extract tokenizer
                tokenizer = analyzer_config.get('tokenizer', {})
                if tokenizer:
                    tokenizer_class = tokenizer.get('class', 'unknown')
                    if tokenizer_class not in tokenizers:
                        tokenizers[tokenizer_class] = []
                    tokenizers[tokenizer_class].append(name)
                
                # Extract filters
                filter_list = analyzer_config.get('filters', [])
                for f in filter_list:
                    filter_class = f.get('class', 'unknown')
                    if filter_class not in filters:
                        filters[filter_class] = []
                    filters[filter_class].append(name)
        
        return {
            "analyzer_count": len(analyzers),
            "tokenizers_used": tokenizers,
            "filters_used": filters,
            "analyzers": analyzers
        }
    
    def _analyze_search_handlers(self, handlers: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze search handler configuration."""
        handler_details = {}
        
        for name, config in handlers.items():
            defaults = config.get('defaults', {})
            invariants = config.get('invariants', {})
            
            handler_details[name] = {
                "class": config.get('class', 'unknown'),
                "defaults": defaults,
                "invariants": invariants,
                "default_field": defaults.get('df'),
                "query_parser": defaults.get('defType', 'lucene'),
                "rows": defaults.get('rows', 'not set')
            }
        
        return handler_details


class SolrQueryInterceptor:
    """Intercepts and logs Solr queries for debugging purposes."""
    
    def __init__(self, solr_url: str, collection: str, debug_logger: SolrDebugLogger):
        self.solr_url = solr_url.rstrip('/')
        self.collection = collection
        self.debug_logger = debug_logger
        self.session = requests.Session()
    
    def execute_query_with_debug(self, query: str, handler: str = "/select", 
                                params: Dict[str, Any] = None) -> SolrQueryDebugInfo:
        """Execute a Solr query with full debugging information."""
        if params is None:
            params = {}
        
        # Add debug parameters
        debug_params = params.copy()
        debug_params.update({
            'q': query,
            'debug': 'true',
            'debug.explain.structured': 'true',
            'wt': 'json'
        })
        
        # Build URL
        url = f"{self.solr_url}/{self.collection}{handler}"
        
        # Execute query with timing
        start_time = time.time()
        
        try:
            response = self.session.get(url, params=debug_params, timeout=30)
            response.raise_for_status()
            
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Parse response
            result = response.json()
            
            # Extract debug information
            debug_info = result.get('debug', {})
            response_header = result.get('responseHeader', {})
            num_found = result.get('response', {}).get('numFound', 0)
            
            # Create debug info object
            query_debug_info = SolrQueryDebugInfo(
                query_string=query,
                solr_url=self.solr_url,
                collection=self.collection,
                handler=handler,
                parameters=debug_params,
                execution_time_ms=execution_time_ms,
                num_found=num_found,
                raw_response=result,
                debug_info=debug_info,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Log the query
            self.debug_logger.log_query_execution(query_debug_info)
            
            return query_debug_info
            
        except Exception as e:
            # Create error debug info
            error_debug_info = SolrQueryDebugInfo(
                query_string=query,
                solr_url=self.solr_url,
                collection=self.collection,
                handler=handler,
                parameters=debug_params,
                execution_time_ms=0.0,
                num_found=0,
                raw_response={"error": str(e)},
                debug_info={"error": str(e)},
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            self.debug_logger.log_query_execution(error_debug_info)
            raise


class DebugMode:
    """Main debug mode orchestrator."""
    
    def __init__(self, solr_url: str, collection: str, log_file: Optional[str] = None):
        self.solr_url = solr_url
        self.collection = collection
        
        # Initialize components
        self.debug_logger = SolrDebugLogger(log_file=log_file)
        self.config_inspector = SolrConfigurationInspector(solr_url, collection)
        self.query_interceptor = SolrQueryInterceptor(solr_url, collection, self.debug_logger)
        
        self.logger = logging.getLogger("DebugMode")
    
    def start_debug_session(self, queries: List[str], params_list: List[Dict[str, Any]] = None):
        """Start a debug session with specified queries."""
        self.logger.info("🐛 STARTING DEBUG SESSION")
        self.logger.info("="*60)
        
        # Inspect current configuration
        self.logger.info("📋 INSPECTING CURRENT CONFIGURATION")
        self.inspect_current_configuration()
        
        # Execute queries with debugging
        self.logger.info("🔍 EXECUTING QUERIES WITH DEBUG MODE")
        
        if params_list is None:
            params_list = [{}] * len(queries)
        
        debug_results = []
        for i, (query, params) in enumerate(zip(queries, params_list)):
            self.logger.info(f"\n--- Query {i+1}/{len(queries)} ---")
            debug_info = self.query_interceptor.execute_query_with_debug(query, params=params)
            debug_results.append(debug_info)
        
        # Show session summary
        self.show_debug_summary()
        
        return debug_results
    
    def inspect_current_configuration(self):
        """Inspect and log current Solr configuration."""
        self.logger.info("🔧 SCHEMA INSPECTION")
        schema_info = self.config_inspector.inspect_schema()
        
        if "error" not in schema_info:
            self.logger.info(f"Schema Version: {schema_info['schema_version']}")
            self.logger.info(f"Unique Key: {schema_info['unique_key']}")
            self.logger.info(f"Default Search Field: {schema_info['default_search_field']}")
            self.logger.info(f"Total Fields: {schema_info['field_count']}")
            self.logger.info(f"Copy Fields: {schema_info['copy_field_count']}")
            self.logger.info(f"Text Fields: {len(schema_info['field_analysis']['text_fields'])}")
            
            # Show copy field targets
            copy_targets = schema_info['copy_field_analysis']['copy_targets']
            for target, sources in copy_targets.items():
                self.logger.info(f"Copy to '{target}': {', '.join(sources)}")
        
        self.logger.info("\n⚙️ CONFIGURATION INSPECTION")
        config_info = self.config_inspector.inspect_config()
        
        if "error" not in config_info:
            self.logger.info(f"Solr Version: {config_info['solr_version']}")
            self.logger.info(f"Search Handlers: {', '.join(config_info['search_handlers'])}")
            
            # Show handler details
            for handler, details in config_info['handler_analysis'].items():
                self.logger.info(f"\nHandler '{handler}':")
                self.logger.info(f"  Query Parser: {details['query_parser']}")
                self.logger.info(f"  Default Field: {details['default_field']}")
                self.logger.info(f"  Default Rows: {details['rows']}")
    
    def show_debug_summary(self):
        """Show summary of debug session."""
        summary = self.debug_logger.get_query_summary()
        
        self.logger.info("\n📊 DEBUG SESSION SUMMARY")
        self.logger.info("="*40)
        self.logger.info(f"Total Queries: {summary['total_queries']}")
        self.logger.info(f"Total Execution Time: {summary['total_execution_time_ms']:.2f}ms")
        self.logger.info(f"Average Execution Time: {summary['average_execution_time_ms']:.2f}ms")
        self.logger.info(f"Total Results Found: {summary['total_results_found']}")
        self.logger.info(f"Average Results per Query: {summary['average_results_per_query']:.1f}")
        
        if summary['handlers_used']:
            self.logger.info("\nHandlers Used:")
            for handler, count in summary['handlers_used'].items():
                self.logger.info(f"  {handler}: {count} queries")
        
        if summary['query_parsers_used']:
            self.logger.info("\nQuery Parsers Used:")
            for parser, count in summary['query_parsers_used'].items():
                self.logger.info(f"  {parser}: {count} queries")
    
    def save_debug_session(self, filename: str = None):
        """Save debug session to file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"debug_session_{timestamp}.json"
        
        self.debug_logger.save_query_log(filename)
        self.logger.info(f"Debug session saved to: {filename}")


# Example usage functions
def debug_basic_search(solr_url: str = "http://localhost:8983/solr", 
                      collection: str = "ecommerce_products"):
    """Debug basic search functionality."""
    debug_mode = DebugMode(solr_url, collection)
    
    test_queries = [
        "laptop",
        "smartphone",
        "wireless headphones",
        "gaming mouse"
    ]
    
    debug_mode.start_debug_session(test_queries)
    debug_mode.save_debug_session()


def debug_different_parsers(solr_url: str = "http://localhost:8983/solr",
                           collection: str = "ecommerce_products"):
    """Debug different query parser configurations."""
    debug_mode = DebugMode(solr_url, collection)
    
    query = "wireless bluetooth headphones"
    
    parser_configs = [
        {"defType": "lucene", "df": "_text_"},
        {"defType": "dismax", "qf": "product_title^2.0 product_description^1.0"},
        {"defType": "edismax", "qf": "product_title^3.0 product_description^1.0 product_bullet_point^1.5"}
    ]
    
    queries = [query] * len(parser_configs)
    
    debug_mode.start_debug_session(queries, parser_configs)
    debug_mode.save_debug_session("parser_comparison_debug.json")


if __name__ == "__main__":
    # Run basic debug session
    debug_basic_search()
