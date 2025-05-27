# Solr Optimizer Demo Troubleshooting Guide

This guide helps diagnose and fix common configuration issues with the Solr Optimizer demo.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Connection Issues](#connection-issues)
3. [Schema Configuration Problems](#schema-configuration-problems)
4. [Search Handler Issues](#search-handler-issues)
5. [Data Loading Problems](#data-loading-problems)
6. [Query Execution Issues](#query-execution-issues)
7. [Performance Problems](#performance-problems)
8. [Advanced Debugging](#advanced-debugging)

## Quick Diagnostics

### Run Automated Diagnostics

```bash
# Run complete validation
python demo/schema_validation.py

# Run preflight checks
python demo/preflight_validation.py

# Test schema configuration
python demo/test_schema_config.py
```

### Check Solr Status

```bash
# Check if Solr is running
curl http://localhost:8983/solr/admin/ping

# Check collection status
curl http://localhost:8983/solr/ecommerce_products/admin/ping

# Check collection data count
curl "http://localhost:8983/solr/ecommerce_products/select?q=*:*&rows=0"
```

### Enable Debug Mode

```python
# Run queries with debug mode
from demo.debug_mode import DebugMode

debug_mode = DebugMode("http://localhost:8983/solr", "ecommerce_products")
debug_mode.start_debug_session(["laptop", "smartphone"])
debug_mode.save_debug_session()
```

## Connection Issues

### Problem: "Connection refused" or "Cannot connect to Solr"

**Symptoms:**
- Error messages like `ConnectionError: HTTPConnectionPool(host='localhost', port=8983)`
- Demo fails immediately with connection errors

**Diagnosis:**
```bash
# Check if Solr is running
ps aux | grep solr

# Check if port 8983 is open
netstat -an | grep 8983
lsof -i :8983
```

**Solutions:**

1. **Start Solr if not running:**
   ```bash
   cd demo/docker-setup
   ./setup.sh
   ```

2. **Check Docker containers:**
   ```bash
   docker ps
   docker logs solr-demo
   ```

3. **Verify Solr URL:**
   - Default: `http://localhost:8983/solr`
   - Docker: May use different IP, check with `docker inspect solr-demo`

4. **Firewall issues:**
   ```bash
   # Allow port 8983
   sudo ufw allow 8983
   ```

### Problem: "Collection not found"

**Symptoms:**
- HTTP 404 errors when accessing collection
- "Collection 'ecommerce_products' does not exist"

**Solutions:**

1. **Check existing collections:**
   ```bash
   curl "http://localhost:8983/solr/admin/collections?action=LIST"
   ```

2. **Recreate collection:**
   ```bash
   cd demo/docker-setup
   ./teardown.sh
   ./setup.sh
   ```

3. **Manual collection creation:**
   ```bash
   curl "http://localhost:8983/solr/admin/collections?action=CREATE&name=ecommerce_products&numShards=1&replicationFactor=1&collection.configName=ecommerce"
   ```

## Schema Configuration Problems

### Problem: Missing required fields

**Symptoms:**
- Schema validation fails with "Missing required fields"
- Data loading errors about unknown fields

**Diagnosis:**
```bash
python demo/schema_validation.py
```

**Solutions:**

1. **Check current schema:**
   ```bash
   curl "http://localhost:8983/solr/ecommerce_products/schema/fields"
   ```

2. **Add missing fields manually:**
   ```bash
   # Example: Add product_title field
   curl -X POST "http://localhost:8983/solr/ecommerce_products/schema" \
     -H 'Content-type:application/json' \
     -d '{
       "add-field": {
         "name": "product_title",
         "type": "text_general",
         "stored": true,
         "indexed": true
       }
     }'
   ```

3. **Reload schema configuration:**
   ```bash
   cd demo/docker-setup
   ./teardown.sh
   ./setup.sh
   ```

### Problem: Copy fields not working

**Symptoms:**
- Search returns no results
- `_text_` field appears empty
- Validation reports missing copy fields

**Diagnosis:**
```bash
# Check copy field configuration
curl "http://localhost:8983/solr/ecommerce_products/schema/copyfields"

# Test _text_ field content
curl "http://localhost:8983/solr/ecommerce_products/select?q=*:*&fl=_text_&rows=1"
```

**Solutions:**

1. **Add missing copy fields:**
   ```bash
   curl -X POST "http://localhost:8983/solr/ecommerce_products/schema" \
     -H 'Content-type:application/json' \
     -d '{
       "add-copy-field": {
         "source": "product_title",
         "dest": "_text_"
       }
     }'
   ```

2. **Verify copy field configuration in schema.xml:**
   ```xml
   <copyField source="product_title" dest="_text_"/>
   <copyField source="product_description" dest="_text_"/>
   <copyField source="product_bullet_point" dest="_text_"/>
   ```

3. **Reindex data after schema changes:**
   ```bash
   python demo/scripts/load_data.py
   ```

### Problem: Text analysis not working

**Symptoms:**
- Searches don't match variations (uppercase/lowercase)
- Stemming or synonyms not working
- Poor search quality

**Diagnosis:**
```bash
# Test analysis
curl "http://localhost:8983/solr/ecommerce_products/analysis/field?analysis.fieldname=_text_&analysis.fieldvalue=Running%20Shoes&wt=json"
```

**Solutions:**

1. **Check field type definition:**
   ```bash
   curl "http://localhost:8983/solr/ecommerce_products/schema/fieldtypes/text_general"
   ```

2. **Verify analyzer configuration in schema.xml:**
   ```xml
   <fieldType name="text_general" class="solr.TextField">
     <analyzer>
       <tokenizer class="solr.StandardTokenizerFactory"/>
       <filter class="solr.StopFilterFactory" ignoreCase="true"/>
       <filter class="solr.LowerCaseFilterFactory"/>
     </analyzer>
   </fieldType>
   ```

3. **Test analysis chain:**
   ```python
   from demo.debug_mode import debug_basic_search
   debug_basic_search()
   ```

## Search Handler Issues

### Problem: Default search field not configured

**Symptoms:**
- Empty search results for simple queries
- Need to specify field explicitly (e.g., `product_title:laptop`)

**Diagnosis:**
```bash
# Check handler configuration
curl "http://localhost:8983/solr/ecommerce_products/config/requestHandler"
```

**Solutions:**

1. **Update search handler defaults:**
   ```bash
   curl -X POST "http://localhost:8983/solr/ecommerce_products/config" \
     -H 'Content-type:application/json' \
     -d '{
       "update-requesthandler": {
         "/select": {
           "class": "solr.SearchHandler",
           "defaults": {
             "echoParams": "explicit",
             "rows": 10,
             "df": "_text_"
           }
         }
       }
     }'
   ```

2. **Verify in solrconfig.xml:**
   ```xml
   <requestHandler name="/select" class="solr.SearchHandler">
     <lst name="defaults">
       <str name="df">_text_</str>
     </lst>
   </requestHandler>
   ```

### Problem: DisMax/eDisMax not working

**Symptoms:**
- Multi-field searches return poor results
- Field boosting not working
- Phrase matching not working

**Diagnosis:**
```python
from demo.debug_mode import debug_different_parsers
debug_different_parsers()
```

**Solutions:**

1. **Check query parser configuration:**
   ```bash
   # Test DisMax
   curl "http://localhost:8983/solr/ecommerce_products/select?q=laptop&defType=dismax&qf=product_title^2.0%20product_description^1.0"
   
   # Test eDisMax
   curl "http://localhost:8983/solr/ecommerce_products/select?q=laptop&defType=edismax&qf=product_title^3.0%20product_description^1.0"
   ```

2. **Add DisMax configuration to solrconfig.xml:**
   ```xml
   <requestHandler name="/dismax" class="solr.SearchHandler">
     <lst name="defaults">
       <str name="defType">dismax</str>
       <str name="qf">product_title^2.0 product_description^1.0</str>
       <str name="pf">product_title^3.0</str>
       <str name="mm">2&lt;-1 5&lt;80%</str>
     </lst>
   </requestHandler>
   ```

## Data Loading Problems

### Problem: No data in collection

**Symptoms:**
- `numFound: 0` for all queries
- Collection exists but is empty

**Diagnosis:**
```bash
# Check document count
curl "http://localhost:8983/solr/ecommerce_products/select?q=*:*&rows=0"

# Check data files
ls -la demo/data/processed/
```

**Solutions:**

1. **Download and load data:**
   ```bash
   python demo/scripts/download_data.py
   python demo/scripts/load_data.py
   ```

2. **Check data loading errors:**
   ```bash
   python demo/scripts/load_data.py --verbose
   ```

3. **Manual data loading:**
   ```bash
   # Load sample document
   curl -X POST "http://localhost:8983/solr/ecommerce_products/update?commit=true" \
     -H 'Content-Type: application/json' \
     -d '[{
       "product_id": "test123",
       "product_title": "Test Laptop",
       "product_description": "A test laptop for debugging"
     }]'
   ```

### Problem: Data loading fails with field errors

**Symptoms:**
- HTTP 400 errors during data loading
- "Unknown field" errors
- Schema mismatch errors

**Solutions:**

1. **Validate schema first:**
   ```bash
   python demo/schema_validation.py
   ```

2. **Check data format:**
   ```python
   import pandas as pd
   df = pd.read_csv('demo/data/processed/products.csv')
   print(df.columns.tolist())
   print(df.head())
   ```

3. **Update field mapping in load_data.py:**
   ```python
   # Check field mapping in scripts/load_data.py
   field_mapping = {
       'title': 'product_title',
       'description': 'product_description',
       # ... other mappings
   }
   ```

## Query Execution Issues

### Problem: Optimization returns no improvements

**Symptoms:**
- All optimization attempts show same NDCG scores
- No variation in results across different strategies

**Diagnosis:**
```python
# Run with debug mode
from demo.debug_mode import DebugMode
debug_mode = DebugMode("http://localhost:8983/solr", "ecommerce_products")

# Test different configurations
test_queries = ["laptop", "smartphone"]
configs = [
    {"defType": "lucene", "df": "_text_"},
    {"defType": "dismax", "qf": "product_title^2.0 product_description^1.0"},
    {"defType": "edismax", "qf": "product_title^3.0 product_description^1.0"}
]

for i, config in enumerate(configs):
    print(f"\n--- Testing config {i+1} ---")
    debug_mode.query_interceptor.execute_query_with_debug(
        "laptop", params=config
    )
```

**Solutions:**

1. **Check relevance judgments:**
   ```bash
   head -20 demo/data/judgments/judgments.csv
   wc -l demo/data/judgments/judgments.csv
   ```

2. **Verify query-judgment mapping:**
   ```python
   import pandas as pd
   queries_df = pd.read_csv('demo/data/processed/queries.csv')
   judgments_df = pd.read_csv('demo/data/judgments/judgments.csv')
   print("Query count:", len(queries_df))
   print("Judgment count:", len(judgments_df))
   print("Unique queries with judgments:", judgments_df['query_id'].nunique())
   ```

3. **Test individual queries:**
   ```bash
   # Test specific query with debug
   curl "http://localhost:8983/solr/ecommerce_products/select?q=laptop&debug=true&debugQuery=true"
   ```

### Problem: NDCG calculation errors

**Symptoms:**
- NDCG scores always 0.0
- Metrics calculation fails
- Division by zero errors

**Solutions:**

1. **Check relevance judgment format:**
   ```python
   import pandas as pd
   df = pd.read_csv('demo/data/judgments/judgments.csv')
   print("Judgment values:", df['judgment'].unique())
   print("Data types:", df.dtypes)
   ```

2. **Validate judgment values:**
   ```python
   # Judgments should be integers (0, 1, 2, 3, 4)
   # Check for missing or invalid values
   df = pd.read_csv('demo/data/judgments/judgments.csv')
   print("Invalid judgments:", df[~df['judgment'].isin([0,1,2,3,4])])
   ```

3. **Test metrics calculation:**
   ```python
   from solr_optimizer.agents.metrics.standard_metrics_agent import StandardMetricsAgent
   
   agent = StandardMetricsAgent()
   # Test with sample data
   test_results = [
       {'product_id': 'P1', 'score': 1.0},
       {'product_id': 'P2', 'score': 0.8}
   ]
   test_judgments = {'P1': 3, 'P2': 1}
   
   ndcg = agent.calculate_ndcg(test_results, test_judgments, k=10)
   print(f"Test NDCG: {ndcg}")
   ```

## Performance Problems

### Problem: Slow query execution

**Symptoms:**
- Individual queries take > 1 second
- Demo takes very long to complete
- Timeout errors

**Solutions:**

1. **Check query complexity:**
   ```python
   from demo.debug_mode import DebugMode
   debug_mode = DebugMode("http://localhost:8983/solr", "ecommerce_products", 
                         log_file="debug_performance.log")
   debug_mode.start_debug_session(["laptop"])
   summary = debug_mode.debug_logger.get_query_summary()
   print(f"Average execution time: {summary['average_execution_time_ms']:.2f}ms")
   ```

2. **Optimize Solr configuration:**
   ```xml
   <!-- Add to solrconfig.xml -->
   <requestHandler name="/select" class="solr.SearchHandler">
     <lst name="defaults">
       <int name="rows">10</int>
       <str name="df">_text_</str>
       <!-- Disable debug by default -->
       <str name="debug">false</str>
     </lst>
   </requestHandler>
   ```

3. **Reduce demo scope:**
   ```python
   # In run_enhanced_demo.py, use fewer queries
   demo_queries = queries[:10]  # Instead of [:20]
   ```

### Problem: High memory usage

**Symptoms:**
- Solr container using excessive memory
- OutOfMemory errors in Solr logs

**Solutions:**

1. **Adjust Java heap size:**
   ```bash
   # In docker-compose.yml
   services:
     solr:
       environment:
         - SOLR_JAVA_MEM=-Xms512m -Xmx1g
   ```

2. **Monitor memory usage:**
   ```bash
   docker stats solr-demo
   ```

3. **Check Solr admin:**
   - Visit http://localhost:8983/solr/#/~java-properties
   - Check memory usage and GC statistics

## Advanced Debugging

### Enable Comprehensive Logging

```python
# Create comprehensive debug session
import logging
logging.basicConfig(level=logging.DEBUG)

from demo.debug_mode import DebugMode
from demo.schema_validation import SolrSchemaValidator

# Full debugging session
debug_mode = DebugMode("http://localhost:8983/solr", "ecommerce_products", 
                      log_file="comprehensive_debug.log")

# Inspect configuration
debug_mode.inspect_current_configuration()

# Test queries with different parsers
test_queries = ["laptop computer", "wireless headphones", "gaming mouse"]
parser_configs = [
    {"defType": "lucene", "df": "_text_"},
    {"defType": "dismax", "qf": "product_title^2.0 product_description^1.0"},
    {"defType": "edismax", "qf": "product_title^3.0 product_description^1.0 product_bullet_point^1.5",
     "pf": "product_title^4.0", "mm": "2<-1 5<80%"}
]

for query in test_queries:
    for config in parser_configs:
        try:
            debug_info = debug_mode.query_interceptor.execute_query_with_debug(
                query, params=config
            )
            print(f"Query: {query}, Parser: {config.get('defType', 'lucene')}, "
                  f"Results: {debug_info.num_found}, Time: {debug_info.execution_time_ms:.2f}ms")
        except Exception as e:
            print(f"Error with query '{query}' and config {config}: {e}")

# Save debug session
debug_mode.save_debug_session("advanced_debug_session.json")

# Run schema validation
validator = SolrSchemaValidator("http://localhost:8983/solr", "ecommerce_products")
passed, results = validator.validate_all()
validator.print_validation_report()
```

### Analyze Query Performance

```python
# Performance analysis script
import json
import statistics

# Load debug session
with open("advanced_debug_session.json", "r") as f:
    debug_data = json.load(f)

# Analyze execution times
execution_times = [float(entry["execution_time_ms"]) for entry in debug_data]
result_counts = [int(entry["num_found"]) for entry in debug_data]

print("PERFORMANCE ANALYSIS")
print("=" * 40)
print(f"Total queries: {len(execution_times)}")
print(f"Average execution time: {statistics.mean(execution_times):.2f}ms")
print(f"Median execution time: {statistics.median(execution_times):.2f}ms")
print(f"Max execution time: {max(execution_times):.2f}ms")
print(f"Min execution time: {min(execution_times):.2f}ms")
print(f"Average results: {statistics.mean(result_counts):.1f}")
print(f"Queries with no results: {sum(1 for count in result_counts if count == 0)}")

# Find slow queries
slow_threshold = statistics.mean(execution_times) + statistics.stdev(execution_times)
slow_queries = [entry for entry in debug_data 
                if float(entry["execution_time_ms"]) > slow_threshold]

if slow_queries:
    print(f"\nSLOW QUERIES (>{slow_threshold:.2f}ms):")
    for query in slow_queries:
        print(f"  {query['query_string']}: {query['execution_time_ms']:.2f}ms, "
              f"{query['num_found']} results")
```

### Export Configuration for Support

```bash
# Create support package
mkdir -p debug_export

# Export schema
curl "http://localhost:8983/solr/ecommerce_products/schema" > debug_export/schema.json

# Export config
curl "http://localhost:8983/solr/ecommerce_products/config" > debug_export/config.json

# Export collection info
curl "http://localhost:8983/solr/admin/collections?action=CLUSTERSTATUS&collection=ecommerce_products" > debug_export/collection_status.json

# Copy debug logs
cp comprehensive_debug.log debug_export/
cp advanced_debug_session.json debug_export/

# Run all validations
python demo/schema_validation.py > debug_export/schema_validation.txt 2>&1
python demo/preflight_validation.py > debug_export/preflight_validation.txt 2>&1

# Package for support
tar -czf solr_optimizer_debug_$(date +%Y%m%d_%H%M%S).tar.gz debug_export/

echo "Debug package created: solr_optimizer_debug_*.tar.gz"
```

## Common Error Messages and Solutions

### "No default field configured"
```bash
# Solution: Set default field in search handler
curl -X POST "http://localhost:8983/solr/ecommerce_products/config" \
  -H 'Content-type:application/json' \
  -d '{"update-requesthandler":{"/select":{"defaults":{"df":"_text_"}}}}'
```

### "Unknown field '_text_'"
```bash
# Solution: Check if _text_ field exists
curl "http://localhost:8983/solr/ecommerce_products/schema/fields/_text_"
# If not found, add it
curl -X POST "http://localhost:8983/solr/ecommerce_products/schema" \
  -H 'Content-type:application/json' \
  -d '{"add-field":{"name":"_text_","type":"text_general","multiValued":true,"stored":false}}'
```

### "Cannot parse query"
```bash
# Solution: Check query syntax
curl "http://localhost:8983/solr/ecommerce_products/select?q=laptop&debug=true&debugQuery=true"
```

### "Field is not indexed"
```bash
# Solution: Add indexed=true to field definition
curl -X POST "http://localhost:8983/solr/ecommerce_products/schema" \
  -H 'Content-type:application/json' \
  -d '{"replace-field":{"name":"FIELD_NAME","type":"text_general","indexed":true,"stored":true}}'
```

## Getting Help

1. **Run full diagnostics:**
   ```bash
   python demo/schema_validation.py
   python demo/preflight_validation.py
   ```

2. **Enable debug mode:**
   ```python
   from demo.debug_mode import debug_basic_search
   debug_basic_search()
   ```

3. **Check Solr admin UI:**
   - Visit: http://localhost:8983/solr/
   - Check Core Admin, Schema Browser, Query interface

4. **Review logs:**
   ```bash
   docker logs solr-demo
   tail -f comprehensive_debug.log
   ```

5. **Create support package:**
   ```bash
   # Run the export script above to package all debug information
   ```

For additional help, consult the [Solr Documentation](https://solr.apache.org/guide/) or create an issue with your debug package.
