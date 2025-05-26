# Solr Optimizer Demo

This directory contains a complete end-to-end demonstration of the Solr Optimizer framework, featuring AI-powered query optimization with real e-commerce data and relevance judgments.

## Demo Overview

The demo showcases:
- **Complete SolrCloud setup** with Docker (3-node cluster + Zookeeper ensemble)
- **Real e-commerce product data** (10,000+ products across 5 categories)
- **Realistic search queries** with graded relevance judgments
- **Progressive optimization** from basic to advanced configurations
- **Detailed performance analysis** with metrics and explanations
- **AI-powered optimization** recommendations (when configured)

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- 4GB+ available RAM
- 10GB+ available disk space

### 1. Setup Environment

```bash
# Clone and setup project
git clone <repository-url>
cd solr-optimizer

# Install dependencies
pip install -e .
```

### 2. Start SolrCloud

```bash
# Start the complete SolrCloud environment
cd demo/docker-setup
./setup.sh
```

This will:
- Start 3 Solr nodes and 3 Zookeeper nodes
- Create the `ecommerce_products` collection
- Configure proper schema and request handlers
- Take ~2-3 minutes to complete

### 3. Generate and Load Data

```bash
# Generate sample e-commerce dataset
python demo/scripts/download_data.py

# Load data into Solr
python demo/scripts/load_data.py
```

This creates:
- 10,000 realistic product records
- 75 test queries with relevance judgments
- Multiple data formats (JSON, CSV, TREC)

### 4. Run the Demo

```bash
# Run the complete optimization demo
python demo/run_complete_demo.py
```

## Demo Workflow

The demo runs through three optimization iterations:

### 1. Baseline (Basic Lucene Search)
- Uses default Lucene query parser
- Simple text matching
- Establishes performance baseline

### 2. Basic Optimization (DisMax)
- Switches to DisMax query parser
- Adds field boosting (title^2.0, description^1.0)
- Implements phrase boosting
- Configures minimum should match

### 3. Advanced Optimization (eDisMax)
- Uses extended DisMax parser
- Advanced field boosting strategy
- Multi-level phrase boosting (pf, pf2, pf3)
- Function queries and conditional boosts

## Expected Results

Typical performance improvements:
- **Baseline**: NDCG@10 ≈ 0.30-0.40
- **Basic Optimization**: NDCG@10 ≈ 0.45-0.55 (+15-25%)
- **Advanced Optimization**: NDCG@10 ≈ 0.55-0.65 (+10-15% more)

The demo provides detailed analysis including:
- Overall metric changes
- Per-query improvement/degradation analysis  
- Ranking change explanations
- Configuration comparisons

## Validation and Testing

### Run End-to-End Tests

```bash
# Validate complete demo readiness
python -m pytest tests/demo/test_end_to_end_demo.py -v
```

This comprehensive test suite validates:
- ✅ Solr connectivity and cluster health
- ✅ Collection existence and configuration
- ✅ Data loading and availability  
- ✅ Query handler functionality (select, dismax, edismax)
- ✅ Framework component integration
- ✅ Sample query execution

### Manual Validation

```bash
# Test Solr directly
curl "http://localhost:8983/solr/ecommerce_products/select?q=laptop&rows=5"

# Test DisMax handler
curl "http://localhost:8983/solr/ecommerce_products/dismax?q=laptop&rows=5"

# Check collection status
curl "http://localhost:8983/solr/admin/collections?action=LIST"
```

## Directory Structure

```
demo/
├── README.md                          # This file
├── docker-setup/                      # SolrCloud Docker environment
│   ├── docker-compose.yml            # Docker services definition
│   ├── setup.sh                      # Environment setup script
│   ├── teardown.sh                   # Cleanup script
│   └── solr-init/                    # Solr configuration
│       └── configsets/ecommerce/      # Collection config
├── scripts/                          # Data processing scripts
│   ├── download_data.py              # Generate sample dataset
│   └── load_data.py                  # Load data into Solr
├── data/                             # Generated demo data
│   ├── processed/                    # Products and queries
│   └── judgments/                    # Relevance judgments
└── run_complete_demo.py              # Main demo orchestrator
```

## Advanced Usage

### AI-Powered Optimization

To enable AI-powered optimization, configure your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"

# Run with AI optimization
python -m solr_optimizer.cli.main ai-optimize \
    --experiment-id your-experiment \
    --ai-model openai:gpt-4
```

### Custom Data

To use your own data instead of generated samples:

1. **Replace products data**: Place your products in `demo/data/processed/products.json`
2. **Replace queries**: Place your queries in `demo/data/processed/queries.csv`  
3. **Replace judgments**: Place relevance judgments in `demo/data/judgments/judgments.csv`

Format requirements:
- Products: JSON array with fields matching the Solr schema
- Queries: CSV with columns `query_id`, `query`, `category`
- Judgments: CSV with columns `query_id`, `product_id`, `judgment` (0-2 scale)

### Performance Tuning

For larger datasets or production testing:

```bash
# Increase Solr heap size
export SOLR_HEAP=2g

# Adjust batch size for data loading
python demo/scripts/load_data.py --batch-size 500

# Use more aggressive optimization
python demo/run_complete_demo.py --optimization-level aggressive
```

## Troubleshooting

### Common Issues

**Docker Issues:**
```bash
# Check Docker status
docker info

# Restart Docker services
cd demo/docker-setup
./teardown.sh
./setup.sh
```

**Memory Issues:**
```bash
# Check available memory
docker stats

# Reduce memory usage
docker-compose down
docker system prune -f
```

**Port Conflicts:**
```bash
# Check port usage
lsof -i :8983
lsof -i :2181

# Use alternative ports (modify docker-compose.yml)
```

**Data Loading Issues:**
```bash
# Verify Solr is accessible
curl http://localhost:8983/solr/admin/info/system

# Check collection status
curl "http://localhost:8983/solr/admin/collections?action=LIST"

# Reload data
python demo/scripts/load_data.py --clear-first
```

### Logs and Debugging

```bash
# View Solr logs
docker-compose logs solr1

# View all container logs  
docker-compose logs

# Check demo test status
python -m pytest tests/demo/ -v --tb=short
```

## Demo Customization

### Adding New Optimization Strategies

1. **Create new query configuration** in `demo/run_complete_demo.py`
2. **Add new iteration** to the demo workflow
3. **Update comparison logic** to analyze results

### Custom Metrics

1. **Extend metrics agent** to support new metrics
2. **Update demo script** to use new metrics
3. **Modify comparison output** to display results

### Integration with External Tools

The demo framework supports integration with:
- **Grafana/Kibana** for visualization
- **MLflow/AimStack** for experiment tracking  
- **Jupyter notebooks** for analysis
- **Custom dashboards** via REST API

## Production Considerations

This demo is designed for **demonstration and development**. For production:

1. **Security**: Add authentication, SSL/TLS, network security
2. **Scalability**: Tune JVM settings, increase cluster size
3. **Monitoring**: Add comprehensive logging, metrics collection
4. **Backup**: Implement data backup and recovery procedures
5. **Performance**: Optimize for your specific hardware and data

## Next Steps

After running the demo:

1. **Explore the framework**: Try different optimization strategies
2. **Use your own data**: Replace sample data with real search requirements
3. **Integrate AI agents**: Configure OpenAI/Claude for intelligent optimization
4. **Build custom agents**: Extend the framework for your specific needs
5. **Deploy to production**: Adapt the configuration for production environments

## Support

For issues, questions, or contributions:
- Review the main project [README.md](../README.md)
- Check the [ARCHITECTURE.md](../ARCHITECTURE.md) for technical details
- Run validation tests: `python -m pytest tests/demo/ -v`
- Examine log files for error details

## Demo Data Attribution

The sample e-commerce dataset is synthetically generated for demonstration purposes. It includes realistic product categories, brands, and search patterns typical of e-commerce applications, but all data is artificial and not based on any real product catalog or customer behavior.
