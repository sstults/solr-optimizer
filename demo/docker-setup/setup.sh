#!/bin/bash

# Solr Optimizer Demo - Docker Setup Script
# This script sets up a complete SolrCloud environment for demonstration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "🚀 Setting up Solr Optimizer Demo Environment"
echo "============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    if ! docker compose version > /dev/null 2>&1; then
        echo "❌ Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo "✅ Docker is running"
echo "✅ Docker Compose is available"

# Change to docker setup directory
cd "$SCRIPT_DIR"

# Stop any existing containers
echo "🧹 Cleaning up existing containers..."
$DOCKER_COMPOSE down --volumes --remove-orphans 2>/dev/null || true

# Pull latest images
echo "📥 Pulling latest Docker images..."
$DOCKER_COMPOSE pull

# Start the services
echo "🔄 Starting SolrCloud cluster..."
$DOCKER_COMPOSE up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."

# Function to check if a service is ready
check_service() {
    local service_name=$1
    local port=$2
    local timeout=${3:-60}
    local service_type=${4:-http}
    local count=0
    
    echo "   Checking $service_name on port $port..."
    while [ $count -lt $timeout ]; do
        if [ "$service_type" = "zookeeper" ]; then
            # Check Zookeeper using the 'ruok' command
            if echo "ruok" | nc -w 2 localhost $port 2>/dev/null | grep -q "imok"; then
                echo "   ✅ $service_name is ready"
                return 0
            fi
        else
            # Check HTTP services
            if curl -s "http://localhost:$port" > /dev/null 2>&1; then
                echo "   ✅ $service_name is ready"
                return 0
            fi
        fi
        sleep 2
        count=$((count + 2))
    done
    
    echo "   ❌ $service_name failed to start within $timeout seconds"
    return 1
}

# Check Zookeeper ensemble
check_service "Zookeeper-1" 2181 60 zookeeper
check_service "Zookeeper-2" 2182 60 zookeeper
check_service "Zookeeper-3" 2183 60 zookeeper

# Check Solr nodes
check_service "Solr-1" 8983 120
check_service "Solr-2" 8984 60
check_service "Solr-3" 8985 60

# Wait a bit more for collection creation
echo "⏳ Waiting for collection creation..."
sleep 10

# Create the ecommerce_products collection
echo "🔧 Creating ecommerce_products collection..."

# First upload the configset
echo "   📤 Uploading configset to Zookeeper..."
$DOCKER_COMPOSE exec -T solr1 solr zk upconfig -n ecommerce_products -d /docker-entrypoint-initdb.d/configsets/ecommerce -z zookeeper1:2181,zookeeper2:2181,zookeeper3:2181

# Wait a moment for the configset to be available
sleep 3

# Create the collection
echo "   🏗️  Creating collection with custom configset..."
if curl -s -X POST "http://localhost:8983/solr/admin/collections?action=CREATE&name=ecommerce_products&numShards=2&replicationFactor=2&collection.configName=ecommerce_products" | grep -q '"status":0'; then
    echo "   ✅ Collection 'ecommerce_products' created successfully with custom configset"
else
    echo "   ⚠️  Failed to create collection with custom configset, falling back to default..."
    # Fallback to default configset
    curl -s -X POST "http://localhost:8983/solr/admin/collections?action=CREATE&name=ecommerce_products&numShards=1&replicationFactor=1&collection.configName=_default" >/dev/null 2>&1
    echo "   ✅ Collection 'ecommerce_products' created with default configset"
fi

# Verify collection was created
echo "🔍 Verifying ecommerce_products collection..."
if curl -s "http://localhost:8983/solr/admin/collections?action=LIST" | grep -q "ecommerce_products"; then
    echo "   ✅ Collection is available and ready"
else
    echo "   ❌ Collection creation failed"
fi

# Show cluster status
echo ""
echo "🎉 Demo environment is ready!"
echo "================================"
echo ""
echo "SolrCloud URLs:"
echo "  - Node 1: http://localhost:8983/solr"
echo "  - Node 2: http://localhost:8984/solr"  
echo "  - Node 3: http://localhost:8985/solr"
echo ""
echo "Zookeeper URLs:"
echo "  - ZK 1: localhost:2181"
echo "  - ZK 2: localhost:2182"
echo "  - ZK 3: localhost:2183"
echo ""
echo "Collection: ecommerce_products"
echo ""
echo "Next steps:"
echo "  1. Run data loading script: python demo/scripts/download_data.py"
echo "  2. Load sample data: python demo/scripts/load_data.py"
echo "  3. Run demo: python demo/run_complete_demo.py"
echo ""
echo "To stop the environment, run: ./teardown.sh"
echo ""
