#!/bin/bash

# Solr Optimizer Demo - Docker Teardown Script
# This script cleanly shuts down the SolrCloud environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🛑 Shutting down Solr Optimizer Demo Environment"
echo "================================================"

# Check if Docker Compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    if ! docker compose version > /dev/null 2>&1; then
        echo "❌ Docker Compose is not available."
        exit 1
    fi
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Change to docker setup directory
cd "$SCRIPT_DIR"

# Stop and remove containers, networks, and volumes
echo "🧹 Stopping containers and cleaning up..."
$DOCKER_COMPOSE down --volumes --remove-orphans

# Remove any dangling volumes
echo "🗑️ Removing dangling volumes..."
docker volume prune -f

# Show cleanup summary
echo ""
echo "✅ Demo environment shutdown complete!"
echo "====================================="
echo ""
echo "All containers, networks, and volumes have been removed."
echo "Your system is now clean."
echo ""
echo "To restart the demo environment, run: ./setup.sh"
echo ""
