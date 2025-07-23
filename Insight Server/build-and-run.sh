#!/bin/bash

# Build and run script for Vision AI Server

set -e

echo "🚀 Building Vision AI Server Docker Image..."

# Build the Docker image
docker build -t vision-ai-server:latest .

echo "✅ Docker image built successfully!"

# Check if we want to run with docker-compose
if [ "$1" = "compose" ]; then
    echo "🐳 Starting services with docker-compose..."
    docker-compose up -d
    echo "✅ Services started! Check status with: docker-compose ps"
    echo "📊 View logs with: docker-compose logs -f"
    echo "🔗 API available at: http://localhost:5000"
elif [ "$1" = "production" ]; then
    echo "🐳 Starting production services with nginx..."
    docker-compose --profile production up -d
    echo "✅ Production services started!"
    echo "🔗 API available at: http://localhost (port 80)"
else
    echo "🐳 Running Docker container..."
    docker run -d \
        --name vision-ai-server \
        -p 5000:5000 \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/config.py:/app/config.py \
        --restart unless-stopped \
        vision-ai-server:latest
    
    echo "✅ Container started successfully!"
    echo "📊 View logs with: docker logs -f vision-ai-server"
    echo "🔗 API available at: http://localhost:5000"
fi

echo ""
echo "📋 Useful commands:"
echo "  Health check: curl http://localhost:5000/health"
echo "  Stop container: docker stop vision-ai-server"
echo "  Remove container: docker rm vision-ai-server"
echo "  Stop compose: docker-compose down"
echo ""