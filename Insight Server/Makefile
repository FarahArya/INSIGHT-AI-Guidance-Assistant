# Makefile for Vision AI Server

.PHONY: build run stop clean logs test health

# Docker image name
IMAGE_NAME = vision-ai-server
CONTAINER_NAME = vision-ai-server

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):latest .

# Run the container
run:
	@echo "Starting Vision AI Server..."
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p 5000:5000 \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/config.py:/app/config.py \
		--restart unless-stopped \
		$(IMAGE_NAME):latest

# Run with docker-compose
compose-up:
	@echo "Starting services with docker-compose..."
	docker-compose up -d

# Run production with nginx
production:
	@echo "Starting production services..."
	docker-compose --profile production up -d

# Stop the container
stop:
	@echo "Stopping Vision AI Server..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

# Stop docker-compose services
compose-down:
	@echo "Stopping docker-compose services..."
	docker-compose down

# Clean up Docker resources
clean: stop
	@echo "Cleaning up Docker resources..."
	-docker rmi $(IMAGE_NAME):latest
	-docker system prune -f

# View logs
logs:
	docker logs -f $(CONTAINER_NAME)

# View compose logs
compose-logs:
	docker-compose logs -f

# Test the API
test:
	@echo "Testing API endpoints..."
	curl -f http://localhost:5000/health || echo "Health check failed"
	curl -f http://localhost:5000/model_info || echo "Model info failed"

# Health check
health:
	@echo "Checking server health..."
	curl -s http://localhost:5000/health | python -m json.tool

# Restart the container
restart: stop run

# Rebuild and run
rebuild: clean build run

# Show container status
status:
	@echo "Container status:"
	docker ps -a --filter name=$(CONTAINER_NAME)

# Show docker-compose status
compose-status:
	docker-compose ps

# Enter container shell
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Show help
help:
	@echo "Available targets:"
	@echo "  build          - Build Docker image"
	@echo "  run            - Run container"
	@echo "  compose-up     - Start with docker-compose"
	@echo "  production     - Start production setup with nginx"
	@echo "  stop           - Stop container"
	@echo "  compose-down   - Stop docker-compose services"
	@echo "  clean          - Remove container and image"
	@echo "  logs           - Show container logs"
	@echo "  compose-logs   - Show docker-compose logs"
	@echo "  test           - Test API endpoints"
	@echo "  health         - Check server health"
	@echo "  restart        - Restart container"
	@echo "  rebuild        - Rebuild and run"
	@echo "  status         - Show container status"
	@echo "  compose-status - Show docker-compose status"
	@echo "  shell          - Enter container shell"
	@echo "  help           - Show this help"