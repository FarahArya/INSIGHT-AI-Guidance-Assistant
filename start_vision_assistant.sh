#!/bin/bash

# Change to the application directory
cd /home/rpi-farah/INSIGHT-AI-Guidance-Assistant

# Activate virtual environment (adjust path if different)
source venv/bin/activate

# Wait for camera to be ready
sleep 5

# Run the vision assistant
python3 vision_assistant.py

# If the script exits, log it
echo "Vision Assistant stopped at $(date)" >> /home/rpi-farah/vision_assistant.log