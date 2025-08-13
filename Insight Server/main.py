from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import logging
import os
import time
from typing import List, Tuple, Dict, Any
from config import AppConfig
from vision.models import DualModelManager
from vision.distance import DistanceEstimator
from vision.processor import FrameProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

class VisionAIServer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.setup_components()
        self.last_model_type = None
        
    def setup_components(self):
        """Initialize vision system components"""
        # Distance estimator
        distance_estimator = DistanceEstimator(
            self.config.vision.real_heights,
            self.config.vision.real_widths
        )
        
        # Model manager
        self.model_manager = DualModelManager(
            self.config.vision.main_model_path,
            self.config.vision.architectural_model_path,
            self.config.vision.model_switch_interval
        )
        
        # Frame processor
        self.processor = FrameProcessor(self.model_manager, distance_estimator)
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame and return detection results"""
        try:
            # Process frame through vision pipeline
            objects, model_switched = self.processor.process(frame)
            
            # Track model type if switched
            if model_switched:
                self.last_model_type = model_switched
            
            # Determine distance threshold based on model type
            threshold = (self.config.vision.arch_near_threshold 
                        if self.last_model_type == "architecture" 
                        else self.config.vision.near_threshold)
            
            # Filter nearby objects
            nearby_objects = [obj for obj in objects if obj[0] <= threshold]
            
            # Format response
            response = {
                "success": True,
                "timestamp": time.time(),
                "model_type": self.last_model_type,
                "total_objects": len(objects),
                "nearby_objects": len(nearby_objects),
                "objects": self._format_objects(nearby_objects),
                "announcement": self._create_announcement(nearby_objects)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _format_objects(self, objects: List[Tuple[float, str, any, str]]) -> List[Dict[str, Any]]:
        """Format objects for JSON response"""
        formatted_objects = []
        for distance, label, bbox, position in objects:
            formatted_objects.append({
                "distance": round(distance, 2),
                "label": label,
                "position": position,
                "bbox": bbox.tolist() if hasattr(bbox, 'tolist') else bbox
            })
        return formatted_objects
    
    def _create_announcement(self, objects: List[Tuple[float, str, any, str]]) -> str:
        """Create natural language announcement with detailed positions"""
        if not objects:
            return "No objects detected nearby."

        # Sort by distance and importance (closest first)
        objects.sort(key=lambda x: x[0])

        # Separate known and unknown objects
        known_objects = [obj for obj in objects if obj[1] != "unknown object"]
        unknown_objects = [obj for obj in objects if obj[1] == "unknown object"]

        # Prioritize doors and walls that are close
        priority_objects = [
            obj for obj in known_objects
            if obj[1] in ["door", "wall"] and obj[0] < 3.0
        ]

        # Build announcement prioritizing known objects
        announcement_objects = []

        if priority_objects:
            # Use priority objects first
            announcement_objects.extend(priority_objects[:2])
            remaining_slots = 3 - len(announcement_objects)
        else:
            # Use closest known objects
            announcement_objects.extend(known_objects[:2])
            remaining_slots = 3 - len(announcement_objects)

        # Add unknown objects if we have remaining slots and they're close
        if remaining_slots > 0 and unknown_objects:
            close_unknown = [obj for obj in unknown_objects if obj[0] < 5.0]
            announcement_objects.extend(close_unknown[:remaining_slots])

        if len(announcement_objects) == 1:
            dist, label, _, position = announcement_objects[0]
            if label == "unknown object":
                return f"There is an unknown object approximately {dist:.1f} meters {position}."
            else:
                return f"There is a {label} approximately {dist:.1f} meters {position}."

        # Create announcement with combined position information
        parts = []
        for i, (dist, label, _, position) in enumerate(announcement_objects):
            article = "an" if label == "unknown object" else "a"

            if i == 0:
                parts.append(f"There is {article} {label} at {dist:.1f} meters {position}")
            elif i == len(announcement_objects) - 1:
                parts.append(f"and {article} {label} at {dist:.1f} meters {position}")
            else:
                parts.append(f"{article} {label} at {dist:.1f} meters {position}")

        return ", ".join(parts) + "."

# Global server instance
vision_server = None

def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image")
            
        return frame
        
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": vision_server is not None
    })

# NEW: Root endpoint for Flutter multipart file upload
@app.route('/', methods=['POST'])
def process_uploaded_file():
    """Process uploaded image file (Flutter multipart format)"""
    try:
        logger.info("Received multipart file upload request")
        
        if vision_server is None:
            logger.error("Vision server not initialized")
            return jsonify({
                "detected": "Vision server not initialized",
                "success": False
            }), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({
                "detected": "No file uploaded",
                "success": False
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({
                "detected": "No file selected",
                "success": False
            }), 400
        
        logger.info(f"Processing file: {file.filename}")
        
        # Read and decode image
        image_data = file.read()
        if len(image_data) == 0:
            logger.error("Empty file received")
            return jsonify({
                "detected": "Empty file received",
                "success": False
            }), 400
        
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image")
            return jsonify({
                "detected": "Failed to decode image",
                "success": False
            }), 400
        
        logger.info(f"Image decoded successfully: {frame.shape}")
        
        # Process frame
        result = vision_server.process_frame(frame)
        
        # Format response for Flutter compatibility
        if result["success"]:
            detected_message = result.get("announcement", "Nothing detected")
            logger.info(f"Detection successful: {detected_message}")
            return jsonify({
                "detected": detected_message,
                "success": True,
                "total_objects": result.get("total_objects", 0),
                "nearby_objects": result.get("nearby_objects", 0),
                "model_type": result.get("model_type", "unknown")
            })
        else:
            error_message = result.get('error', 'Unknown processing error')
            logger.error(f"Processing failed: {error_message}")
            return jsonify({
                "detected": f"Processing error: {error_message}",
                "success": False
            }), 500
            
    except Exception as e:
        logger.error(f"Error in process_uploaded_file: {e}")
        return jsonify({
            "detected": f"Server error: {str(e)}",
            "success": False
        }), 500

@app.route('/process', methods=['POST'])
def process_image():
    """Process uploaded image and return object detection results (JSON base64 format)"""
    try:
        if vision_server is None:
            return jsonify({
                "success": False,
                "error": "Vision server not initialized"
            }), 500
        
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image data provided"
            }), 400
        
        # Decode image
        frame = decode_image(data['image'])
        
        # Process frame
        result = vision_server.process_frame(frame)
        
        if result["success"]:
            # Add 'detected' field for compatibility
            result["detected"] = result.get("announcement", "Nothing detected")
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in process_image: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@app.route('/process_url', methods=['POST'])
def process_image_url():
    """Process image from URL"""
    try:
        if vision_server is None:
            return jsonify({
                "success": False,
                "error": "Vision server not initialized"
            }), 500
        
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                "success": False,
                "error": "No URL provided"
            }), 400
        
        # Download image from URL
        import requests
        response = requests.get(data['url'])
        response.raise_for_status()
        
        # Convert to numpy array
        nparr = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                "success": False,
                "error": "Failed to decode image from URL"
            }), 400
        
        # Process frame
        result = vision_server.process_frame(frame)
        
        if result["success"]:
            # Add 'detected' field for compatibility
            result["detected"] = result.get("announcement", "Nothing detected")
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except requests.RequestException as e:
        return jsonify({
            "success": False,
            "error": f"Failed to download image: {e}"
        }), 400
    except Exception as e:
        logger.error(f"Error in process_image_url: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    try:
        if vision_server is None:
            return jsonify({
                "success": False,
                "error": "Vision server not initialized"
            }), 500
        
        return jsonify({
            "success": True,
            "current_model": vision_server.last_model_type,
            "main_model_path": vision_server.config.vision.main_model_path,
            "architectural_model_path": vision_server.config.vision.architectural_model_path,
            "model_switch_interval": vision_server.config.vision.model_switch_interval
        })
        
    except Exception as e:
        logger.error(f"Error in model_info: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

def initialize_server():
    """Initialize the vision server"""
    global vision_server
    
    try:
        config = AppConfig()
        
        # Check if models exist
        for path in [config.vision.main_model_path, config.vision.architectural_model_path]:
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
        
        vision_server = VisionAIServer(config)
        logger.info("Vision AI server initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        return False

def main():
    """Main function to start the Flask server"""
    if not initialize_server():
        logger.error("Failed to initialize vision server")
        return
    
    # Start Flask server
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting Vision AI server on {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == "__main__":
    main()