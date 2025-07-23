# INSIGHT AI Vision Server

The INSIGHT AI Vision Server is a Flask-based REST API that provides object detection and distance estimation capabilities for mobile applications. It uses YOLO models for object detection and provides real-time analysis of images uploaded from mobile devices.

##  Features

- **Object Detection**: Uses YOLO11 models for accurate object recognition
- **Distance Estimation**: Calculates approximate distances to detected objects
- **Mobile App Integration**: Optimized endpoints for Flutter/mobile app integration
- **Real-time Processing**: Fast image processing with JSON responses
- **Cross-Origin Support**: CORS enabled for web and mobile app access

## Mobile App Integration

This server is designed to work seamlessly with the insight mobile app. 
The primary endpoint for mobile apps is:

### Primary Endpoint for Mobile Apps

**POST /** - Upload image file (multipart/form-data)

This endpoint accepts image uploads in multipart format, which is the standard for mobile file uploads.

```
Endpoint: POST /
Content-Type: multipart/form-data
Field name: file
```

**Response Format:**
```json
{
  "detected": "Person at 2.5 meters ahead, Chair at 1.8 meters to the right",
  "success": true,
  "total_objects": 5,
  "nearby_objects": 2,
  "model_type": "general"
}
```

### Alternative Endpoints

**POST /process** - JSON with base64 encoded image
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**POST /process_url** - Process image from URL
```json
{
  "url": "https://example.com/image.jpg"
}
```

**GET /health** - Health check endpoint
```json
{
  "status": "healthy",
  "timestamp": 1642694400.123,
  "model_loaded": true
}
```

**GET /model_info** - Get model information
```json
{
  "success": true,
  "current_model": "general",
  "main_model_path": "./yolo11n_object365.pt",
  "architectural_model_path": "./architectural_model.pt"
}
```

## 🛠️ Setup and Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- At least 4GB RAM for model loading
- GPU support (optional, for faster processing)

### Local Installation

1. **Clone and navigate to the server directory:**
```bash
cd "Insight Server"
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Ensure model files are present:**
- `yolo11n_object365.pt` - Main object detection model
- `architectural_model.pt` - Architectural elements detection model

5. **Run the server:**
```bash
python main.py
```

The server will start on `http://localhost:5000` by default.

### Docker Installation

1. **Build Docker image:**
```bash
docker build -t insight-ai-server .
```

2. **Run container:**
```bash
docker run -p 5000:5000 insight-ai-server
```

### Using Docker Compose

```bash
docker-compose up -d
```

## 🌐 Publishing with ngrok

ngrok is the recommended way to expose your local server to mobile apps for testing and development.

### Install ngrok

1. **Download ngrok:**
   - Visit [https://ngrok.com/download](https://ngrok.com/download)
   - Create a free account
   - Download the appropriate version for your system

2. **Authenticate ngrok:**
```bash
ngrok authtoken YOUR_AUTH_TOKEN
```

### Expose the Server

1. **Start your Insight Server locally:**
```bash
python main.py
```

2. **In a new terminal, expose the server:**
```bash
ngrok http 5000
```

3. **ngrok will provide URLs like:**
```
Forwarding  http://abc123.ngrok.io -> http://localhost:5000
Forwarding  https://abc123.ngrok.io -> http://localhost:5000
```

4. **Use the HTTPS URL in your mobile app:**
```
Base URL: https://abc123.ngrok.io
Primary endpoint: https://abc123.ngrok.io/
Health check: https://abc123.ngrok.io/health
```

### ngrok Configuration Options

For production-like testing, you can configure ngrok with custom domains:

```bash
# Custom subdomain (requires paid plan)
ngrok http 5000 --subdomain=insight-ai

# Custom domain (requires paid plan)
ngrok http 5000 --hostname=insight-ai.yourdomain.com

# Basic auth for security
ngrok http 5000 --basic-auth="username:password"
```


## ⚙️ Configuration

The server can be configured through environment variables or by modifying `config.py`:

### Environment Variables

```bash
export PORT=5000                    # Server port
export HOST=0.0.0.0                # Server host
export CONF_THRESHOLD=0.45          # Detection confidence threshold
export NEAR_THRESHOLD=6.0           # Distance threshold for "nearby" objects
export MODEL_SWITCH_INTERVAL=5      # Seconds between model switches
```

### Configuration File

Edit `config.py` to modify:
- Model paths
- Detection thresholds
- Object height/width mappings for distance estimation
- Camera parameters

## 🔧 API Response Details

### Success Response
```json
{
  "success": true,
  "timestamp": 1642694400.123,
  "model_type": "general",
  "total_objects": 5,
  "nearby_objects": 2,
  "objects": [
    {
      "distance": 2.5,
      "label": "Person",
      "position": "ahead",
      "bbox": [100, 150, 200, 400]
    }
  ],
  "announcement": "Person at 2.5 meters ahead, Chair at 1.8 meters to the right",
  "detected": "Person at 2.5 meters ahead, Chair at 1.8 meters to the right"
}
```

### Error Response
```json
{
  "success": false,
  "error": "No file uploaded",
  "detected": "No file uploaded"
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Model files not found:**
   - Ensure `yolo11n_object365.pt` and `architectural_model.pt` are in the server directory
   - Check file permissions

2. **Memory issues:**
   - Ensure at least 4GB RAM available
   - Consider using smaller model variants

3. **Image processing errors:**
   - Verify image format (JPEG, PNG supported)
   - Check image file size (large images may cause timeout)

4. **Mobile app connection issues:**
   - Verify ngrok URL is accessible
   - Check CORS settings
   - Ensure network connectivity

### Performance Optimization

1. **Use GPU acceleration:**
   - Install CUDA-compatible PyTorch
   - Ensure GPU drivers are installed

2. **Optimize for mobile:**
   - Compress images before upload
   - Implement client-side image resizing
   - Use appropriate confidence thresholds

## 📊 Model Information

### Primary Model (yolo11n_object365.pt)
- **Objects Detected:** 365+ common objects
- **Use Case:** General object detection
- **Suitable For:** Indoor/outdoor environments

### Architectural Model (architectural_model.pt)
- **Objects Detected:** Architectural elements (doors, walls, stairs, etc.)
- **Use Case:** Navigation assistance
- **Suitable For:** Indoor navigation, accessibility

The server automatically switches between models based on detection patterns to provide optimal results for different scenarios.


