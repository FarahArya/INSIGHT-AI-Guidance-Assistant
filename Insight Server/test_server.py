#!/usr/bin/env python3
"""
Test script to verify the Vision AI server is working properly
"""

import requests
import json
import time
import os
import sys

def test_server_health(base_url):
    """Test server health endpoint"""
    try:
        print(f"🔍 Testing server health at {base_url}/health")
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {base_url}")
        print("   Make sure the server is running and the URL is correct")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_file_upload(base_url, image_path=None):
    """Test file upload endpoint (simulating Flutter app)"""
    print(f"\n📤 Testing file upload at {base_url}/")
    
    # Create a simple test image if none provided
    if image_path is None or not os.path.exists(image_path):
        print("📝 Creating test image...")
        try:
            import cv2
            import numpy as np
            
            # Create a simple test image
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            test_image[:] = (100, 150, 200)  # Blue-ish color
            cv2.putText(test_image, 'TEST IMAGE', (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            image_path = 'test_image.jpg'
            cv2.imwrite(image_path, test_image)
            print(f"✅ Test image created: {image_path}")
            
        except ImportError:
            print("❌ OpenCV not available, cannot create test image")
            return False
        except Exception as e:
            print(f"❌ Failed to create test image: {e}")
            return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            
            print(f"📤 Uploading {image_path}...")
            response = requests.post(f"{base_url}/", files=files, timeout=30)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("✅ Upload successful!")
                print(f"   Detected: {data.get('detected', 'N/A')}")
                print(f"   Success: {data.get('success', 'N/A')}")
                if 'total_objects' in data:
                    print(f"   Total objects: {data.get('total_objects')}")
                if 'model_type' in data:
                    print(f"   Model type: {data.get('model_type')}")
                return True
            except json.JSONDecodeError:
                print("❌ Invalid JSON response")
                print(f"Raw response: {response.text[:200]}")
                return False
        else:
            print(f"❌ Upload failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detected', error_data.get('error', 'Unknown error'))}")
            except:
                print(f"   Raw error: {response.text[:200]}")
            return False
            
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out (server might be processing)")
        return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def test_model_info(base_url):
    """Test model info endpoint"""
    print(f"\n📊 Testing model info at {base_url}/model_info")
    
    try:
        response = requests.get(f"{base_url}/model_info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved!")
            print(f"   Current model: {data.get('current_model')}")
            print(f"   Main model: {data.get('main_model_path')}")
            print(f"   Arch model: {data.get('architectural_model_path')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def main():
    """Main test function"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip('/')
    else:
        base_url = input("Enter server URL (e.g., http://192.168.1.100:5000): ").rstrip('/')
    
    if not base_url.startswith('http'):
        base_url = f"http://{base_url}"
    
    print(f"🚀 Testing Vision AI Server at: {base_url}")
    print("=" * 50)
    
    # Test server health
    if not test_server_health(base_url):
        print("\n❌ Server health check failed. Cannot continue tests.")
        return
    
    # Test model info
    test_model_info(base_url)
    
    # Test file upload
    image_path = None
    if len(sys.argv) > 2:
        image_path = sys.argv[2]
    
    success = test_file_upload(base_url, image_path)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests completed! Your server should work with the Flutter app.")
    else:
        print("❌ Some tests failed. Check the server logs and configuration.")
    
    # Clean up test image
    if os.path.exists('test_image.jpg'):
        try:
            os.remove('test_image.jpg')
            print("🗑️ Test image cleaned up")
        except:
            pass

if __name__ == "__main__":
    main()
