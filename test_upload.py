#!/usr/bin/env python3
"""
Simple test script to test image upload endpoint
Usage: python test_upload.py <image_path>
"""

import requests
import sys
from pathlib import Path

def test_upload(image_path: str, base_url: str = "http://localhost:8000"):
    """Test the /identify endpoint with an image file."""
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"❌ Error: File not found: {image_path}")
        return
    
    print(f"🖼️  Testing with image: {image_path}")
    print(f"🌐 Sending to: {base_url}/identify")
    
    # Prepare the file
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        
        # Send request
        try:
            response = requests.post(
                f"{base_url}/identify",
                files=files,
                timeout=30
            )
            
            print(f"\n📊 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"Plant: {result.get('plant_name')}")
                print(f"Confidence: {result.get('confidence'):.2%}")
                print(f"Processing Time: {result.get('processing_time_ms'):.0f}ms")
            else:
                print("❌ Error!")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_upload.py <image_path> [base_url]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    test_upload(image_path, base_url)

