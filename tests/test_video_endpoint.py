#!/usr/bin/env python3
"""
Test script for the video endpoint.
Standalone executable: python tests/test_video_endpoint.py

Creates a minimal synthetic test video and sends it to the API.
"""

import cv2
import numpy as np
import tempfile
import requests
import json
import os
import sys

def create_synthetic_video(output_path, num_frames=10, size=(224, 224)):
    """
    Create a minimal synthetic test video with solid color frames.
    
    Args:
        output_path: Path to save the video file
        num_frames: Number of frames to create
        size: Frame size (width, height)
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, size)
    
    for i in range(num_frames):
        # Create a simple colored frame (varying to make it interesting)
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # Alternate between red and blue frames
        if i % 2 == 0:
            frame[:, :] = [255, 0, 0]  # Red (BGR format)
        else:
            frame[:, :] = [0, 0, 255]  # Blue
        out.write(frame)
    
    out.release()
    print(f"✓ Created synthetic test video: {output_path}")


def test_video_endpoint(video_path, api_url="http://127.0.0.1:8000"):
    """
    Send a video file to the API and test the response.
    
    Args:
        video_path: Path to the video file
        api_url: Base URL of the API
    """
    endpoint = f"{api_url}/predict_video"
    
    print(f"\n📤 Sending video to {endpoint}...")
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': ('test_video.mp4', f, 'video/mp4')}
            response = requests.post(endpoint, files=files, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ Server error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        print(f"\n📥 API Response:")
        print(json.dumps(data, indent=2))
        
        # Assertions
        print(f"\n🔍 Validating response...")
        required_fields = [
            'is_fake',
            'confidence',
            'label',
            'frame_confidences',
            'top_frame_index',
            'heatmap_base64',
            'frames_analyzed'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
            print(f"  ✓ {field}")
        
        assert isinstance(data['frame_confidences'], list), "frame_confidences should be a list"
        assert len(data['frame_confidences']) > 0, "frame_confidences should not be empty"
        print(f"  ✓ frame_confidences is non-empty list ({len(data['frame_confidences'])} frames)")
        
        assert isinstance(data['is_fake'], bool), "is_fake should be a boolean"
        assert isinstance(data['confidence'], (int, float)), "confidence should be numeric"
        assert isinstance(data['label'], str), "label should be a string"
        assert isinstance(data['top_frame_index'], int), "top_frame_index should be an int"
        assert isinstance(data['heatmap_base64'], str), "heatmap_base64 should be a string"
        assert isinstance(data['frames_analyzed'], int), "frames_analyzed should be an int"
        
        print(f"\n✅ PASSED - All assertions passed!")
        return True
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error: Could not reach {endpoint}")
        print(f"   Make sure the FastAPI server is running at {api_url}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main test execution."""
    print("=" * 60)
    print("Video Endpoint Test")
    print("=" * 60)
    
    # Create temporary video file
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_video.mp4")
        create_synthetic_video(video_path)
        
        # Test the endpoint
        success = test_video_endpoint(video_path)
        
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
