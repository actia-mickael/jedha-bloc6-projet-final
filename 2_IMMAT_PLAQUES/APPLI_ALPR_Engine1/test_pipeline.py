#!/usr/bin/env python3
"""
Test script to verify ALPR pipeline with demo images.
"""

import sys
import os
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline import ALPRPipeline
import random

def test_pipeline():
    """Test the pipeline with random demo images."""
    
    print("=" * 60)
    print("🧪 Testing ALPR Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n📦 Initializing pipeline...")
    pipeline = ALPRPipeline()
    
    # Get demo images
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print(f"❌ Demo images directory not found: {demo_dir}")
        return
    
    images = list(demo_dir.glob("*.jpg"))
    if not images:
        print(f"❌ No images found in {demo_dir}")
        return
    
    print(f"✅ Found {len(images)} demo images")
    
    # Test with 5 random images
    test_images = random.sample(images, min(5, len(images)))
    
    print(f"\n🎯 Testing with {len(test_images)} random images...\n")
    
    success_count = 0
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_images)}: {img_path.name}")
        print(f"{'='*60}")
        
        try:
            # Process image
            results = pipeline.process_image(str(img_path), conf_threshold=0.5)
            
            # Check results
            detections = results['metadata']['detections']
            conditions = results['metadata']['conditions']
            ocr_results = results['step4_ocr']
            
            print(f"\n📊 Image Conditions:")
            print(f"   Lighting: {conditions['lighting_emoji']} {conditions['lighting']}")
            print(f"   Brightness: {conditions['brightness']:.1f}/255")
            print(f"   Blur: {conditions['blur']}")
            
            print(f"\n🔍 Detection Results:")
            print(f"   Plates detected: {len(detections)}")
            
            if detections:
                for j, det in enumerate(detections, 1):
                    print(f"\n   Plate {j}:")
                    print(f"      Confidence: {det['confidence']:.2%}")
                    print(f"      BBox: {det['bbox']}")
                
                print(f"\n📝 OCR Results:")
                if ocr_results:
                    for j, ocr in enumerate(ocr_results, 1):
                        print(f"   Plate {j}:")
                        print(f"      Text: '{ocr['text']}'")
                        print(f"      OCR Confidence: {ocr['confidence']:.2%}")
                        print(f"      Detection Confidence: {ocr['detection_confidence']:.2%}")
                    
                    if any(r['text'] for r in ocr_results):
                        success_count += 1
                        print(f"\n   ✅ SUCCESS - Plate text detected!")
                    else:
                        print(f"\n   ⚠️ PARTIAL - Plate detected but OCR failed")
                else:
                    print(f"   ❌ No OCR results")
            else:
                print(f"   ❌ No plates detected")
                
        except Exception as e:
            print(f"\n❌ Error processing image: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(test_images)}")
    print(f"Successful detections: {success_count}")
    print(f"Success rate: {success_count/len(test_images)*100:.1f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    test_pipeline()
