#!/usr/bin/env python3
"""
Extract sample images from UC3M-LP dataset for demo.
Extracts 500 random images from the dataset ZIP file.
"""

import zipfile
import random
import os
from pathlib import Path

# Configuration
DATASET_ZIP = "../License Plate Recognition.v4-resized640_aug3x-accurate.yolov8.zip"
OUTPUT_DIR = "demo_images"
NUM_SAMPLES = 500

def extract_demo_images():
    """Extract random sample images from dataset."""
    
    print(f"📦 Opening dataset: {DATASET_ZIP}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Open ZIP file
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        # Get all image files
        all_files = [f for f in zip_ref.namelist() 
                     if f.endswith(('.jpg', '.jpeg', '.png')) 
                     and '/images/' in f]
        
        print(f"📊 Found {len(all_files)} images in dataset")
        
        # Sample random images
        num_to_extract = min(NUM_SAMPLES, len(all_files))
        selected_files = random.sample(all_files, num_to_extract)
        
        print(f"🎯 Extracting {num_to_extract} random images...")
        
        # Extract selected files
        for i, file_path in enumerate(selected_files, 1):
            # Get just the filename
            filename = Path(file_path).name
            
            # Extract to output directory
            source = zip_ref.open(file_path)
            target_path = os.path.join(OUTPUT_DIR, filename)
            
            with open(target_path, 'wb') as target:
                target.write(source.read())
            
            if i % 50 == 0:
                print(f"  ✓ Extracted {i}/{num_to_extract} images...")
        
        print(f"\n✅ Successfully extracted {num_to_extract} images to {OUTPUT_DIR}/")
        print(f"📁 Total size: {sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR)) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    extract_demo_images()
