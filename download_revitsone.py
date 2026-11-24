"""
Download Revitsone 5-Class Driver Behavior Dataset
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def download_revitsone():
    """Download Revitsone dataset from source"""
    
    print("Downloading Revitsone 5-Class Dataset...")
    print("(This contains ~10K images for driver behavior classification)")
    
    # Direct download link (adjust if needed)
    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/mansimov/revitsone-5classes"
    zip_path = "revitsone.zip"
    extract_path = "."
    
    try:
        print("\n⏳ Downloading dataset (may take a few minutes)...")
        urllib.request.urlretrieve(dataset_url, zip_path)
        
        print("✅ Download complete!")
        print("⏳ Extracting...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        print("✅ Extraction complete!")
        print(f"📁 Dataset ready at: Revitsone-5classes/")
        
        # Clean up zip
        os.remove(zip_path)
        print("✅ Cleaned up zip file")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nAlternative: Download manually from Kaggle:")
        print("  1. Go to: https://www.kaggle.com/datasets/mansimov/revitsone-5classes")
        print("  2. Click 'Download' and extract to this folder")
        print("  3. Then run: python prepare_small_dataset.py")
        return False

if __name__ == "__main__":
    if not Path("Revitsone-5classes").exists():
        download_revitsone()
    else:
        print("✅ Revitsone dataset already exists!")
