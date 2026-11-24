"""
Augment existing dataset by applying transformations
Expands 100 images to ~400-500 images
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import random

def augment_image(img_path, output_dir, output_name, transformations_count=3):
    """Apply random transformations to create augmented copies"""
    img = cv2.imread(str(img_path))
    if img is None:
        return
    
    h, w = img.shape[:2]
    
    for aug_idx in range(transformations_count):
        aug_img = img.copy()
        
        # Random brightness
        if random.random() > 0.5:
            brightness = random.uniform(0.7, 1.3)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=brightness, beta=0)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, matrix, (w, h))
        
        # Random horizontal flip
        if random.random() > 0.5:
            aug_img = cv2.flip(aug_img, 1)
        
        # Random noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 5, aug_img.shape).astype(np.uint8)
            aug_img = cv2.add(aug_img, noise)
        
        # Random contrast
        if random.random() > 0.5:
            contrast = random.uniform(0.8, 1.2)
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(aug_img)
            l = cv2.convertScaleAbs(l, alpha=contrast, beta=0)
            aug_img = cv2.merge([l, a, b])
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_LAB2BGR)
        
        # Save augmented image
        aug_filename = f"{output_name.stem}_aug{aug_idx}.jpg"
        aug_path = output_dir / aug_filename
        cv2.imwrite(str(aug_path), aug_img)

def augment_dataset(target_images=500):
    """Augment existing dataset to reach target size"""
    
    data_dir = Path("data_small")
    train_imgs = data_dir / "images" / "train"
    train_labels = data_dir / "labels" / "train"
    val_imgs = data_dir / "images" / "val"
    val_labels = data_dir / "labels" / "val"
    
    print("=" * 60)
    print(f"AUGMENTING DATASET TO {target_images} IMAGES")
    print("=" * 60)
    
    # Count current images
    current_train = len(list(train_imgs.glob("*")))
    current_val = len(list(val_imgs.glob("*")))
    current_total = current_train + current_val
    
    print(f"\n📊 Current dataset:")
    print(f"   - Train: {current_train} images")
    print(f"   - Val: {current_val} images")
    print(f"   - Total: {current_total} images")
    
    # Calculate augmentations needed
    augmentations_per_image = (target_images - current_total) // current_total
    augmentations_per_image = max(1, min(5, augmentations_per_image))  # 1-5 aug per image
    
    print(f"\n🔄 Augmenting {augmentations_per_image} versions per image...")
    
    # Augment training images
    train_files = list(train_imgs.glob("*.jpg")) + list(train_imgs.glob("*.png"))
    for img_file in train_files:
        label_file = train_labels / f"{img_file.stem}.txt"
        try:
            augment_image(img_file, train_imgs, img_file, augmentations_per_image)
            
            # Copy & rename labels for augmented images
            if label_file.exists():
                label_content = label_file.read_text()
                for aug_idx in range(augmentations_per_image):
                    aug_label_name = f"{img_file.stem}_aug{aug_idx}.txt"
                    (train_labels / aug_label_name).write_text(label_content)
        except Exception as e:
            print(f"⚠️  Error augmenting {img_file.name}: {e}")
    
    # Augment validation images (fewer augmentations)
    val_files = list(val_imgs.glob("*.jpg")) + list(val_imgs.glob("*.png"))
    for img_file in val_files:
        label_file = val_labels / f"{img_file.stem}.txt"
        try:
            augment_image(img_file, val_imgs, img_file, max(1, augmentations_per_image // 2))
            
            # Copy & rename labels for augmented images
            if label_file.exists():
                label_content = label_file.read_text()
                for aug_idx in range(max(1, augmentations_per_image // 2)):
                    aug_label_name = f"{img_file.stem}_aug{aug_idx}.txt"
                    (val_labels / aug_label_name).write_text(label_content)
        except Exception as e:
            print(f"⚠️  Error augmenting {img_file.name}: {e}")
    
    # Count final
    final_train = len(list(train_imgs.glob("*")))
    final_val = len(list(val_imgs.glob("*")))
    final_total = final_train + final_val
    
    print(f"\n✅ Augmentation complete!")
    print(f"📊 Final dataset:")
    print(f"   - Train: {final_train} images")
    print(f"   - Val: {final_val} images")
    print(f"   - Total: {final_total} images (+{final_total - current_total} new)")

if __name__ == "__main__":
    print("\n⚠️  Note: Original Revitsone dataset was deleted during cleanup.")
    print("Using data augmentation instead to expand existing dataset.\n")
    
    augment_dataset(target_images=500)
