"""
Prepare Small Dataset from Revitsone (Max 100 Images)
Converts classification labels to YOLO detection format
Efficient for CPU training on Intel i3
"""

import os
import shutil
import random
from pathlib import Path

def prepare_revitsone_small_dataset(max_images=500):
    """Create small YOLO dataset from Revitsone classification data"""
    
    output_dir = Path("data_small")
    train_imgs = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    val_imgs = output_dir / "images" / "val"
    val_labels = output_dir / "labels" / "val"
    
    # Create directories
    for d in [train_imgs, train_labels, val_imgs, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Creating YOLO dataset from Revitsone")
    print(f"🎯 Maximum {max_images} images\n")
    
    # Source folder
    source_root = Path("Revitsone-5classes/Revitsone-5classes")
    
    if not source_root.exists():
        print(f"❌ Revitsone dataset not found at {source_root}")
        return False
    
    # Class folders and their phone usage
    classes_info = {
        "safe_driving": {"has_phone": False, "yolo_id": 0},
        "other_activities": {"has_phone": False, "yolo_id": 0},
        "talking_phone": {"has_phone": True, "yolo_id": 1},
        "texting_phone": {"has_phone": True, "yolo_id": 1},
        "turning": {"has_phone": False, "yolo_id": 0},
    }
    
    all_images = []
    
    # Collect images from each class
    for class_name, class_info in classes_info.items():
        class_path = source_root / class_name
        if not class_path.exists():
            print(f"⚠️  Folder not found: {class_name}")
            continue
        
        images = list(class_path.glob("*"))
        images = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        print(f"  {class_name}: {len(images)} images (phone={class_info['has_phone']})")
        
        for img in images:
            all_images.append((img, class_name, class_info))
    
    if not all_images:
        print("❌ No images found!")
        return False
    
    # Sample max images
    selected = random.sample(all_images, min(max_images, len(all_images)))
    print(f"\n✅ Selected {len(selected)} images")
    
    # Split: 80% train, 20% val
    split_idx = int(len(selected) * 0.8)
    train_set = selected[:split_idx]
    val_set = selected[split_idx:]
    
    print(f"   - Train: {len(train_set)} images")
    print(f"   - Val: {len(val_set)} images\n")
    
    # Helper function to create YOLO label
    def create_yolo_label(has_phone, img_path):
        """Create YOLO format label file"""
        labels = []
        
        # Person always fills most of frame: (center_x, center_y, width, height) normalized
        labels.append("0 0.5 0.5 0.9 0.9")  # Person box - large
        
        # Phone box if phone usage detected
        if has_phone:
            # Phone in hand area (upper right for right hand)
            labels.append("1 0.65 0.35 0.25 0.35")  # Phone box - hand area
        
        return "\n".join(labels)
    
    # Process training set
    for img_path, class_name, class_info in train_set:
        new_name = f"{class_name}_{img_path.stem}.jpg"
        new_img_path = train_imgs / new_name
        new_label_path = train_labels / f"{new_name[:-4]}.txt"
        
        shutil.copy2(img_path, new_img_path)
        
        label_content = create_yolo_label(class_info["has_phone"], img_path)
        with open(new_label_path, "w") as f:
            f.write(label_content)
    
    # Process validation set
    for img_path, class_name, class_info in val_set:
        new_name = f"{class_name}_{img_path.stem}.jpg"
        new_img_path = val_imgs / new_name
        new_label_path = val_labels / f"{new_name[:-4]}.txt"
        
        shutil.copy2(img_path, new_img_path)
        
        label_content = create_yolo_label(class_info["has_phone"], img_path)
        with open(new_label_path, "w") as f:
            f.write(label_content)
    
    # Create data_small.yaml
    yaml_content = """path: data_small
train: images/train
val: images/val

nc: 2
names: ['person', 'phone']
"""
    
    with open("data_small.yaml", "w") as f:
        f.write(yaml_content)
    
    print("✅ Dataset created and labeled!")
    print(f"   📁 Location: {output_dir}/")
    print(f"   📝 Config: data_small.yaml")
    print(f"   🖼️  Total images: {len(train_set) + len(val_set)}")
    print(f"   📊 Person boxes: {len(train_set) + len(val_set)} (all)")
    print(f"   📱 Phone boxes: {sum(1 for _, _, info in selected if info['has_phone'])}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("PREPARING DATASET FROM REVITSONE (MAX 500 IMAGES)")
    print("=" * 60 + "\n")
    
    success = prepare_revitsone_small_dataset(max_images=500)
    
    if success:
        print("\n" + "=" * 60)
        print("🚀 Next: Run 'python train_small.py'")
        print("=" * 60)
    else:
        print("\n❌ Failed to create dataset")
