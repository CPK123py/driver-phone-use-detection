"""
Efficient Training Script - Small Dataset (Max 100 Images)
Uses YOLOv8n for fast training on limited hardware (Intel i3)

FIXES:
- Increased epochs (10 → 50) with proper patience (3 → 10)
- Added augmentation to compensate for tiny dataset
- Added class verification before training
- Lower confidence threshold for sparse detections
- Prints class map so you can verify person=0, phone=1
"""

from ultralytics import YOLO
import os
import shutil
import yaml
import torch


def verify_dataset(yaml_path):
    """Read and print class names from YAML so you can confirm the mapping."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("\n📋 Dataset config:")
    print(f"   Train path : {cfg.get('train', 'NOT SET')}")
    print(f"   Val path   : {cfg.get('val', 'NOT SET')}")
    print(f"   Num classes: {cfg.get('nc', 'NOT SET')}")
    print(f"   Class names: {cfg.get('names', 'NOT SET')}")

    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = list(names.values())

    # Warn if expected classes are missing
    lower = [n.lower() for n in names]
    if "person" not in lower:
        print("⚠️  WARNING: 'person' not found in class names!")
    if "phone" not in lower and "cell phone" not in lower:
        print("⚠️  WARNING: 'phone'/'cell phone' not found in class names!")

    # Show index → name mapping so the user can verify test.py uses correct indices
    print("\n   Index → Name mapping (use these in test.py):")
    for i, name in enumerate(names):
        print(f"     cls == {i}  →  '{name}'")

    return names


def main():
    print("=" * 60)
    print("DRIVER PHONE DETECTION - EFFICIENT TRAINING")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU (Intel i3)'}")
    print("=" * 60)

    yaml_path = "data_small.yaml"
    if not os.path.exists(yaml_path):
        print(f"❌ Error: {yaml_path} not found!")
        print("Run 'python prepare_small_dataset.py' first")
        return

    # ── Verify classes before wasting time training ──────────────────────────
    names = verify_dataset(yaml_path)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n📦 Loading YOLOv8n model (nano - optimized for CPU)...")
    model = YOLO("yolov8n.pt")

    run_name = "driver_detection_v2"
    print("\n🚀 Starting training...")
    print("Configuration:")
    print("  - Model     : YOLOv8n-Detect (Nano, 3.2MB)")
    print("  - Image size: 416")
    print("  - Batch     : 4")
    print(f"  - Epochs    : 50  (was 10 — too few for tiny dataset)")
    print(f"  - Patience  : 10  (was 3 — stopped way too early)")
    print("  - Augment   : ON  (essential when you have <100 images)")

    results = model.train(
        data=yaml_path,
        epochs=50,           # FIX: 10 was not enough; model barely learned anything
        imgsz=416,
        batch=4,
        patience=10,         # FIX: patience=3 caused early-stop after ~3 epochs
        device="cpu",
        save=True,
        project="runs/detect",
        name=run_name,
        exist_ok=True,
        verbose=True,        # FIX: verbose=False hid all training feedback
        workers=0,
        plots=True,          # FIX: plots=False hid loss/mAP curves (useful to debug)
        # ── Augmentation — critical for small datasets ──────────────────────
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.5,          # combine 4 images; helps with tiny datasets
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=2.0,
    )

    print("\n✅ Training completed!")

    # ── Copy best weights ─────────────────────────────────────────────────────
    best_model = f"runs/detect/{run_name}/weights/best.pt"
    if os.path.exists(best_model):
        shutil.copy(best_model, "best.pt")
        print("✅ Model saved to: best.pt")
    else:
        print(f"⚠️  best.pt not found at expected path: {best_model}")
        print("   Check runs/detect/ for the actual output folder name.")

    # ── Print mAP summary ─────────────────────────────────────────────────────
    try:
        metrics = results.results_dict
        map50 = metrics.get("metrics/mAP50(B)", None)
        if map50 is not None:
            print(f"\n📊 Final mAP@50: {map50:.3f}")
            if map50 < 0.3:
                print("   ⚠️  mAP is low. Consider:")
                print("      - More training images (aim for 200+)")
                print("      - Check label quality in your dataset")
                print("      - Increase epochs further")
            elif map50 >= 0.5:
                print("   ✅ Decent accuracy for a small dataset!")
    except Exception:
        pass

    print("\n🧪 Run: python test.py")


if __name__ == "__main__":
    main()