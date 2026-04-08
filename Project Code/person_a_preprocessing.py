# =============================================================================
# IT 371 — Smart Traffic Monitoring System
# PERSON A: Data Collection & Preprocessing
# Run this in Google Colab, cell by cell.
# When you are done, share the output folder link with Person B.
# =============================================================================


# =============================================================================
# CELL 1 — Install libraries
# Run this first. It will take 1-2 minutes.
# =============================================================================

# !pip install fiftyone tensorflow opencv-python-headless


# =============================================================================
# CELL 2 — Mount Google Drive
# This is where your processed dataset will be saved.
# After running, click the link and allow access.
# =============================================================================

# from google.colab import drive
# drive.mount('/content/drive')


# =============================================================================
# CELL 3 — Imports
# =============================================================================

import os
import json
import shutil
import numpy as np
import cv2
from pathlib import Path


# =============================================================================
# CELL 4 — Configuration
# You can change these numbers if Colab runs out of memory.
# 1000 images per class is a safe starting point.
# =============================================================================

IMAGES_PER_CLASS = 1000        # How many images to use per class
IMAGE_SIZE       = (224, 224)  # Standard CNN input size — do not change
TRAIN_RATIO      = 0.70        # 70% of images go to training
VAL_RATIO        = 0.15        # 15% go to validation
TEST_RATIO       = 0.15        # 15% go to testing (used by Person C)
RANDOM_SEED      = 42          # Keeps splits consistent across all team members

# Where the final dataset will be saved on your Google Drive
OUTPUT_DIR = '/content/drive/MyDrive/IT371_Dataset'

# The 6 classes we care about — these match COCO label names exactly
CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']

print("Configuration loaded.")
print(f"  Images per class : {IMAGES_PER_CLASS}")
print(f"  Image size       : {IMAGE_SIZE}")
print(f"  Train / Val / Test split: {TRAIN_RATIO} / {VAL_RATIO} / {TEST_RATIO}")
print(f"  Output directory : {OUTPUT_DIR}")


# =============================================================================
# CELL 5 — Download COCO dataset (filtered to our 6 classes)
# fiftyone makes this very easy. It only downloads what we need.
# This will take 5-15 minutes depending on your connection.
# =============================================================================

import fiftyone as fo
import fiftyone.zoo as foz

print("Downloading COCO 2017 validation split (filtered to our 6 classes)...")
print("This may take several minutes — do not close the tab.\n")

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",           # Use validation split — smaller and faster to download
    label_types=["detections"],
    classes=CLASSES,
    max_samples=IMAGES_PER_CLASS * len(CLASSES),
    seed=RANDOM_SEED,
    shuffle=True,
)

print(f"\nDownload complete. Total samples loaded: {len(dataset)}")


# =============================================================================
# CELL 6 — Inspect a few samples to confirm the data looks right
# You should see image paths and label annotations printed below.
# =============================================================================

print("Sample inspection — first 3 entries:\n")
for i, sample in enumerate(dataset.take(3)):
    labels = []
    if sample.ground_truth is not None:
        labels = list(set([det.label for det in sample.ground_truth.detections]))
    print(f"  Sample {i+1}: {sample.filepath}")
    print(f"    Labels present: {labels}\n")


# =============================================================================
# CELL 7 — Extract and organize images by their dominant class
#
# COCO images can contain multiple object types. We assign each image
# to one class by finding whichever of our 6 classes appears most
# frequently in that image. This gives us clean single-label data
# that our CNN can train on.
# =============================================================================

def get_dominant_class(sample, classes):
    """Return whichever of our 6 classes appears most in this image."""
    if sample.ground_truth is None:
        return None
    counts = {}
    for det in sample.ground_truth.detections:
        if det.label in classes:
            counts[det.label] = counts.get(det.label, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


print("Sorting images by dominant class...")

# Build a dict: class_name -> list of image file paths
class_images = {cls: [] for cls in CLASSES}

for sample in dataset:
    dominant = get_dominant_class(sample, CLASSES)
    if dominant is None:
        continue
    if len(class_images[dominant]) < IMAGES_PER_CLASS:
        class_images[dominant].append(sample.filepath)

print("\nImages collected per class:")
for cls, paths in class_images.items():
    print(f"  {cls:12s}: {len(paths)} images")


# =============================================================================
# CELL 8 — Preprocess and save images
#
# For each image we:
#   1. Read it with OpenCV
#   2. Resize to 224x224
#   3. Normalize pixel values from 0-255 to 0.0-1.0
#   4. Apply augmentation (horizontal flip + brightness jitter)
#   5. Assign to train, val, or test split
#   6. Save to Google Drive
#
# This cell does the real work. It will take a few minutes.
# =============================================================================

import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def augment_image(img):
    """
    Apply simple augmentation to a single image.
    Returns a list containing the original + augmented versions.
    """
    augmented = [img]

    # Horizontal flip
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)

    # Brightness adjustment (slightly brighter)
    bright = np.clip(img * 1.2, 0.0, 1.0)
    augmented.append(bright)

    # Brightness adjustment (slightly darker)
    dark = np.clip(img * 0.8, 0.0, 1.0)
    augmented.append(dark)

    return augmented


def preprocess_image(filepath, target_size):
    """
    Read, resize, and normalize a single image.
    Returns None if the image is corrupted or unreadable.
    """
    try:
        img = cv2.imread(filepath)
        if img is None:
            return None                          # Skip corrupted files
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR, convert to RGB
        img = cv2.resize(img, target_size)           # Resize to 224x224
        img = img.astype(np.float32) / 255.0         # Normalize to 0-1
        return img
    except Exception:
        return None


# Create output folder structure
splits = ['train', 'val', 'test']
for split in splits:
    for cls in CLASSES:
        Path(f"{OUTPUT_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

# Stats tracking
stats = {'train': {}, 'val': {}, 'test': {}, 'corrupted': 0}
for split in splits:
    for cls in CLASSES:
        stats[split][cls] = 0

print("Preprocessing and saving images...")
print("(This will take a few minutes)\n")

for cls, paths in class_images.items():
    random.shuffle(paths)

    n = len(paths)
    train_end = int(n * TRAIN_RATIO)
    val_end   = train_end + int(n * VAL_RATIO)

    split_map = (
        [(p, 'train') for p in paths[:train_end]] +
        [(p, 'val')   for p in paths[train_end:val_end]] +
        [(p, 'test')  for p in paths[val_end:]]
    )

    for filepath, split in split_map:
        img = preprocess_image(filepath, IMAGE_SIZE)
        if img is None:
            stats['corrupted'] += 1
            continue

        # Only augment training images (never augment val or test)
        if split == 'train':
            versions = augment_image(img)
        else:
            versions = [img]

        for i, version in enumerate(versions):
            base_name = Path(filepath).stem
            out_name  = f"{base_name}_aug{i}.npy" if i > 0 else f"{base_name}.npy"
            out_path  = f"{OUTPUT_DIR}/{split}/{cls}/{out_name}"
            np.save(out_path, version)
            stats[split][cls] += 1

    print(f"  {cls:12s} done.")

print("\nPreprocessing complete!")


# =============================================================================
# CELL 9 — Print final dataset summary
# Copy this output and paste it into your Update 2 slide.
# =============================================================================

print("\n" + "="*55)
print("  DATASET SUMMARY — paste this into your slide")
print("="*55)

total_train = sum(stats['train'].values())
total_val   = sum(stats['val'].values())
total_test  = sum(stats['test'].values())
total_all   = total_train + total_val + total_test

print(f"\n  Corrupted images skipped: {stats['corrupted']}")
print(f"\n  {'Class':<14} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
print(f"  {'-'*42}")
for cls in CLASSES:
    t  = stats['train'][cls]
    v  = stats['val'][cls]
    te = stats['test'][cls]
    print(f"  {cls:<14} {t:>7} {v:>7} {te:>7} {t+v+te:>7}")
print(f"  {'-'*42}")
print(f"  {'TOTAL':<14} {total_train:>7} {total_val:>7} {total_test:>7} {total_all:>7}")
print(f"\n  Split ratios: {TRAIN_RATIO*100:.0f}% train / "
      f"{VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test")
print(f"  Image size:   {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels, normalized to [0, 1]")
print(f"  Augmentation: horizontal flip, brightness +20%, brightness -20%")
print(f"  Saved to:     {OUTPUT_DIR}")
print("="*55)


# =============================================================================
# CELL 10 — Save dataset metadata
# This creates a JSON file that Person B can load to know exactly
# what the dataset contains without having to re-read every file.
# =============================================================================

metadata = {
    "classes"       : CLASSES,
    "image_size"    : list(IMAGE_SIZE),
    "split_ratios"  : {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
    "random_seed"   : RANDOM_SEED,
    "counts"        : {
        "train"     : stats['train'],
        "val"       : stats['val'],
        "test"      : stats['test'],
    },
    "output_dir"    : OUTPUT_DIR,
    "augmentation"  : ["horizontal_flip", "brightness_+20pct", "brightness_-20pct"],
    "normalization" : "pixel values divided by 255.0 — range [0.0, 1.0]",
}

metadata_path = f"{OUTPUT_DIR}/dataset_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved to: {metadata_path}")
print("\nShare the following folder with Person B:")
print(f"  {OUTPUT_DIR}")
print("\nPerson B should load data using the path above.")
print("Done! Your preprocessing is complete.")
