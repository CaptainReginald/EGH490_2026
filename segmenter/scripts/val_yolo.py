#! /usr/bin/env python3
""" val_segmenter.py
 Validate a YOLO model against a dataset. Computes True Positives (TP), False Positives (FP),
 False Negatives (FN), True Negatives (TN), precision, recall, and F1 scores.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION: MODIFY AS NEEDED
# -----------------------------
PROJECT_DIR = "cgras_segmentation"
NAME = "val_dataAmil_modelAlor_20250624"
CONF_THRESH = 0.5
IOU_THRESH = 0.5
IGNORE_CLASSES = None
weights_file = "cgras_segmentation/train_aspa/cgras_data_2023Aspa-Alor_trained_20250624_modelyolov8nseg.pt"
data_file = "data/aspa/segmentation_outputs/aspa_alor117_filtered_split_tiled_balanced/cgras_data.yaml"

# data/aspa/segmentation_outputs/aspa_alor117_filtered_split_tiled_balanced/cgras_data.yaml
# data/pdae/segmentation_outputs/pdae140_filtered_split_tiled_balanced/cgras_data.yaml

# -----------------------------
# MODEL INITIALIZATION & VALIDATION
# -----------------------------
model = YOLO(weights_file)
metrics_d = model.val(conf=CONF_THRESH, iou=IOU_THRESH, project=PROJECT_DIR, name=NAME, data=data_file, plots=True, split="test")

# Get confusion matrix
conf_mat_d = metrics_d.confusion_matrix.matrix  # Confusion matrix

# Access the segmentation mAP@50
map50 = metrics_d.seg.map50

# Access precision, recall and F1
precision = metrics_d.seg.p  # Precision
recall = metrics_d.seg.r     # Recall
f1 = metrics_d.seg.f1        # F1 score


# Print the values
print(f"Segmentation mAP@50: {map50:.4f}")

# Handle precision, recall, and F1 which may be arrays
if hasattr(precision, 'shape') and precision.size > 1:
    # If it's an array with multiple values (per class)
    print(f"Precision (mean): {np.mean(precision):.4f}")
    print("Precision per class:")
    class_names = metrics_d.names if hasattr(metrics_d, 'names') else [f'Class {i}' for i in range(len(precision))]
    for i, p in enumerate(precision):
        print(f"  {class_names[i]}: {p:.4f}")
else:
    # If it's a single value
    print(f"Precision: {float(precision):.4f}")

# Do the same for recall
if hasattr(recall, 'shape') and recall.size > 1:
    print(f"Recall (mean): {np.mean(recall):.4f}")
    print("Recall per class:")
    class_names = metrics_d.names if hasattr(metrics_d, 'names') else [f'Class {i}' for i in range(len(recall))]
    for i, r in enumerate(recall):
        print(f"  {class_names[i]}: {r:.4f}")
else:
    print(f"Recall: {float(recall):.4f}")

# And for F1
if hasattr(f1, 'shape') and f1.size > 1:
    print(f"F1 Score (mean): {np.mean(f1):.4f}")
    print("F1 Score per class:")
    class_names = metrics_d.names if hasattr(metrics_d, 'names') else [f'Class {i}' for i in range(len(f1))]
    for i, f in enumerate(f1):
        print(f"  {class_names[i]}: {f:.4f}")
else:
    print(f"F1 Score: {float(f1):.4f}")

# Extract per-class mAP50 for segmentation
print(f"\n=== PER-CLASS mAP50 EXTRACTION ===")

import code
code.interact(local=dict(globals(), **locals()))

# Method 1: Try to get per-class mAP50 from segmentation metrics
if hasattr(metrics_d.seg, 'ap'):
    # The ap attribute typically contains [num_classes, num_iou_thresholds]
    # mAP@50 corresponds to IoU threshold 0.5 (index 0)
    ap_values = metrics_d.seg.ap
    print(f"AP array shape: {ap_values.shape}")
    
    if ap_values.shape[1] > 0:
        # Extract mAP50 per class (first IoU threshold)
        map50_per_class = ap_values[:, 0]
        print(f"Per-class Segmentation mAP@50:")
        
        class_names = metrics_d.names if hasattr(metrics_d, 'names') else {i: f'Class {i}' for i in range(len(map50_per_class))}
        
        for i, class_map50 in enumerate(map50_per_class):
            class_name = class_names[i] if i in class_names else f'Class {i}'
            print(f"  {class_name}: {class_map50:.4f}")
            
            # Highlight alive coral
            if 'alive' in class_name.lower():
                print(f"  *** ALIVE CORAL mAP@50: {class_map50:.4f} ***")

import code
code.interact(local=dict(globals(), **locals()))

# Method 2: Check if there's a direct maps attribute for segmentation
if hasattr(metrics_d.seg, 'maps'):
    seg_maps = metrics_d.seg.maps
    print(f"\nSegmentation maps (mAP50-95 per class): {seg_maps}")

import code
code.interact(local=dict(globals(), **locals()))

# Method 3: Check for map50 per class in results_dict
if hasattr(metrics_d, 'results_dict'):
    results = metrics_d.results_dict
    print(f"\nAvailable metrics keys:")
    for key in sorted(results.keys()):
        if 'seg' in key.lower() and ('map50' in key.lower() or 'class' in key.lower()):
            print(f"  {key}: {results[key]}")

print(f"\nOverall Segmentation mAP@50: {map50:.4f}")

import code
code.interact(local=dict(globals(), **locals()))


# Plot confusion matrix
plt.figure(figsize=(10, 8))
conf_mat_norm = conf_mat_d.astype('float') / (conf_mat_d.sum(axis=1)[:, np.newaxis] + 1e-6)  # normalize

# Get class names if available
class_names = metrics_d.names if hasattr(metrics_d, 'names') else [f'Class {i}' for i in range(conf_mat_d.shape[0])]

# Create heatmap
sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues',
           xticklabels=class_names,
           yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/confusion_matrix.png")
plt.show()