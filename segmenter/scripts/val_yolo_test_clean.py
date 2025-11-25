from ultralytics import YOLO
import numpy as np

# Define constants
CONF_THRESH = 0.25
IOU_THRESH = 0.7
PROJECT_DIR = "/home/alexanderjones/Alex/hpc-home/cgras_segmentation"
NAME = "val"
# Use COCO dataset for testing the metrics extraction functionality
data_file = "/home/alexanderjones/Alex/hpc-home/data/genera/cgras_data.yaml"  # This will use the built-in COCO dataset
weights_file = "/home/alexanderjones/Alex/hpc-home/cgras_segmentation/train_genera/weights/best.pt"

print(f"YOLO validation with conf_thresh={CONF_THRESH}, iou_thresh={IOU_THRESH}")
print(f"Data file: {data_file}")
print(f"Weights file: {weights_file}")

# Validation
# -----------------------------
model = YOLO(weights_file)
metrics_d = model.val(conf=CONF_THRESH, iou=IOU_THRESH, project=PROJECT_DIR, name=NAME, data=data_file, plots=True, split="test")

# Get confusion matrix
conf_mat_d = metrics_d.confusion_matrix.matrix  # Confusion matrix

# Print all available attributes on metrics_d
print("\nAll available attributes on metrics_d:")
for attr in dir(metrics_d):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Try to access segmentation metrics
if hasattr(metrics_d, 'seg'):
    print("\nFound segmentation metrics!")
    seg_metrics = metrics_d.seg
    
    # Print available attributes
    print("Segmentation metrics attributes:")
    for attr in dir(seg_metrics):
        if not attr.startswith('_'):
            print(f"  {attr}")
            
    # Extract values
    map50 = getattr(seg_metrics, 'map50', 0)
    precision = getattr(seg_metrics, 'p', None) or getattr(seg_metrics, 'precision', [])
    recall = getattr(seg_metrics, 'r', None) or getattr(seg_metrics, 'recall', [])
    f1 = getattr(seg_metrics, 'f1', [])
    
    # Convert to numpy if needed
    if precision is not None:
        precision = np.array(precision) if not isinstance(precision, (np.ndarray, float, int)) else precision
    if recall is not None:
        recall = np.array(recall) if not isinstance(recall, (np.ndarray, float, int)) else recall
    if f1 is not None:
        f1 = np.array(f1) if not isinstance(f1, (np.ndarray, float, int)) else f1
        
    # Get class names once
    class_names = {}
    if hasattr(metrics_d, 'names'):
        class_names = metrics_d.names
    elif hasattr(model, 'names'):
        class_names = model.names
    else:
        # Default fallback
        max_len = max(
            len(precision) if hasattr(precision, '__len__') else 0,
            len(recall) if hasattr(recall, '__len__') else 0,
            len(f1) if hasattr(f1, '__len__') else 0
        )
        class_names = {i: f'Class {i}' for i in range(max_len)}

    print(f"\nClass names: {class_names}")
    
    # Print the values with focus on alive coral
    print(f"\nSegmentation mAP@50: {map50:.4f}")

    # Handle precision, recall, and F1 which may be arrays
    if hasattr(precision, 'shape') and precision.size > 1:
        # If it's an array with multiple values (per class)
        print(f"Precision (mean): {np.mean(precision):.4f}")
        print("Precision per class:")
        for i, (class_index, class_name) in enumerate(class_names.items()):
            if i < len(precision):
                p_value = float(precision[i])
                print(f"  {class_name}: {p_value:.4f}")
                if class_name.lower() == 'alive_coral':
                    print(f"  *** ALIVE CORAL Precision: {p_value:.4f} ***")
    else:
        # If it's a single value
        print(f"Precision: {float(precision):.4f}")

    # Do the same for recall
    if hasattr(recall, 'shape') and recall.size > 1:
        print(f"Recall (mean): {np.mean(recall):.4f}")
        print("Recall per class:")
        for i, (class_index, class_name) in enumerate(class_names.items()):
            if i < len(recall):
                r_value = float(recall[i])
                print(f"  {class_name}: {r_value:.4f}")
                if class_name.lower() == 'alive_coral':
                    print(f"  *** ALIVE CORAL Recall: {r_value:.4f} ***")
    else:
        print(f"Recall: {float(recall):.4f}")

    # And for F1
    if hasattr(f1, 'shape') and f1.size > 1:
        print(f"F1 Score (mean): {np.mean(f1):.4f}")
        print("F1 Score per class:")
        for i, (class_index, class_name) in enumerate(class_names.items()):
            if i < len(f1):
                f1_value = float(f1[i])
                print(f"  {class_name}: {f1_value:.4f}")
                if class_name.lower() == 'alive_coral':
                    print(f"  *** ALIVE CORAL F1: {f1_value:.4f} ***")
    else:
        print(f"F1 Score: {float(f1):.4f}")

    # Try to get per-class mAP50 values if available
    print("\nAttempting to get per-class mAP50 values:")
    
    # Method 1: Check for map_per_class attribute
    if hasattr(seg_metrics, 'map_per_class'):
        map_per_class = seg_metrics.map_per_class
        print(f"Found map_per_class: {map_per_class}")
        if map_per_class is not None and len(map_per_class) > 0:
            for i, (class_index, class_name) in enumerate(class_names.items()):
                if i < len(map_per_class):
                    class_map50 = float(map_per_class[i])
                    print(f"  {class_name} mAP@50: {class_map50:.4f}")
                    if class_name.lower() == 'alive_coral':
                        print(f"  *** ALIVE CORAL mAP@50: {class_map50:.4f} ***")
    
    # Method 2: Check for ap_class_index
    if hasattr(seg_metrics, 'ap_class_index'):
        ap_class_index = seg_metrics.ap_class_index
        print(f"Found ap_class_index: {ap_class_index}")
    
    # Method 3: Check for ap attribute
    if hasattr(seg_metrics, 'ap'):
        ap = seg_metrics.ap
        print(f"Found ap attribute: shape={ap.shape if hasattr(ap, 'shape') else 'no shape'}")
        if hasattr(ap, 'shape') and ap.size > 0:
            # YOLO ap is typically [num_classes, num_iou_thresholds]
            # mAP@50 is the first IoU threshold (0.5)
            map50_per_class = ap[:, 0] if ap.shape[1] > 0 else ap
            print(f"Per-class mAP@50 from ap[:, 0]: {map50_per_class}")
            
            for i, (class_index, class_name) in enumerate(class_names.items()):
                if i < len(map50_per_class):
                    class_map50 = float(map50_per_class[i])
                    print(f"  {class_name} mAP@50: {class_map50:.4f}")
                    if class_name.lower() == 'alive_coral':
                        print(f"  *** ALIVE CORAL mAP@50: {class_map50:.4f} ***")

else:
    print("No segmentation metrics found!")
    # Fallback to box metrics if available
    if hasattr(metrics_d, 'box'):
        map50 = metrics_d.box.map50
        print(f"Detection mAP@50: {map50:.4f}")
    elif hasattr(metrics_d, 'results_dict'):
        map50 = metrics_d.results_dict.get('metrics/mAP50(B)', 0)
        print(f"Detection mAP@50: {map50:.4f}")

# Interactive debugging
print("\nEntering interactive mode for further exploration...")
import code
code.interact(local=dict(globals(), **locals()))
