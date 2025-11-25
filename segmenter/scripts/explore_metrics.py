from ultralytics import YOLO
import numpy as np

# Test script to understand YOLO metrics structure
print("Testing YOLO metrics structure...")

# Load model
weights_file = "/home/alexanderjones/Alex/hpc-home/cgras_segmentation/train_genera/weights/best.pt"
model = YOLO(weights_file)

print(f"Model loaded: {weights_file}")
print(f"Model task: {model.task}")
print(f"Model names: {model.names}")

# Create a minimal dataset config for testing
test_config = """
path: /tmp/test_dataset  # Root directory (doesn't need to exist for this test)
train: images/train
val: images/val  
test: images/test

nc: 2  # Number of classes
names: ['alive_coral', 'dead_coral']  # Class names
"""

with open('/tmp/test_dataset.yaml', 'w') as f:
    f.write(test_config)

print("\nCreated test dataset config")
print("Attempting validation with test config...")

try:
    # Try validation - this might fail due to missing data, but we can still see the metrics structure
    metrics_d = model.val(data='/tmp/test_dataset.yaml', imgsz=640, conf=0.25, iou=0.7, plots=False, verbose=True)
    
    # Print all available attributes
    print("\nValidation completed! Available attributes on metrics_d:")
    for attr in dir(metrics_d):
        if not attr.startswith('_'):
            try:
                value = getattr(metrics_d, attr)
                print(f"  {attr}: {type(value)} - {value if not callable(value) else 'callable'}")
            except:
                print(f"  {attr}: <error accessing>")
    
    # Check for segmentation metrics
    if hasattr(metrics_d, 'seg'):
        print("\nSegmentation metrics found!")
        seg_metrics = metrics_d.seg
        print("Segmentation attributes:")
        for attr in dir(seg_metrics):
            if not attr.startswith('_'):
                try:
                    value = getattr(seg_metrics, attr)
                    print(f"  {attr}: {type(value)} - {value if not callable(value) else 'callable'}")
                except:
                    print(f"  {attr}: <error accessing>")

except Exception as e:
    print(f"Validation failed as expected: {e}")
    print("But we can still examine the model structure...")

# Let's also check what happens with model prediction
print(f"\nModel info:")
print(f"  Task: {getattr(model, 'task', 'unknown')}")
print(f"  Names: {getattr(model, 'names', 'unknown')}")

# Interactive mode for further exploration
print("\nEntering interactive mode...")
import code
code.interact(local=dict(globals(), **locals()))
