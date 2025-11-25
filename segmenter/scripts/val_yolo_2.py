#! /usr/bin/env python3
""" val_segmenter.py
 Validate a YOLO model against a dataset. Computes True Positives (TP), False Positives (FP),
 False Negatives (FN), True Negatives (TN), precision, recall, and F1 scores.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
import sys
import argparse
import yaml
import os
# -----------------------------
# DEFAULT CONFIGURATION: MODIFY AS NEEDED
# -----------------------------
PROJECT_DIR = "cgras_segmentation"
NAME = "full_validation"
CONF_THRESH = 0.5
IOU_THRESH = 0.5
IGNORE_CLASSES = None
WEIGHTS_FILE = "cgras_segmentation/train_genera/weights/best.pt"
DATA_FILE = "data/genera/cgras_data.yaml"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate YOLO model with YAML configuration.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


# -----------------------------
# MODEL INITIALIZATION & VALIDATION
# -----------------------------

def validate(conf_thresh, iou_thresh, project_dir, name, weights_files, data_files, ignore_classes):
    """
    Validate multiple model/dataset combinations and collect mAP50 results.
    
    Args:
        weights_files: List of paths to model weight files
        data_files: List of paths to data YAML files
        Other parameters remain the same
    """
    results = []
    
    # Ensure weights and data inputs are lists
    if isinstance(weights_files, str):
        weights_files = [weights_files]
    if isinstance(data_files, str):
        data_files = [data_files]
    print(f"Validating {len(weights_files)} models against {len(data_files)} datasets...")

    # Run validation on every combination
    for i, weights_file in enumerate(weights_files):
        for j, data_file in enumerate(data_files):
            
            # For weights: get the parent directory name (e.g., train_pdae, train_amag)
            weights_parent = os.path.basename(os.path.dirname(os.path.dirname(weights_file)))
            model_identifier = f"{weights_parent}"
            
            # For data: get the parent directory name (e.g., pdae, amag, genera)  
            data_parent = os.path.basename(os.path.dirname(data_file))
            dataset_identifier = f"{data_parent}"
            
            # Create unique combination identifier
            combo_name = f"{name}_{model_identifier}_on_{dataset_identifier}"
            
            print(f"\n[{i+1}/{len(weights_files)}][{j+1}/{len(data_files)}] Testing combination:")
            print(f"  Model: {weights_file} -> {model_identifier}")
            print(f"  Dataset: {data_file} -> {dataset_identifier}")
            print(f"  Combo ID: {combo_name}")
            print("-" * 60)
            
            try:
                # Initialize model for this combination
                model = YOLO(weights_file)
                
                # Run validation with ignore_classes parameter
                val_params = {
                    'conf': conf_thresh,
                    'iou': iou_thresh,
                    'project': project_dir,
                    'name': combo_name,
                    'data': data_file,
                    'plots': True,
                    'split': "test"
                }
                
                # Add ignore_classes if specified
                if ignore_classes is not None:
                    # Convert class names to indices if needed
                    if isinstance(ignore_classes[0], str):
                        # Load the data YAML to get class names
                        with open(data_file, 'r') as f:
                            data_config = yaml.safe_load(f)
                        class_names = data_config.get('names', [])
                        ignore_indices = [class_names.index(cls) for cls in ignore_classes if cls in class_names]
                        val_params['classes'] = [i for i in range(len(class_names)) if i not in ignore_indices]
                    else:
                        # Already indices
                        with open(data_file, 'r') as f:
                            data_config = yaml.safe_load(f)
                        total_classes = len(data_config.get('names', []))
                        val_params['classes'] = [i for i in range(total_classes) if i not in ignore_classes]
                
                metrics_d = model.val(**val_params)
                
                # Extract metrics
                map50 = float(metrics_d.seg.map50)
                precision = metrics_d.seg.p
                recall = metrics_d.seg.r
                f1 = metrics_d.seg.f1
                import code
                code.interact(local=dict(globals(), **locals()))
                # Extract per-class map50
                map50_per_class = metrics_d.seg.map50_per_class
                class_names = []

               

                # Load class names from data file
                with open(data_file, 'r') as f:
                    data_config = yaml.safe_load(f)
                    class_names = data_config.get('names', [])
                
                # Store results
                result = {
                    'model_path': weights_file,
                    'model_name': model_identifier,
                    'data_path': data_file, 
                    'dataset_name': dataset_identifier,
                    'combination_name': combo_name,
                    'map50': map50,
                    'conf_thresh': conf_thresh,
                    'iou_thresh': iou_thresh,
                    'status': 'Success'
                }

                if map50_per_class is not None and len(map50_per_class) > 0:
                    # Add individual class mAP50 values
                    for i, class_name in enumerate(class_names):
                        if i < len(map50_per_class):
                            result[f'map50_{class_name}'] = float(map50_per_class[i])
                    
                    # Add specific alive coral mAP50 for easy access
                    if 'alive_coral' in class_names:
                        alive_idx = class_names.index('alive_coral')
                        if alive_idx < len(map50_per_class):
                            result['map50_alive_coral'] = float(map50_per_class[alive_idx])
                    
                    # Add specific dead coral mAP50 for completeness
                    if 'dead_coral' in class_names:
                        dead_idx = class_names.index('dead_coral')
                        if dead_idx < len(map50_per_class):
                            result['map50_dead_coral'] = float(map50_per_class[dead_idx])

                
                # Add precision, recall, F1 (handle both single values and arrays)
                if hasattr(precision, 'shape') and precision.size > 1:
                    result['precision_mean'] = float(np.mean(precision))
                    result['precision_per_class'] = precision.tolist()
                    for i, class_name in enumerate(class_names):
                        if i < len(precision):
                            result[f'precision_{class_name}'] = float(precision[i])
                else:
                    result['precision_mean'] = float(precision)
                    
                if hasattr(recall, 'shape') and recall.size > 1:
                    result['recall_mean'] = float(np.mean(recall))
                    result['recall_per_class'] = recall.tolist()
                    for i, class_name in enumerate(class_names):
                        if i < len(recall):
                            result[f'recall_{class_name}'] = float(recall[i])
                else:
                    result['recall_mean'] = float(recall)
                    
                if hasattr(f1, 'shape') and f1.size > 1:
                    result['f1_mean'] = float(np.mean(f1))
                    result['f1_per_class'] = f1.tolist()
                    for i, class_name in enumerate(class_names):
                        if i < len(f1):
                            result[f'f1_{class_name}'] = float(f1[i])
                else:
                    result['f1_mean'] = float(f1)
                
                results.append(result)
                
                print(f"✓ SUCCESS: mAP@50 = {map50:.4f}")
                if map50_per_class is not None and len(class_names) > 0:
                    print("  Per-class mAP@50:")
                    for i, class_name in enumerate(class_names):
                        if i < len(map50_per_class):
                            print(f"    {class_name}: {map50_per_class[i]:.4f}")
                print(f"  Precision (mean): {result['precision_mean']:.4f}")
                print(f"  Recall (mean): {result['recall_mean']:.4f}")
                print(f"  F1 Score (mean): {result['f1_mean']:.4f}")
                
                # Plot confusion matrix for this combination
                #plot_confusion_matrix(metrics_d, combo_name, project_dir)
                
            except Exception as e:
                print(f"✗ FAILED: {str(e)}")
                
                # Still record the failed attempt
                result = {
                    'model_path': weights_file,
                    'model_name': model_identifier,
                    'data_path': data_file,
                    'dataset_name': dataset_identifier,
                    'combination_name': combo_name,
                    'map50': 0.0,
                    'conf_thresh': conf_thresh,
                    'iou_thresh': iou_thresh,
                    'status': f'Failed: {str(e)}',
                    'precision_mean': 0.0,
                    'recall_mean': 0.0,
                    'f1_mean': 0.0
                }
                results.append(result)
    
    # Generate comprehensive results summary
    generate_results_summary(results, project_dir, name)
    
    return results

def plot_confusion_matrix(metrics_d, combo_name, project_dir):
    """Plot and save confusion matrix for a specific model/dataset combination"""
    try:
        conf_mat_d = metrics_d.confusion_matrix.matrix
        conf_mat_norm = conf_mat_d.astype('float') / (conf_mat_d.sum(axis=1)[:, np.newaxis] + 1e-6)
        
        class_names = metrics_d.names if hasattr(metrics_d, 'names') else [f'Class {i}' for i in range(conf_mat_d.shape[0])]
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
        
        # Dynamic text color
        for i, text in enumerate(ax.texts):
            value = float(text.get_text())
            if value > 0.5:
                text.set_color('white')
                text.set_weight('bold')
            else:
                text.set_color('black')
                text.set_weight('normal')
        
        plt.xlabel('Predicted', fontsize=14, fontweight='bold')
        plt.ylabel('True', fontsize=14, fontweight='bold')
        plt.title(f'Confusion Matrix - {combo_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save with unique name
        output_path = os.path.join(project_dir, f"confusion_matrix_{combo_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to prevent memory issues with multiple plots
        
        print(f"  Confusion matrix saved: {output_path}")
        
    except Exception as e:
        print(f"  Warning: Could not generate confusion matrix: {e}")

def generate_results_summary(results, project_dir, name):
    """Generate comprehensive summary of all validation results"""
    import pandas as pd
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save detailed results to CSV
    csv_path = os.path.join(project_dir, f"validation_results_{name}.csv")
    df.to_csv(csv_path, index=False)
    
    # Create summary table with key metrics
    summary_columns = ['model_name', 'dataset_name', 'map50', 'precision_mean', 'recall_mean', 'f1_mean', 'status']
    
    # Add alive coral specific columns if they exist
    if 'map50_alive_coral' in df.columns:
        summary_columns.insert(3, 'map50_alive_coral')  # Insert after overall map50
    if 'precision_alive_coral' in df.columns:
        summary_columns.insert(-2, 'precision_alive_coral')  # Before status
    if 'recall_alive_coral' in df.columns:
        summary_columns.insert(-2, 'recall_alive_coral')
    if 'f1_alive_coral' in df.columns:
        summary_columns.insert(-2, 'f1_alive_coral')
    
    # Create summary with available columns
    available_columns = [col for col in summary_columns if col in df.columns]
    summary_df = df[available_columns].copy()
    summary_df = summary_df.sort_values('map50', ascending=False)
    
    # Save summary
    summary_path = os.path.join(project_dir, f"validation_summary_{name}.csv") 
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False, float_format='{:.4f}'.format))
    
    
    # Create alive-coral-only summary if data exists
    if 'map50_alive_coral' in df.columns:
        alive_summary = df[['model_name', 'dataset_name', 'map50_alive_coral', 
                           'precision_alive_coral', 'recall_alive_coral', 'f1_alive_coral', 'status']].copy()
        alive_summary = alive_summary.sort_values('map50_alive_coral', ascending=False)
        
        alive_path = os.path.join(project_dir, f"validation_alive_coral_only_{name}.csv")
        alive_summary.to_csv(alive_path, index=False)
        
        print(f"\n{'='*60}")
        print("ALIVE CORAL ONLY RESULTS")
        print(f"{'='*60}")
        print(alive_summary.to_string(index=False, float_format='{:.4f}'.format))
        
        print(f"\n📊 Results saved to:")
        print(f"   All classes (detailed): {csv_path}")
        print(f"   Summary (all classes): {summary_path}")
        print(f"   Alive coral only: {alive_path}")
    else:
        print(f"\n📊 Results saved to:")
        print(f"   Detailed: {csv_path}")
        print(f"   Summary: {summary_path}")

def main():
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Run validation
    validate(
        conf_thresh=config.get('CONF_THRESH', CONF_THRESH),
        iou_thresh=config.get('IOU_THRESH', IOU_THRESH),
        project_dir=config.get('PROJECT_DIR', PROJECT_DIR),
        name=config.get('NAME', NAME),
        weights_files=config.get('WEIGHTS', [WEIGHTS_FILE]),
        data_files=config.get('DATA', [DATA_FILE]),
        ignore_classes=config.get('IGNORE_CLASSES', IGNORE_CLASSES)
    )



if __name__ == "__main__":
    main()
