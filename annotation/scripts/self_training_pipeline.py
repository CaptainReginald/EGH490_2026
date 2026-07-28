#!/usr/bin/env python3
"""
Self-training pipeline orchestration for semi-supervised segmentation.

Steps:
 1. (Optional) Train teacher model using segmenter/scripts/train.py and a teacher config YAML
 2. Run annotation/scripts/predict_pipeline.py on unlabeled images using the teacher weights to create pseudo-label COCO
 3. Convert COCO pseudo-labels into YOLO format (labels/.txt) using ultralytics converter (fallback to helper)
 4. Merge labeled dataset (YOLO format) with pseudo-labeled dataset into a combined YOLO dataset folder
 5. Create a combined cgras_data.yaml and then run student training using segmenter/scripts/train.py with --yaml-path pointing to combined dataset

This script re-uses existing repo utilities (predict_pipeline, combine_coco, ultralytics converter) where possible and keeps outputs under a specified workdir.

Usage example:
  python self_training_pipeline.py --teacher-config segmenter/config/genera_model.yaml \
    --student-config segmenter/config/genera_model.yaml \
    --unlabeled-dir /path/to/unlabeled/images \
    --labeled-dataset-yaml /path/to/labeled/cgras_data.yaml \
    --workdir /path/to/workdir --skip-teacher-train

"""

import argparse
import subprocess
import sys
import os
import shutil
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("self_training")


def run_command(cmd, check=True):
    logger.info("Running: %s", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logger.info(res.stdout)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nstdout:\n{res.stdout}")
    return res


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def write_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def convert_coco_to_yolo(coco_json_path, images_dir, out_dir):
    """Convert COCO annotations to YOLO format using ultralytics converter.
    Falls back to using the repository helper if import fails.
    The converter expects an annotations folder with instances_default.json; place coco_json_path accordingly.
    """
    annotations_dir = Path(out_dir) / 'coco_annotations'
    annotations_dir.mkdir(parents=True, exist_ok=True)
    target_json = annotations_dir / 'instances_default.json'
    shutil.copy2(coco_json_path, target_json)

    # Create a temp folder expected by converter: save_dir will contain /images and /labels
    save_dir = Path(out_dir) / 'converted'
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try to use ultralytics converter
        from ultralytics.data.converter import convert_coco

        logger.info("Converting COCO to YOLO using ultralytics.convert_coco")
        convert_coco(labels_dir=str(annotations_dir), save_dir=str(save_dir), use_segments=True, use_keypoints=False, cls91to80=False)

        # After conversion, the converter writes coco/labels/default/*.txt; move into save_dir/labels and images
        # The helper above may already create images structure; we will assume converted labels are under save_dir/labels/default
        return save_dir
    except Exception as e:
        logger.warning("ultralytics.convert_coco not available or failed: %s", e)
        # Fallback: attempt to import repo helper if present
        try:
            helper = Path(__file__).resolve().parents[1] / 'general_scripts' / 'cvatcoco_to_yolo.py'
            if helper.exists():
                logger.info("Falling back to general_scripts/cvatcoco_to_yolo.py helper")
                # Execute the helper in a subprocess - it is interactive; run in non-interactive mode by setting convert=True via env
                # The helper is not ideal for programmatic use, so raise error to prompt user to install ultralytics
                raise RuntimeError("Fallback converter exists but is interactive; please install ultralytics in the environment to enable automatic conversion.")
            else:
                raise RuntimeError("No fallback converter available; install ultralytics package to enable COCO->YOLO conversion.")
        except Exception:
            raise


def merge_yolo_datasets(base_yolo_path, pseudo_converted_path, combined_out):
    """Merge two YOLO-style datasets (base and pseudo) into combined_out.
    Both inputs are expected to contain 'images' and 'labels' directories (or base may be structured with data/images labels).
    """
    combined_out = Path(combined_out)
    images_out = combined_out / 'data' / 'images'
    labels_out = combined_out / 'data' / 'labels'
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    def copy_contents(src, dest):
        src = Path(src)
        if not src.exists():
            logger.warning("Source path %s does not exist, skipping", src)
            return
        for p in src.iterdir():
            if p.is_file():
                shutil.copy2(p, dest / p.name)

    # Support multiple possible base layouts
    base_candidates = [
        Path(base_yolo_path) / 'data' / 'images',
        Path(base_yolo_path) / 'images',
        Path(base_yolo_path) / 'data' / 'labels',
        Path(base_yolo_path) / 'labels',
    ]

    # Copy base images and labels
    copied_any = False
    if Path(base_yolo_path).is_dir():
        # If base contains data/images, prefer that
        for candidate in [Path(base_yolo_path) / 'data' / 'images', Path(base_yolo_path) / 'images', Path(base_yolo_path) / 'data' / 'labels', Path(base_yolo_path) / 'labels']:
            if candidate.exists():
                if 'images' in str(candidate):
                    copy_contents(candidate, images_out)
                    copied_any = True
                elif 'labels' in str(candidate):
                    copy_contents(candidate, labels_out)
                    copied_any = True

    if not copied_any:
        logger.warning("No base YOLO images/labels found in %s; combined dataset will contain only pseudo-labeled images", base_yolo_path)

    # For pseudo_converted_path, ultralytics converter writes into converted/images and converted/labels
    pseudo_images = Path(pseudo_converted_path) / 'images'
    pseudo_labels = Path(pseudo_converted_path) / 'labels'

    # Some converter outputs to converted/images and converted/labels; if nested, try 'coco/labels/default'
    if not pseudo_images.exists():
        # try nested patterns
        for cand in Path(pseudo_converted_path).rglob('images'):
            if cand.is_dir():
                pseudo_images = cand
                break
    if not pseudo_labels.exists():
        for cand in Path(pseudo_converted_path).rglob('labels'):
            if cand.is_dir():
                pseudo_labels = cand
                break

    copy_contents(pseudo_images, images_out)
    copy_contents(pseudo_labels, labels_out)

    # Ensure every image has a label file (create blank label if missing)
    for img in images_out.iterdir():
        if img.is_file():
            label_name = img.stem + '.txt'
            label_path = labels_out / label_name
            if not label_path.exists():
                label_path.write_text('')

    logger.info("Merged datasets into %s", combined_out)
    return combined_out


def create_cgras_data_yaml(names_map, combined_out, yaml_out_path):
    cgras = {
        'names': names_map,
        'path': str(Path(combined_out).absolute()),
        'data': ['data/images']
    }
    write_yaml(cgras, yaml_out_path)
    logger.info("Wrote combined cgras_data.yaml to %s", yaml_out_path)
    return yaml_out_path


def main():
    parser = argparse.ArgumentParser(description="Self-training pipeline orchestrator")
    parser.add_argument('--teacher-config', required=True, help='Path to teacher training config YAML (uses segmenter/scripts/train.py)')
    parser.add_argument('--student-config', required=True, help='Path to student training config YAML')
    parser.add_argument('--labeled-dataset-yaml', required=True, help='Path to existing labeled cgras_data.yaml (YOLO-style)')
    parser.add_argument('--unlabeled-dir', required=True, help='Directory of unlabeled images to generate pseudo labels for')
    parser.add_argument('--workdir', required=True, help='Working directory to store intermediate outputs')
    parser.add_argument('--skip-teacher-train', action='store_true', help='Skip teacher training (assume teacher weights present in teacher-config model_path)')
    parser.add_argument('--teacher-weights', default=None, help='Optional path to teacher weights to override teacher-config model_path')
    parser.add_argument('--sahi-config', default=None, help='Optional predict_pipeline YAML config to use for SAHI (overrides defaults)')
    parser.add_argument('--conf-thresh', type=float, default=0.4, help='Confidence threshold for SAHI predictions')

    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Step 0: load configs
    teacher_cfg = load_yaml(args.teacher_config)
    student_cfg = load_yaml(args.student_config)
    labeled_yaml = load_yaml(args.labeled_dataset_yaml)

    # Optionally override teacher model path
    if args.teacher_weights:
        teacher_cfg['model_path'] = args.teacher_weights

    # Step 1: Train teacher (if not skipped)
    if not args.skip_teacher_train:
        logger.info("Starting teacher training using config %s", args.teacher_config)
        run_command([sys.executable, 'segmenter/scripts/train.py', '--config', args.teacher_config])
        logger.info("Teacher training finished")
    else:
        logger.info("Skipping teacher train; using weights from teacher-config or --teacher-weights")

    # Determine teacher weights path
    teacher_weights = teacher_cfg.get('model_path')
    if not teacher_weights or not Path(teacher_weights).exists():
        logger.warning("Teacher weights not found at %s; continuing but SAHI inference may fail", teacher_weights)

    # Step 2: Run predict_pipeline to generate pseudo-label COCO
    predict_cfg_path = None
    if args.sahi_config:
        predict_cfg_path = args.sahi_config
    else:
        # Create a minimal temporary predict config
        predict_cfg = {
            'sahi_predict': {
                'data_dir': str(args.unlabeled_dir),
                'output_dir': str(workdir),
                'model_path': teacher_weights,
                'name': 'selftrain_pseudo',
                'slice_width': 640,
                'slice_height': 640,
                'overlap': 0.5,
                'conf_thresh': args.conf_thresh,
                'device': teacher_cfg.get('device', 0)
            },
            'pipeline': {
                'steps': ['sahi_predict', 'fix_coco']
            },
            'fix_coco': {
                # leave input_file blank so predict_pipeline uses sahi output
                'output_file': None
            }
        }
        predict_cfg_path = workdir / 'selftrain_predict_config.yaml'
        write_yaml(predict_cfg, predict_cfg_path)
        predict_cfg_path = str(predict_cfg_path)

    logger.info("Running predict_pipeline with config %s", predict_cfg_path)
    run_command([sys.executable, 'annotation/scripts/predict_pipeline.py', '--config', str(predict_cfg_path)])

    # predict_pipeline prints outputs but we need to locate the fixed COCO path
    # Convention: run_fix_coco returns path; the predict step produces an annotations folder under workdir name like *_annotations_*/annotations/instances_default.json
    # Search workdir for files named fixed_*.json or instances_default.json
    coco_json = None
    for p in workdir.rglob('fixed_*.json'):
        coco_json = p
        break
    if coco_json is None:
        # Search for instances_default.json created by SAHI
        for p in workdir.rglob('instances_default.json'):
            coco_json = p
            break
    if coco_json is None:
        # Also check for any .json under workdir
        jsons = list(workdir.rglob('*.json'))
        if jsons:
            coco_json = jsons[0]

    if coco_json is None:
        raise RuntimeError("Could not locate SAHI/COCO output JSON under workdir")

    coco_json = str(coco_json)
    logger.info("Found COCO pseudo-labels at %s", coco_json)

    # Step 3: Convert COCO to YOLO labels
    converted_out = Path(workdir) / 'pseudo_converted'
    converted_out.mkdir(parents=True, exist_ok=True)
    convert_coco_to_yolo(coco_json, args.unlabeled_dir, str(converted_out))

    # Step 4: Merge labeled dataset with pseudo-labeled converted dataset
    combined_out = Path(workdir) / 'combined_dataset'
    merged = merge_yolo_datasets(args.labeled_dataset_yaml and Path(args.labeled_dataset_yaml).parent or args.labeled_dataset_yaml, converted_out, combined_out)

    # Step 5: Create combined cgras_data.yaml and run student training
    # Use names from labeled dataset if available
    names_map = labeled_yaml.get('names', {})
    combined_yaml = Path(workdir) / 'cgras_data_combined.yaml'
    create_cgras_data_yaml(names_map, combined_out, combined_yaml)

    # Run student training using existing train.py and override with --yaml-path
    logger.info("Starting student training using config %s and dataset %s", args.student_config, combined_yaml)
    run_command([sys.executable, 'segmenter/scripts/train.py', '--config', args.student_config, '--yaml-path', str(combined_yaml)])
    logger.info("Student training finished")


if __name__ == '__main__':
    main()
