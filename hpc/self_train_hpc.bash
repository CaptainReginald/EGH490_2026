#!/bin/bash -l

#PBS -N selftrain_yolov8_seg
#PBS -l select=1:ncpus=8:ngpus=1:mem=64GB:gpu_id=A100
#PBS -l walltime=36:00:00
#PBS -m abe
#PBS -j oe

set -euo pipefail

cd "$PBS_O_WORKDIR"
pwd

REPO_ROOT="Corals/EGH490_2026"
TEACHER_CONFIG="${TEACHER_CONFIG:-${REPO_ROOT}/segmenter/config/2024_amag_model.yaml}"
STUDENT_CONFIG="${STUDENT_CONFIG:-${REPO_ROOT}/segmenter/config/2024_amag_model.yaml}"
LABELED_YAML="${LABELED_YAML:-${REPO_ROOT}/data/genera/cgras_data.yaml}"
UNLABELED_DIR="${UNLABELED_DIR:-${REPO_ROOT}/data/unlabeled}"
WORKDIR="${WORKDIR:-${REPO_ROOT}/analysis/selftrain_workdir}"

if [ -f /home/wardlewo/miniforge3/bin/activate ]; then
  source /home/wardlewo/miniforge3/bin/activate cgras
elif [ -f /home/wardlewo/mambaforge/bin/activate ]; then
  source /home/wardlewo/mambaforge/bin/activate cgras
else
  echo "Could not find conda activate script for user wardlewo"
  exit 1
fi

CMD=(
  python -u ${REPO_ROOT}/annotation/scripts/self_training_pipeline.py
  --teacher-config "$TEACHER_CONFIG"
  --student-config "$STUDENT_CONFIG"
  --labeled-dataset-yaml "$LABELED_YAML"
  --unlabeled-dir "$UNLABELED_DIR"
  --workdir "$WORKDIR"
)

"${CMD[@]}"

conda deactivate

echo "self-training job done"
