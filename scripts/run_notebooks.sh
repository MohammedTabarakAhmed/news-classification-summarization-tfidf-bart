#!/bin/bash
# Handy script to run notebooks end-to-end (replace names as needed)
set -e
NOTEBOOKS=("notebook/EDA STUDENT PERFORMANCE.ipynb" "notebook/MODEL TRAINING.ipynb")
for nb in "${NOTEBOOKS[@]}"; do
  if [ -f "$nb" ]; then
    jupyter nbconvert --to notebook --execute "$nb" --inplace
  else
    echo "Notebook $nb not found, skipping"
  fi
done
