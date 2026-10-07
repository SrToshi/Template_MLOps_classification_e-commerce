#!/usr/bin/env bash
# Initialise DVC and start tracking the Rakuten datasets.
#
# Run once, from the repository root:
#     bash scripts/setup_dvc.sh            # CSVs only (fast)
#     bash scripts/setup_dvc.sh --images   # CSVs + the 84,916 training images
#
# On Windows, run the same commands by hand in cmd — they are identical
# apart from the shell syntax.
#
# `dvc init --no-scm` is deliberate: the brief asks for DVC *without* Git, so
# the .dvc pointer files are themselves the record of which data version was
# used, rather than a Git commit. training.py reads those pointers and tags
# each MLflow run with the hashes (see src/data_version.py), which is what
# makes a run reproducible: parameters from MLflow, data from DVC.
#
# Note on disk: `dvc add` copies tracked data into .dvc/cache, so adding the
# images costs roughly another 2.4 GB. That is the price of being able to
# restore an exact dataset version, and it is why the images are opt-in here.

set -eu

TRACK_IMAGES=0
[ "${1:-}" = "--images" ] && TRACK_IMAGES=1

if ! command -v dvc >/dev/null 2>&1; then
    echo "DVC is not installed. In its own environment, to keep it away from"
    echo "the pinned TensorFlow stack:"
    echo
    echo "    conda create -n rakuten-dvc python=3.10 -y"
    echo "    conda activate rakuten-dvc"
    echo "    pip install 'dvc==3.55.2'"
    exit 1
fi

if [ ! -d .dvc ]; then
    echo "==> dvc init --no-scm"
    dvc init --no-scm
fi

echo "==> tracking the tabular data"
dvc add data/preprocessed/X_train_update.csv
dvc add data/preprocessed/Y_train_CVw08PX.csv
dvc add data/preprocessed/X_test_update.csv

if [ "$TRACK_IMAGES" = "1" ]; then
    echo "==> tracking the training images (slow: ~85k files)"
    dvc add data/preprocessed/image_train
else
    echo "==> skipping images (pass --images to include them)"
fi

echo
echo "Done. The .dvc pointer files now identify this data version:"
ls -1 data/preprocessed/*.dvc 2>/dev/null || true
echo
echo "The next training run will tag its MLflow run with these hashes."
