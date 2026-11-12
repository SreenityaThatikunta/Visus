#!/usr/bin/env bash
# Push a Visus job to Kaggle.
#
#   KAGGLE_USERNAME=you kaggle/push.sh generate   # build the occlusion dataset (CPU, ~3.5 h)
#   KAGGLE_USERNAME=you kaggle/push.sh code       # upload this repo as the private dataset <you>/visus-code
#   KAGGLE_USERNAME=you kaggle/push.sh train      # sanity checks + training + evaluation (GPU)
#   KAGGLE_USERNAME=you kaggle/push.sh ablate     # analysis figures + ablations (GPU, after train)
#
# Needs the Kaggle CLI (pip install kaggle) and an API token in ~/.kaggle/.
# `train` reads the code from the visus-code dataset, so run `code` first after changing code.
set -euo pipefail
: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME to your Kaggle username}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

render() { sed "s/KAGGLE_USERNAME/$KAGGLE_USERNAME/g" "$1" > "$2"; }

case "${1:-}" in
  generate)
    render "$ROOT/kaggle/generate/kernel-metadata.json" "$TMP/kernel-metadata.json"
    cp "$ROOT/scripts/data/kaggle_generate_occlusions.py" "$TMP/"
    kaggle kernels push -p "$TMP" ;;
  code)
    tar -C "$ROOT" --exclude='__pycache__' -cf "$TMP/visus_code.tar" visus scripts requirements.txt
    printf '{"title": "visus-code", "id": "%s/visus-code", "licenses": [{"name": "other"}]}\n' \
      "$KAGGLE_USERNAME" > "$TMP/dataset-metadata.json"
    if kaggle datasets status "$KAGGLE_USERNAME/visus-code" >/dev/null 2>&1; then
      kaggle datasets version -p "$TMP" -m "code update $(git -C "$ROOT" rev-parse --short HEAD)"
    else
      kaggle datasets create -p "$TMP"
    fi ;;
  train)
    render "$ROOT/kaggle/train/kernel-metadata.json" "$TMP/kernel-metadata.json"
    cp "$ROOT/scripts/kaggle_train.py" "$TMP/"
    kaggle kernels push -p "$TMP" ;;
  ablate)
    render "$ROOT/kaggle/ablate/kernel-metadata.json" "$TMP/kernel-metadata.json"
    cp "$ROOT/scripts/kaggle_ablation.py" "$TMP/"
    kaggle kernels push -p "$TMP" ;;
  *)
    sed -n '2,11p' "$0"; exit 1 ;;
esac
