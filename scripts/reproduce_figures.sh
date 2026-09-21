#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
python3 source/generate_corrected_main_figures.py --pred data/MAIN_4MODEL_PREDICTIONS.npz --out reproduced_figures
