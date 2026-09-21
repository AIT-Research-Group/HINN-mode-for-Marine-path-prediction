#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
python3 source/verify_saved_results.py
