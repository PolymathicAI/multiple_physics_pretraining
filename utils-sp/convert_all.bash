#!/bin/bash

INPUT_DIR="/lustre/scratch5/exempt/artimis/data/lsc240420"
OUTPUT_DIR="/lustre/scratch5/exempt/artimis/data/lsc240420_hdf5"
TIMEOUT=10  # seconds

mkdir -p "$OUTPUT_DIR"

for npz_file in "$INPUT_DIR"/*.npz; do
    filename=$(basename "$npz_file")

    h5_file="${OUTPUT_DIR}/${filename%.npz}.h5"
    echo -e "\n🔄 Processing $filename"

    timeout "$TIMEOUT" python3 convert_one.py "$npz_file" "$h5_file"

    case $? in
        0) echo "✅ Done $filename" ;;
        124) echo "⏱️ Timeout: Skipped $filename after ${TIMEOUT}s" ;;
        *) echo "❌ Error converting $filename" ;;
    esac
done

