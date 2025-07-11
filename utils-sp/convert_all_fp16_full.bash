#!/bin/bash

DATA_DIR="/lustre/scratch5/exempt/artimis/data"
MPMM_DATA_DIR="lsc240420_fp16_full"

INPUT_DIR="${DATA_DIR}/${MPMM_DATA_DIR}"
OUTPUT_DIR="${INPUT_DIR}_HDF5"

TIMEOUT=10  # seconds

echo -e "\nINPUT DIR: ${INPUT_DIR}"
echo -e "\nOUTPUT DIR: ${OUTPUT_DIR}"

if [ ! -d ${OUTPU_DIR} ]; then
    mkdir -p "$OUTPUT_DIR"
fi

for npz_file in "$INPUT_DIR"/*.npz; do
    filename=$(basename "$npz_file")

    h5_file="${OUTPUT_DIR}/${filename%.npz}.h5"
    echo -e "\n🔄 Processing $filename"

    #timeout "$TIMEOUT" python3 convert_one.py "$npz_file" "$h5_file"

    case $? in
        0) echo "✅ Done $filename" ;;
        124) echo "⏱️ Timeout: Skipped $filename after ${TIMEOUT}s" ;;
        *) echo "❌ Error converting $filename" ;;
    esac
done

