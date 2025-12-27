#!/bin/bash

# Base directory
BASE_DIR=$(dirname $(dirname $(readlink -f $0)))
cd $BASE_DIR

METHOD=${1:-"BadNets"}
OUTPUT_DIR=${2:-"./output"}
INPUT_DIR=${3:-"./input"}

echo "Running Attack Simulation: $METHOD"
python3 main.py attack --method "$METHOD" --output_path "$OUTPUT_DIR" --input_path "$INPUT_DIR"
