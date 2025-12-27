#!/bin/bash

# Base directory
BASE_DIR=$(dirname $(dirname $(readlink -f $0)))
cd $BASE_DIR

IMAGE_NAME="nudt-poisoning:latest"
METHOD=${1:-"STRIP"}
HOST_OUTPUT_DIR=${2:-"$(pwd)/output_docker"}
HOST_INPUT_DIR=${3:-"$(pwd)/input"}

mkdir -p "$HOST_OUTPUT_DIR"
mkdir -p "$HOST_INPUT_DIR"

echo "Running Defense in Docker: $METHOD"
docker run --rm \
    -v "$HOST_OUTPUT_DIR":/workspace/output \
    -v "$HOST_INPUT_DIR":/workspace/input \
    --network host \
    $IMAGE_NAME python3 main.py defense --method "$METHOD" --output_path /workspace/output --input_path /workspace/input

