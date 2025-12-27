#!/bin/bash

# Base directory
BASE_DIR=$(dirname $(readlink -f $0))
cd $BASE_DIR

IMAGE_NAME="nudt-poisoning:latest"
HOST_OUTPUT_DIR="$(pwd)/output_docker"
HOST_INPUT_DIR="$(pwd)/input"

mkdir -p "$HOST_OUTPUT_DIR"
mkdir -p "$HOST_INPUT_DIR"

# Specify the method directly here
METHOD="STRIP"

echo "----------------------------------------"
echo "Running Defense in Docker: $METHOD"
echo "----------------------------------------"

# Displaying the command in EOFSCRIPT style
cat << EOFSCRIPT
docker run --rm \\
    -v "$(pwd)":/workspace \\
    -v "$HOST_OUTPUT_DIR":/workspace/output \\
    -v "$HOST_INPUT_DIR":/workspace/input \\
    -e mode=defense \\
    -e method="$METHOD" \\
    -e OUTPUT_PATH=/workspace/output \\
    -e INPUT_PATH=/workspace/input \\
    --network host \\
    $IMAGE_NAME
EOFSCRIPT

# Executing the command
docker run --rm \
    -v "$(pwd)":/workspace \
    -v "$HOST_OUTPUT_DIR":/workspace/output \
    -v "$HOST_INPUT_DIR":/workspace/input \
    -e mode=defense \
    -e method="$METHOD" \
    -e OUTPUT_PATH=/workspace/output \
    -e INPUT_PATH=/workspace/input \
    --network host \
    $IMAGE_NAME
