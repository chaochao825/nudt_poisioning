#!/bin/bash

# Base directory
BASE_DIR=$(dirname $(dirname $(readlink -f $0)))
cd $BASE_DIR

IMAGE_NAME="nudt-poisoning:latest"

echo "Building Docker image..."
docker build --network=host -t $IMAGE_NAME .

# Output directory on host
HOST_OUTPUT_DIR="$(pwd)/output_docker"
mkdir -p $HOST_OUTPUT_DIR

echo "Starting Docker Tests for All Attacks..."
ATTACKS=("BadNets" "CleanLabel" "FeatureCollision" "GradientShift" "LabelFlip" "ModelPoisoning" "NeuronInterference" "PhysicalBackdoor" "RandomNoise" "SampleMix" "TriggerlessDynamicBackdoor" "Trojan")

for attack in "${ATTACKS[@]}"; do
    echo "Testing Attack in Docker: $attack"
    docker run --rm -v "$HOST_OUTPUT_DIR":/workspace/output $IMAGE_NAME python3 main.py attack --method "$attack" --output_path /workspace/output
    echo "------------------------------------------------"
done

echo "Starting Docker Tests for All Defenses..."
DEFENSES=("STRIP" "NC" "DifferentialPrivacy")

for defense in "${DEFENSES[@]}"; do
    echo "Testing Defense in Docker: $defense"
    docker run --rm -v "$HOST_OUTPUT_DIR":/workspace/output $IMAGE_NAME python3 main.py defense --method "$defense" --output_path /workspace/output
    echo "------------------------------------------------"
done

echo "Docker testing completed. Results are in $HOST_OUTPUT_DIR"

