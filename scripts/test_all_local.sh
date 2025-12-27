#!/bin/bash

# Base directory
BASE_DIR=$(dirname $(dirname $(readlink -f $0)))
cd $BASE_DIR

# Output directory
OUTPUT_DIR="./output"
mkdir -p $OUTPUT_DIR

echo "Starting Local Tests for All Attacks..."
ATTACKS=("BadNets" "CleanLabel" "FeatureCollision" "GradientShift" "LabelFlip" "ModelPoisoning" "NeuronInterference" "PhysicalBackdoor" "RandomNoise" "SampleMix" "TriggerlessDynamicBackdoor" "Trojan")

for attack in "${ATTACKS[@]}"; do
    echo "Testing Attack: $attack"
    python3 main.py attack --method "$attack" --output_path "$OUTPUT_DIR"
    echo "------------------------------------------------"
done

echo "Starting Local Tests for All Defenses..."
DEFENSES=("STRIP" "NC" "DifferentialPrivacy")

for defense in "${DEFENSES[@]}"; do
    echo "Testing Defense: $defense"
    python3 main.py defense --method "$defense" --output_path "$OUTPUT_DIR"
    echo "------------------------------------------------"
done

echo "Local testing completed."

