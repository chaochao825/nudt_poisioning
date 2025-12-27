#!/bin/bash

# Base directory
BASE_DIR=$(dirname $(dirname $(readlink -f $0)))
cd $BASE_DIR

# Use the copy in home directory for testing
TEST_DIR="/home/wangmeiqi/nudt_poisioning_v2"
cd $TEST_DIR

# 1. Attacks to test
ATTACKS=(
    "BadNets"
    "Trojan"
    "Feature Collision"
    "Triggerless"
    "Dynamic Backdoor"
    "Physical Backdoor"
    "Neuron Interference"
    "Model Poisoning"
    "CleanLabel"
    "GradientShift"
    "LabelFlip"
    "RandomNoise"
    "SampleMix"
)

# 2. Defenses to test
DEFENSES=(
    "STRIP"
    "NC"
    "DifferentialPrivacy"
)

echo "=========================================================="
echo "Starting Individual Tests for All Poisoning Methods"
echo "=========================================================="

# Function to check results
check_result() {
    local method_dir=$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d ' ')
    local summary_file="$TEST_DIR/output_docker/$method_dir/${2}_summary.json"
    
    if [ -f "$summary_file" ]; then
        echo "[SUCCESS] $1: Result saved to $summary_file"
        # Check for is_final in the summary
        if grep -q '"is_final": true' "$summary_file"; then
            echo "[SUCCESS] $1: Found is_final marker."
        else
            echo "[WARNING] $1: is_final marker NOT found in summary file!"
        fi
    else
        echo "[FAILURE] $1: Summary file NOT found at $summary_file"
    fi
}

# --- Test Attacks ---
echo ""
echo ">>> Testing Attacks..."
for attack in "${ATTACKS[@]}"; do
    echo ""
    echo "Testing Attack: $attack"
    ./docker_run_attack.sh "$attack"
    check_result "$attack" "attack"
    echo "----------------------------------------------------------"
done

# --- Test Defenses ---
echo ""
echo ">>> Testing Defenses..."
for defense in "${DEFENSES[@]}"; do
    echo ""
    echo "Testing Defense: $defense"
    ./docker_run_defense.sh "$defense"
    check_result "$defense" "defense"
    echo "----------------------------------------------------------"
done

echo "=========================================================="
echo "All Individual Tests Completed"
echo "=========================================================="

