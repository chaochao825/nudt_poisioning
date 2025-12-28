#!/bin/bash

# Base directory
BASE_DIR=$(dirname $(dirname $(readlink -f $0)))
cd $BASE_DIR

# Use the copy in home directory for testing
TEST_DIR="/home/wangmeiqi/nudt_poisioning_v2"
cd $TEST_DIR

# 1. Attacks to test with varied parameters
declare -A ATTACK_PARAMS
ATTACK_PARAMS["BadNets"]="-e poison_rate=0.15 -e trigger_size=4"
ATTACK_PARAMS["Trojan"]="-e poison_rate=0.05 -e trigger_size=2"
ATTACK_PARAMS["Feature Collision"]="-e poison_rate=0.08"
ATTACK_PARAMS["Triggerless"]="-e poison_rate=0.2"
ATTACK_PARAMS["Dynamic Backdoor"]="-e poison_rate=0.12 -e trigger_size=5"
ATTACK_PARAMS["Physical Backdoor"]="-e poison_rate=0.1 -e trigger_size=8"
ATTACK_PARAMS["Neuron Interference"]="-e poison_rate=0.03"
ATTACK_PARAMS["Model Poisoning"]="-e poison_rate=0.25"

ATTACKS=(
    "BadNets"
    "Trojan"
    "Feature Collision"
    "Triggerless"
    "Dynamic Backdoor"
    "Physical Backdoor"
    "Neuron Interference"
    "Model Poisoning"
)

# 2. Defenses to test with varied parameters
declare -A DEFENSE_PARAMS
DEFENSE_PARAMS["STRIP"]="-e sensitivity=0.8 -e threshold=0.4"
DEFENSE_PARAMS["NC"]="-e sensitivity=0.3"
DEFENSE_PARAMS["DifferentialPrivacy"]="-e sensitivity=0.6 -e train_subset=500"

DEFENSES=(
    "STRIP"
    "NC"
    "DifferentialPrivacy"
)

echo "=========================================================="
echo "Starting Individual Tests for All Poisoning Methods"
echo "Format: SSE Simplified, Random Delays, 100% Progress"
echo "Features: Dynamic Parameters, Professional Language"
echo "=========================================================="

# Function to check results
check_result() {
    local method_name="$1"
    local type="$2"
    local method_norm=$(echo "$method_name" | tr '[:upper:]' '[:lower:]' | tr -d ' ')
    
    local method_dir
    if [ "$type" == "defense" ]; then
        method_dir="defense_$method_norm"
    else
        method_dir="$method_norm"
    fi
    
    local summary_file="$TEST_DIR/output_docker/$method_dir/${type}_summary.json"
    
    if [ -f "$summary_file" ]; then
        echo "[SUCCESS] $method_name: Result saved to $summary_file"
        if grep -q '"is_final": true' "$summary_file"; then
            echo "[SUCCESS] $method_name: Found is_final marker."
        else
            echo "[WARNING] $method_name: is_final marker NOT found in summary file!"
        fi
    else
        echo "[FAILURE] $method_name: Summary file NOT found at $summary_file"
    fi
}

# --- Test Attacks ---
echo ""
echo ">>> Testing Attacks with Dynamic Parameters..."
for attack in "${ATTACKS[@]}"; do
    params=${ATTACK_PARAMS["$attack"]}
    echo ""
    echo "----------------------------------------------------------"
    echo "Executing Attack: $attack (Params: $params)"
    
    # Overriding the docker run command logic within the script context
    IMAGE_NAME="nudt-poisoning:latest"
    HOST_OUTPUT_DIR="$TEST_DIR/output_docker"
    HOST_INPUT_DIR="$TEST_DIR/input"
    
    docker run --rm \
        -v "$TEST_DIR":/workspace \
        -v "$HOST_OUTPUT_DIR":/workspace/output \
        -v "$HOST_INPUT_DIR":/workspace/input \
        -e mode=attack \
        -e method="$attack" \
        -e OUTPUT_PATH=/workspace/output \
        -e INPUT_PATH=/workspace/input \
        $params \
        --network host \
        $IMAGE_NAME
        
    check_result "$attack" "attack"
    echo "----------------------------------------------------------"
done

# --- Test Defenses ---
echo ""
echo ">>> Testing Defenses with Dynamic Parameters..."
for defense in "${DEFENSES[@]}"; do
    params=${DEFENSE_PARAMS["$defense"]}
    echo ""
    echo "----------------------------------------------------------"
    echo "Executing Defense: $defense (Params: $params)"
    
    IMAGE_NAME="nudt-poisoning:latest"
    HOST_OUTPUT_DIR="$TEST_DIR/output_docker"
    HOST_INPUT_DIR="$TEST_DIR/input"
    
    docker run --rm \
        -v "$TEST_DIR":/workspace \
        -v "$HOST_OUTPUT_DIR":/workspace/output \
        -v "$HOST_INPUT_DIR":/workspace/input \
        -e mode=defense \
        -e method="$defense" \
        -e OUTPUT_PATH=/workspace/output \
        -e INPUT_PATH=/workspace/input \
        $params \
        --network host \
        $IMAGE_NAME
        
    check_result "$defense" "defense"
    echo "----------------------------------------------------------"
done

echo "=========================================================="
echo "All Individual Tests Completed"
echo "=========================================================="
