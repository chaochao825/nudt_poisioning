"""
Lightweight SSE-style simulator for multiple poisoning / backdoor attack types.
Matches the structure found in the attached json specs under attack_json/.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torchvision
from utils import ensure_path, sse_envelope, default_callback_params

RESNET50_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"

# Mapping attack name -> default params from the provided JSON stubs
ATTACK_PRESETS: Dict[str, Dict] = {
    "BadNets": {
        "model_name": "resnet50_v1",
        "dataset": "CIFAR-10",
        "poison_ratio": 0.04,
        "target_label": 0,
        "category": "实战",
    },
    "Trojan": {
        "model_name": "resnet101",
        "dataset": "CIFAR-10",
        "poison_ratio": 0.03,
        "target_label": 5,
        "category": "实战",
    },
    "FeatureCollision": {
        "model_name": "vgg19",
        "dataset": "CIFAR-10",
        "poison_ratio": 0.02,
        "target_label": 1,
        "category": "实战",
    },
    "Triggerless": {
        "model_name": "resnet50_stream",
        "dataset": "CIFAR-10",
        "poison_ratio": 0.05,
        "target_label": 2,
        "category": "实战",
    },
    "DynamicBackdoor": {
        "model_name": "resnet50",
        "dataset": "CIFAR-10",
        "poison_ratio": 0.05,
        "target_label": 2,
        "category": "模型注入",
    },
    "PhysicalBackdoor": {
        "model_name": "resnet50",
        "dataset": "CIFAR-10",
        "poison_ratio": 0.06,
        "target_label": 7,
        "category": "模型注入",
    },
    "NeuronInterference": {
        "model_name": "resnet34",
        "dataset": "CIFAR-10",
        "poison_ratio": 0.03,
        "target_label": 5,
        "category": "模型注入",
    },
    "ModelPoisoning": {
        "model_name": "resnet50_sync",
        "dataset": "CIFAR-10",
        "poison_ratio": 0.03,
        "target_label": 8,
        "category": "模型注入",
    },
    # basic simulated gradient-style attacks
    "FGSM": {"model_name": "resnet50_v1", "dataset": "CIFAR-10", "epsilon": 8 / 255},
    "PGD": {"model_name": "resnet50_v1", "dataset": "CIFAR-10", "steps": 10},
    "Optimization": {"model_name": "resnet50_v1", "dataset": "CIFAR-10"},
    "MI-FGSM": {"model_name": "resnet50_v1", "dataset": "CIFAR-10"},
    "BackdoorPoison": {"model_name": "resnet50_v1", "dataset": "CIFAR-10"},
}


def _now() -> str:
    return datetime.now().strftime("%Y/%m/%d-%H:%M:%S:%f")[:-3]


def emit(event: str, progress: float, message: str, details: Dict, log: str = None, resp_code: int = 0):
    sse_envelope(
        event=event,
        progress=progress,
        message=message,
        log=log,
        details=details,
        resp_code=resp_code,
        resp_msg="ok" if resp_code == 0 else "unsupported",
        callback_params=default_callback_params(),
    )


def download_resnet50(weight_dir: str) -> str:
    ensure_path("model_dir_ready", weight_dir, create=True)
    weight_path = os.path.join(weight_dir, "resnet50.pth")
    if os.path.exists(weight_path):
        return weight_path
    state_dict = torch.hub.load_state_dict_from_url(RESNET50_URL, model_dir=weight_dir, progress=True, map_location="cpu")
    model = torchvision.models.resnet50()
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), weight_path)
    return weight_path


def simulate_training(attack: str, preset: Dict):
    emit(
        "process_start",
        5,
        "投毒流程初始化",
        {
            "dataset": preset.get("dataset", "CIFAR-10"),
            "poison_ratio": preset.get("poison_ratio", 0.05),
            "target_label": preset.get("target_label", 0),
            "category": preset.get("category", "基础模拟"),
            "attack_name": attack,
        },
    )

    emit(
        "dataset_loaded",
        15,
        "数据集加载完成",
        {
            "train_samples": 2048,
            "validation_samples": 256,
        },
    )

    samples_to_poison = max(20, int(2048 * preset.get("poison_ratio", 0.05)))
    emit(
        "poison_generation_start",
        30,
        "开始构造投毒样本",
        {"target_label": preset.get("target_label", 0), "samples_to_poison": samples_to_poison},
    )
    emit(
        "poison_generation_progress",
        40,
        "投毒样本生成中",
        {"completed": int(samples_to_poison * 0.6), "stealth_score": 0.82},
    )
    emit(
        "poison_generation_completed",
        50,
        "投毒样本准备就绪",
        {"poisoned_samples": samples_to_poison, "storage_path": f"poison/{attack.lower()}_patches.bin"},
    )

    # Epoch checkpoints similar to JSON (2,4,6)
    for p, ep, loss, acc in [(60, 2, 0.85, 0.8), (70, 4, 0.75, 0.82), (78, 6, 0.65, 0.84)]:
        emit(
            "training_epoch",
            p,
            f"训练轮次{ep}完成",
            {"epoch": ep, "loss": loss, "accuracy": acc},
        )

    emit(
        "evaluation_clean",
        85,
        "干净样本评估",
        {"clean_accuracy": 0.86, "accuracy_drop": 0.02},
    )
    emit(
        "evaluation_backdoor",
        92,
        "后门触发评估",
        {"backdoor_success_rate": 0.68, "trigger_samples": 64},
    )
    emit(
        "result_finalized",
        100,
        "输出训练摘要",
        {
            "clean_accuracy": 0.86,
            "poisoned_accuracy": 0.84,
            "backdoor_success_rate": 0.68,
            "notes": "占位结果，便于调试",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack",
        type=str,
        default="BadNets",
        choices=list(ATTACK_PRESETS.keys()),
        help="Attack name from the predefined list",
    )
    parser.add_argument("--model_dir", type=str, default="./input/model", help="where to store resnet50 weights")
    args = parser.parse_args()
    run_attack(args.attack, args.model_dir)


def run_attack(attack: str, model_dir: str):
    preset = ATTACK_PRESETS[attack]
    emit(
        "model_loaded",
        10,
        "目标模型加载成功",
        {
            "model_name": preset.get("model_name", "resnet50"),
            "model_path": download_resnet50(model_dir),
            "model_type": "classification",
            "input_shape": [3, 224, 224],
            "num_classes": 1000,
        },
    )
    simulate_training(attack, preset)


if __name__ == "__main__":
    main()

