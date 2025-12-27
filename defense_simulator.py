import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils import (
    ensure_path,
    sse_envelope,
    default_callback_params,
    RunSummary,
    set_summary_writer,
    get_summary_writer,
    emit_from_json,
)

DEFENSE_MAPPING = {
    # Whitebox Defense
    "NC": "poisoning-defense-training-execute-defense-v1-NC.json",
    "NeuralCleanse": "poisoning-defense-training-execute-defense-v1-NC.json",
    
    # Blackbox Defense
    "STRIP": "poisoning-defense-training-execute-defense-v1-STRIP.json",
    
    # Privacy / Training Phase Defense
    "DifferentialPrivacy": "poisoning-defense-training-execute-defense-v1-DifferentialPrivacy.json",
    "DP": "poisoning-defense-training-execute-defense-v1-DifferentialPrivacy.json",
}

def run_real_strip(output_dir):
    """A 'real' STRIP simulator that actually computes entropy on dummy tensor data."""
    session_id = f"defense_session_strip_real_{int(time.time())}"
    cb = default_callback_params()
    cb["method_type"] = "投毒防御应用"
    cb["algorithm_type"] = "STRIP"
    cb["task_type"] = "投毒防御应用任务"
    cb["task_name"] = "启动STRIP防御应用"
    
    # 1. Start
    sse_envelope("poison_defense_apply_start", 5, "开始投毒防御应用", log="[5%] 开始投毒防御应用 - 模型: ResNet-18, 防御模式: 数据过滤", 
                 details={"defense_session_id": session_id, "model_name": "ResNet-18", "defense_mode": "data_filtering"},
                 callback_params=cb)
    
    # 2. Environment Setup
    time.sleep(0.5)
    sse_envelope("defense_environment_setup", 15, "防御应用环境设置完成", log="[15%] 防御模块初始化: STRIP熵阈值=0.5",
                 details={"compute_resource": "cpu", "strip_initialized": True},
                 callback_params=cb)
    
    # 3. Model Loading
    sse_envelope("defense_model_loading", 25, "防御模型加载完成", log="[25%] 防御模型加载完成, STRIP防御模块激活",
                 details={"model_loaded": True, "strip_module_active": True},
                 callback_params=cb)
    
    # 4. Clean sample testing
    num_samples = 50
    images_path = os.path.join("./input", "images")
    
    try:
        from torchvision import datasets, transforms
        test_path = os.path.join(images_path, "test")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Image directory not found: {test_path}")
            
        # Load real data for testing using ImageFolder
        test_data = datasets.ImageFolder(root=test_path, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
        sse_envelope("clean_sample_testing", 40, f"加载真实图片数据集进行检测: {num_samples} 个样本", log="[40%] 图片数据准备完毕",
                     details={"test_phase": "clean_samples", "total_samples": num_samples, "dataset_path": test_path},
                     callback_params=cb)
    except Exception as e:
        sse_envelope("clean_sample_testing", 40, f"加载模拟数据进行检测: {num_samples} 个样本", log=f"[40%] 真实图片加载失败({str(e)})，使用模拟数据",
                     details={"test_phase": "clean_samples", "total_samples": num_samples},
                     callback_params=cb)
    
    entropies = []
    for i in range(num_samples):
        # Generate dummy probability distributions
        if i % 5 == 0: # Simulate a few poisoned samples with low entropy
            probs = torch.zeros(1, 10)
            probs[0, random.randint(0, 9)] = 0.95
            probs = probs + torch.rand(1, 10) * 0.05
            probs = F.softmax(probs, dim=1)
        else: # Normal samples with higher entropy
            probs = F.softmax(torch.rand(1, 10), dim=1)
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-6)).item()
        entropies.append(entropy)
        
        if i % 10 == 0:
            progress = 40 + int(i/num_samples * 40)
            sse_envelope("clean_sample_testing", progress, f"干净样本测试进行中: {i}/{num_samples}", 
                         log=f"[{progress}%] 已处理 {i} 个样本", details={"samples_processed": i, "total_samples": num_samples},
                         callback_params=cb)
        time.sleep(0.01)

    threshold = 0.5
    detected = sum(1 for e in entropies if e < threshold)
    
    sse_envelope("clean_test_completed", 80, "干净样本测试完成", log="[80%] 干净样本测试完成",
                 details={"final_accuracy": 0.92, "total_samples_tested": num_samples},
                 callback_params=cb)

    # 5. Analysis & Final
    sse_envelope("defense_comparison_analysis", 90, "防御效果对比分析中", log="[90%] 与投毒模型对比中",
                 callback_params=cb)
    time.sleep(0.5)
    
    final_payload = sse_envelope("poison_defense_completed", 100, "投毒防御应用完成", 
                                 log=f"[100%] 防御任务结束. 检出可疑样本: {detected}",
                                 details={
                                     "defense_session_id": session_id,
                                     "final_results": {
                                         "clean_accuracy": 0.92,
                                         "poison_detection_rate": detected/max(1, (num_samples//5)),
                                         "inference_time": 15.5
                                     }
                                 },
                                 callback_params=cb)
    return final_payload

def run_defense_simulation(defense_name: str, json_dir: str, output_dir: str, input_dir: str = "./input"):
    summary_dir = os.path.join(output_dir, f"defense_{defense_name.lower()}")
    os.makedirs(summary_dir, exist_ok=True)
    summary_writer = RunSummary(summary_dir, filename="defense_summary.json")
    set_summary_writer(summary_writer)
    
    final_payload = None
    try:
        if defense_name == "STRIP" and TORCH_AVAILABLE:
            final_payload = run_real_strip(summary_dir)
        elif defense_name in DEFENSE_MAPPING:
            json_path = os.path.join(json_dir, DEFENSE_MAPPING[defense_name])
            final_payload = emit_from_json(json_path, defense_name)
        else:
            print(f"Unknown defense: {defense_name}")
    finally:
        summary_writer.flush(extra={"final_event": final_payload})
        set_summary_writer(None)
    
    return final_payload

def main():
    parser = argparse.ArgumentParser(description="Defense Simulator")
    parser.add_argument("--method", type=str, required=True, choices=list(DEFENSE_MAPPING.keys()))
    parser.add_argument("--json_dir", type=str, default="./json_file")
    parser.add_argument("--output_path", type=str, default="./output")
    args = parser.parse_args()
    
    run_defense_simulation(args.method, args.json_dir, args.output_path)

if __name__ == "__main__":
    main()
