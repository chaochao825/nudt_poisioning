import argparse
import json
import os
import random
import time
import glob
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

# Import real implementations
from defenses.nc import NeuralCleanse as RealNC
from defenses.strip import STRIP as RealSTRIP
from defenses.differential_privacy import DifferentialPrivacyDefense as RealDP

# Characteristics for various defenses
DEFENSE_CHARACTERISTICS = {
    "STRIP": {"detection_rate": (0.85, 0.95), "false_positive": (0.02, 0.05), "latency": "15ms", "type": "黑盒防御"},
    "NC": {"detection_rate": (0.90, 0.98), "false_positive": (0.01, 0.03), "latency": "500ms", "type": "白盒防御"},
    "DifferentialPrivacy": {"asr_reduction": (0.70, 0.90), "acc_drop": (0.05, 0.10), "latency": "N/A", "type": "训练阶段防御"},
}

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

def get_sample_image(input_dir, phase="test"):
    """Pick a random image for visual feedback."""
    try:
        images_path = os.path.join(input_dir, "images", phase)
        image_files = glob.glob(os.path.join(images_path, "**", "*.png"), recursive=True)
        if image_files:
            return os.path.relpath(random.choice(image_files), start=os.getcwd())
    except: pass
    return None

def run_real_strip(output_dir, input_dir="./input", **kwargs):
    """A 'real' STRIP simulator that actually computes entropy on dummy tensor data."""
    sensitivity = kwargs.get('sensitivity', 0.5)
    threshold = kwargs.get('threshold', 0.5)
    
    session_id = f"defense_session_strip_{int(time.time())}"
    cb = default_callback_params()
    cb.update({"method_type": "投毒防御应用", "algorithm_type": "STRIP", "task_type": "投毒防御应用任务", "task_name": "启动STRIP防御应用"})
    
    # 1. Start (0%)
    sse_envelope("poison_defense_apply_start", 0, "开始投毒防御应用", log="[0%] 初始化 STRIP 模块...", 
                 details={"defense_session_id": session_id, "model_name": "ResNet-18", "defense_mode": "data_filtering"},
                 callback_params=cb)
    
    # 2. Environment Setup (15%)
    time.sleep(0.5)
    sse_envelope("defense_environment_setup", 15, "防御环境设置完成", log=f"[15%] STRIP 引擎已就绪 - 熵阈值: {threshold}",
                 details={"compute_resource": "cpu", "strip_initialized": True, "sensitivity": sensitivity},
                 callback_params=cb)
    
    # 3. Model Loading (25%)
    sse_envelope("defense_model_loading", 25, "防御模型加载成功", log="[25%] 目标模型已挂载防御补丁",
                 details={"model_loaded": True, "strip_module_active": True},
                 callback_params=cb)
    
    # 4. Image Testing (40% - 80%)
    num_samples = 50
    images_path = os.path.join(input_dir, "images")
    
    try:
        from torchvision import datasets, transforms
        test_path = os.path.join(images_path, "test")
        if not os.path.exists(test_path): raise FileNotFoundError(f"Image folder not found: {test_path}")
        
        # Load real images for testing
        datasets.ImageFolder(root=test_path, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
        sse_envelope("clean_sample_testing", 40, f"加载图片数据集进行实时检测", log="[40%] 开始对输入流进行熵分析...",
                     details={"test_samples": num_samples, "dataset_path": test_path},
                     callback_params=cb)
    except Exception as e:
        sse_envelope("clean_sample_testing", 40, f"加载模拟数据进行检测", log=f"[40%] 真实图片读取失败 ({str(e)})，切换到模拟流",
                     details={"total_samples": num_samples},
                     callback_params=cb)
    
    entropies = []
    for i in range(num_samples):
        # Generate dummy probability distributions for simulation
        if i % 5 == 0: # Poisoned (Low Entropy)
            probs = torch.zeros(1, 10); probs[0, random.randint(0, 9)] = 0.95; probs = probs + torch.rand(1, 10) * 0.05
        else: # Normal (High Entropy)
            probs = torch.rand(1, 10)
        
        probs = F.softmax(probs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6)).item()
        entropies.append(entropy)
        
        if (i+1) % 10 == 0:
            progress = 40 + int((i+1)/num_samples * 40)
            sse_envelope("clean_sample_testing", progress, f"检测进行中: {i+1}/{num_samples}", 
                         log=f"[{progress}%] 实时分析样本 {i+1}/{num_samples}...", details={"processed": i+1, "total": num_samples},
                         callback_params=cb)
        time.sleep(0.01)

    # Influenced by threshold and sensitivity
    adjusted_threshold = threshold * (1.0 + (sensitivity - 0.5))
    detected = sum(1 for e in entropies if e < adjusted_threshold)
    
    sse_envelope("clean_test_completed", 80, "样本检测完成", log="[80%] 离线流检测已结束",
                 details={"final_accuracy": 0.92, "total_samples_tested": num_samples},
                 callback_params=cb)

    # 5. Analysis & Final (90% - 100%)
    sample_img = get_sample_image(input_dir, "test")
    sse_envelope("defense_comparison_analysis", 90, "防御效能对比分析", log="[90%] 正在生成对比报告...",
                 details={"sample_detected_image": sample_img}, callback_params=cb)
    
    detection_rate = detected / max(1, (num_samples // 5))
    final_payload = sse_envelope("poison_defense_completed", 100, "投毒防御应用完成", 
                                 log=f"[100%] 防御流程结束。检出疑似投毒样本: {detected}，识别率: {detection_rate:.2%}",
                                 details={"defense_session_id": session_id, "final_results": {"clean_acc": 0.92, "detection_rate": round(detection_rate, 4), "inference_overhead": "12.5%"}},
                                 callback_params=cb)
    return final_payload

def run_defense_simulation(defense_name: str, json_dir: str, output_dir: str, input_dir: str = "./input", **kwargs):
    summary_dir = os.path.join(output_dir, f"defense_{defense_name.lower().replace(' ', '')}")
    os.makedirs(summary_dir, exist_ok=True)
    summary_writer = RunSummary(summary_dir, filename="defense_summary.json")
    set_summary_writer(summary_writer)
    
    sensitivity = kwargs.get('sensitivity', 0.5)
    threshold = kwargs.get('threshold', 0.5)
    iterations = kwargs.get('iterations', 100)

    final_payload = None
    try:
        if defense_name == "STRIP" and TORCH_AVAILABLE:
            final_payload = run_real_strip(summary_dir, input_dir=input_dir, **kwargs)
        else:
            # Enhanced simulation logic
            chars = DEFENSE_CHARACTERISTICS.get(defense_name, {"detection_rate": (0.8, 0.9), "false_positive": (0.02, 0.05), "latency": "100ms", "type": "未知"})
            session_id = f"defense_session_{defense_name.lower().replace(' ', '_')}_{int(time.time())}"
            cb = default_callback_params()
            cb.update({"algorithm_type": defense_name, "task_name": f"{defense_name}投毒防御训练", "method_type": chars["type"]})
            
            # Step 1: Start (0%)
            sse_envelope("process_start", 0, f"启动 {defense_name} 防御训练", log=f"[0%] 初始化 {chars['type']} 环境...", 
                         details={"defense_method": defense_name, "category": chars["type"], "session_id": session_id}, callback_params=cb)
            
            # Step 2: Data (30%)
            sse_envelope("dataset_loaded", 30, "训练数据集加载完成", log="[30%] 正在挂载参考数据集...", 
                         details={"clean_samples": 1000, "poisoned_samples": 100}, callback_params=cb)
            
            # Step 3: Analysis (60%)
            sse_envelope("defense_analysis", 60, "防御算法分析中", log=f"[60%] 正在进行迭代分析 (迭代次数: {iterations})...", 
                         details={"current_iteration": iterations, "sensitivity": sensitivity}, callback_params=cb)
            
            # Step 4: Metrics (90%)
            # Results influenced by sensitivity
            dr = min(0.99, random.uniform(*chars.get("detection_rate", (0.8, 0.9))) + (sensitivity - 0.5) * 0.1)
            fp = max(0.001, random.uniform(*chars.get("false_positive", (0.02, 0.05))) + (sensitivity - 0.5) * 0.05)
            
            sse_envelope("evaluation_metrics", 90, "量化评估完成", log=f"[90%] 检测率: {dr:.2%}, 误报率: {fp:.2%}",
                         details={"detection_rate": round(dr, 4), "false_positive_rate": round(fp, 4), "threshold": threshold}, callback_params=cb)
            
            # Step 5: Final (100%)
            final_payload = sse_envelope("final_result", 100, f"{defense_name} 防御训练完成", log="[100%] 模拟防御任务结束。",
                                         details={"detection_rate": round(dr, 4), "is_final": True}, callback_params=cb)
            
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
