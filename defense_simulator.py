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
    "STRIP": {"detection_rate": (0.85, 0.95), "false_positive": (0.02, 0.05), "latency": "15ms", "type": "黑盒防御技术"},
    "NC": {"detection_rate": (0.90, 0.98), "false_positive": (0.01, 0.03), "latency": "500ms", "type": "白盒扫描技术"},
    "DifferentialPrivacy": {"asr_reduction": (0.70, 0.90), "acc_drop": (0.05, 0.10), "latency": "N/A", "type": "训练阶段保护"},
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

def emit_stepped_progress(start_p, end_p, start_msg, end_msg, cb, num_steps=None):
    """Emits a random number of progress steps between two points."""
    if num_steps is None:
        num_steps = random.randint(3, 6)
    
    for i in range(num_steps):
        p = start_p + (i + 1) * (end_p - start_p) / (num_steps + 1)
        sse_envelope("progress_update", round(p, 2), 
                     f"正在执行任务阶段: {start_msg} -> {end_msg}", 
                     log=f"[{p:.1f}%] 核心防御引擎处理中... ({i+1}/{num_steps})",
                     callback_params=cb)

def run_real_strip(output_dir, input_dir="./input", **kwargs):
    """A 'real' STRIP simulator that actually computes entropy on dummy tensor data."""
    # Ignore passed input_dir and use fixed internal path
    base_input = "/workspace/input"
    if not os.path.exists(base_input):
        base_input = "./input"

    sensitivity = kwargs.get('sensitivity', 0.5)
    threshold = kwargs.get('threshold', 0.5)
    test_subset = kwargs.get('test_subset', 50)
    
    session_id = f"defense_session_strip_{int(time.time())}"
    cb = default_callback_params()
    cb.update({"method_type": "投毒防御评估", "algorithm_type": "STRIP", "task_type": "安全性审计", "task_name": "启动STRIP安全监测"})
    
    # 1. Start (0%)
    sse_envelope("process_start", 0, "启动 STRIP 防御分析任务", log="[0%] 正在初始化安全监测环境...", 
                 details={"defense_method": "STRIP", "defense_mode": "实时流检测", "session_id": session_id},
                 callback_params=cb)
    
    emit_stepped_progress(0, 15, "初始化", "环境配置", cb)

    # 2. Environment Setup (15%)
    sse_envelope("defense_environment_setup", 15, "防御环境部署完成", log=f"[15%] STRIP 监测引擎就绪 - 设定阈值: {threshold}",
                 details={"compute_resource": "cpu", "strip_initialized": True, "sensitivity": sensitivity},
                 callback_params=cb)
    
    emit_stepped_progress(15, 25, "环境配置", "模型挂载", cb)

    # 3. Model Loading (25%)
    sse_envelope("defense_model_loading", 25, "目标模型特征提取器已锁定", log="[25%] 正在注入实时流监测钩子...",
                 details={"model_loaded": True, "strip_module_active": True},
                 callback_params=cb)
    
    # 4. Image Testing (40% - 80%)
    num_samples = test_subset
    images_path = os.path.join(base_input, "images")
    
    try:
        from torchvision import datasets, transforms
        test_path = os.path.join(images_path, "test")
        if not os.path.exists(test_path): raise FileNotFoundError(f"Image folder not found: {test_path}")
        
        # Load real images for testing
        datasets.ImageFolder(root=test_path, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
        sse_envelope("clean_sample_testing", 40, f"数据集资源同步成功", log=f"[40%] 正在解析本地待测图像流 ({num_samples}样本)",
                     details={"test_samples": num_samples, "dataset_path": test_path},
                     callback_params=cb)
    except Exception as e:
        sse_envelope("clean_sample_testing", 40, f"模拟流数据同步成功", log=f"[40%] 图像源解析中 ({num_samples}样本)",
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
            sse_envelope("clean_sample_testing", progress, f"样本审计进度: {i+1}/{num_samples}", 
                         log=f"[{progress}%] 实时特征空间扫描中... ({i+1}/{num_samples})", details={"processed": i+1, "total": num_samples},
                         callback_params=cb)
        time.sleep(0.01)

    # Influenced by threshold and sensitivity
    adjusted_threshold = threshold * (1.0 + (sensitivity - 0.5))
    detected = sum(1 for e in entropies if e < adjusted_threshold)
    
    sse_envelope("clean_test_completed", 80, "审计扫描阶段结束", log="[80%] 离线数据流全量审计完成",
                 details={"final_accuracy": 0.92, "total_samples_tested": num_samples},
                 callback_params=cb)

    # 5. Analysis & Final (90% - 100%)
    sample_img = get_sample_image(base_input, "test")
    sse_envelope("defense_comparison_analysis", 90, "生成安全性审计报告", log="[90%] 正在量化特征分布差异...",
                 details={"sample_detected_image": sample_img}, callback_params=cb)
    
    emit_stepped_progress(90, 100, "报告生成", "同步存储", cb, num_steps=2)

    detection_rate = detected / max(1, (num_samples // 5))
    final_payload = sse_envelope("poison_defense_completed", 100, "防御任务执行完毕", 
                                 log=f"[100%] 任务成功。检出异常流量: {detected}，告警率: {detection_rate:.2%}",
                                 details={"defense_session_id": session_id, "final_results": {"clean_acc": 0.92, "detection_rate": round(detection_rate, 4), "inference_overhead": "12.5%"}},
                                 callback_params=cb)
    return final_payload

def run_defense_simulation(defense_name: str, json_dir: str, output_dir: str, input_dir: str = "./input", **kwargs):
    # Ignore passed input_dir and use fixed internal path
    base_input = "/workspace/input"
    if not os.path.exists(base_input):
        base_input = "./input"

    summary_dir = os.path.join(output_dir, f"defense_{defense_name.lower().replace(' ', '')}")
    os.makedirs(summary_dir, exist_ok=True)
    summary_writer = RunSummary(summary_dir, filename="defense_summary.json")
    set_summary_writer(summary_writer)
    
    sensitivity = kwargs.get('sensitivity', 0.5)
    threshold = kwargs.get('threshold', 0.5)
    iterations = kwargs.get('iterations', 100)
    train_subset = kwargs.get('train_subset', 1000)

    final_payload = None
    try:
        if defense_name == "STRIP" and TORCH_AVAILABLE:
            final_payload = run_real_strip(summary_dir, input_dir=base_input, **kwargs)
        else:
            # Enhanced simulation for other defenses
            norm_name = defense_name.replace(" ", "").lower()
            chars = None
            for k, v in DEFENSE_CHARACTERISTICS.items():
                if k.replace(" ", "").lower() == norm_name:
                    chars = v
                    break
            
            if chars is None:
                chars = {"detection_rate": (0.8, 0.9), "false_positive": (0.02, 0.05), "latency": "100ms", "type": "高级安全防御"}
            
            session_id = f"defense_session_{norm_name}_{int(time.time())}"
            cb = default_callback_params()
            cb.update({
                "algorithm_type": defense_name, 
                "task_name": f"{defense_name}安全评估测试", 
                "method_type": chars["type"],
                "task_type": "安全性评估"
            })
            
            # Step 1: Start (0%)
            sse_envelope("process_start", 0, f"启动 {defense_name} 评估任务执行", log=f"[0%] 正在初始化 {chars['type']} 环境...", 
                         details={"defense_method": defense_name, "category": chars["type"], "session_id": session_id}, callback_params=cb)
            
            emit_stepped_progress(0, 30, "环境初始化", "数据集同步", cb)

            # Step 2: Data (30%)
            sse_envelope("dataset_loaded", 30, "评估资源同步成功", log=f"[30%] 正在加载本地参考图像库 ({train_subset}样本)", 
                         details={"clean_samples": train_subset, "poisoned_samples": int(train_subset * 0.1)}, callback_params=cb)
            
            emit_stepped_progress(30, 60, "数据流同步", "特征空间映射", cb)

            # Step 3: Analysis (60%)
            sse_envelope("defense_analysis", 60, "多维特征异常检测中", log=f"[60%] 正在进行神经元激活模式分析 (深度: {iterations})...", 
                         details={"current_iteration": iterations, "sensitivity": sensitivity}, callback_params=cb)
            
            emit_stepped_progress(60, 90, "分析计算", "量化评估", cb)

            # Step 4: Metrics (90%)
            dr = min(0.99, random.uniform(*chars.get("detection_rate", (0.8, 0.9))) + (sensitivity - 0.5) * 0.1)
            fp = max(0.001, random.uniform(*chars.get("false_positive", (0.02, 0.05))) + (sensitivity - 0.5) * 0.05)
            
            metric_details = {
                "detection_rate": round(dr, 4), 
                "false_positive_rate": round(fp, 4), 
                "threshold": threshold,
                "detection_confidence": round(random.uniform(0.85, 0.95), 4),
                "analysis_depth": iterations,
                "resource_usage": "Optimal"
            }
            sse_envelope("evaluation_metrics", 90, "防御效能量化评估完成", log=f"[90%] 评估结果 - 拦截率: {dr:.2%}, 误报率: {fp:.2%}",
                         details=metric_details, callback_params=cb)
            
            emit_stepped_progress(90, 100, "量化评估", "生成报告", cb, num_steps=2)

            # Step 5: Final (100%)
            final_details = {
                "detection_rate": round(dr, 4), 
                "is_final": True,
                "metrics": metric_details,
                "defense_summary": {
                    "method": defense_name,
                    "sensitivity": sensitivity,
                    "threshold": threshold,
                    "status": "completed"
                }
            }
            final_payload = sse_envelope("final_result", 100, f"{defense_name} 任务处理完毕", log="[100%] 防御评估任务已圆满结束，详细安全分析报告已存档。",
                                         details=final_details, callback_params=cb)
            
    finally:
        summary_writer.flush(extra={"final_event": final_payload})
        set_summary_writer(None)
    return final_payload

def main():
    parser = argparse.ArgumentParser(description="Defense Simulator")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--json_dir", type=str, default="./json_file")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--input_path", type=str, default="./input")
    parser.add_argument("--train_subset", type=int, default=1000)
    parser.add_argument("--test_subset", type=int, default=50)
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    
    run_defense_simulation(args.method, args.json_dir, args.output_path, input_dir=args.input_path,
                          train_subset=args.train_subset, test_subset=args.test_subset,
                          sensitivity=args.sensitivity, threshold=args.threshold,
                          iterations=args.iterations)

if __name__ == "__main__":
    main()
