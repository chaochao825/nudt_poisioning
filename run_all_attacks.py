"""
Run all predefined attacks sequentially and emit SSE-style logs with a final summary.
"""

import argparse
from typing import List, Dict

from attack_simulator import ATTACK_PRESETS, MODEL_CHOICES, normalize_model_name, run_attack
from utils import sse_envelope, default_callback_params, RunSummary, set_summary_writer


def emit(event: str, progress, message: str, details: Dict = None, log: str = None, resp_code: int = 0, resp_msg: str = "ok"):
    return sse_envelope(
        event=event,
        progress=progress,
        message=message,
        log=log,
        details=details or {},
        resp_code=resp_code,
        resp_msg=resp_msg,
        callback_params=default_callback_params(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./input/model", help="directory for cached model weights")
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=MODEL_CHOICES,
        help="base model to reuse for all attacks",
    )
    parser.add_argument("--output_path", type=str, default="./output", help="where to store summaries and logs")
    args = parser.parse_args()

    attacks: List[str] = list(ATTACK_PRESETS.keys())
    results = []
    summary_writer = RunSummary(args.output_path, filename="batch_attacks_summary.json")
    set_summary_writer(summary_writer)

    final_payload = None
    try:
        emit(
            "batch_start",
            1,
            "批量攻击测试开始",
            {"attacks": attacks},
            log="[1%] 开始批量攻击模拟",
        )

        for idx, attack in enumerate(attacks, start=1):
            emit(
                "attack_start",
                3 + idx,
                f"开始 {attack}",
                {"attack": attack, "index": idx, "total": len(attacks)},
                log=f"[{attack}] 启动",
            )
            try:
                resolved = normalize_model_name(args.model_name)
                run_attack(attack, args.model_dir, args.output_path, model_name=resolved)
                results.append({"attack": attack, "status": "success"})
                emit(
                    "attack_done",
                    3 + idx,
                    f"{attack} 完成",
                    {"attack": attack, "status": "success"},
                    log=f"[{attack}] 完成",
                )
            except Exception as exc:
                results.append({"attack": attack, "status": "failed", "error": str(exc)})
                emit(
                    "attack_failed",
                    3 + idx,
                    f"{attack} 失败",
                    {"attack": attack, "status": "failed", "error": str(exc)},
                    log=f"[{attack}] 失败: {exc}",
                    resp_code=1,
                    resp_msg="failed",
                )
        final_payload = emit(
            "final_result",
            100,
            "批量攻击测试完成",
            {"summary": results},
            log="[100%] 所有攻击任务结束",
        )
    finally:
        summary_writer.flush(extra={"final_event": final_payload})
        set_summary_writer(None)


if __name__ == "__main__":
    main()

