"""
Run all predefined attacks sequentially and emit SSE-style logs with a final summary.
"""

import argparse
from typing import List, Dict

from attack_simulator import ATTACK_PRESETS, run_attack
from utils import sse_envelope, default_callback_params


def emit(event: str, progress, message: str, details: Dict = None, log: str = None, resp_code: int = 0, resp_msg: str = "ok"):
    sse_envelope(
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
    parser.add_argument("--model_dir", type=str, default="./input/model", help="where to store resnet50 weights")
    args = parser.parse_args()

    attacks: List[str] = list(ATTACK_PRESETS.keys())
    results = []

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
            run_attack(attack, args.model_dir)
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

    emit(
        "final_result",
        100,
        "批量攻击测试完成",
        {"summary": results},
        log="[100%] 所有攻击任务结束",
    )


if __name__ == "__main__":
    main()

