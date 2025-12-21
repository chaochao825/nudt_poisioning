import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


def _serialize(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _now_string() -> str:
    # Format: 2024/07/01-15:30:00:123
    return datetime.now().strftime("%Y/%m/%d-%H:%M:%S:%f")[:-3]


def default_callback_params() -> Dict[str, str]:
    return {
        "task_run_id": os.getenv("TASK_RUN_ID", "task_run_id_default"),
        "method_type": os.getenv("METHOD_TYPE", "数据投毒"),
        "algorithm_type": os.getenv("ALGORITHM_TYPE", "BadNets"),
        "task_type": os.getenv("TASK_TYPE", "投毒攻击执行"),
        "task_name": os.getenv("TASK_NAME", "BadNets数据投毒攻击"),
        "parent_task_id": os.getenv("PARENT_TASK_ID", "parent_task_default"),
        "user_name": os.getenv("USER_NAME", "nudt_user"),
    }


def sse_print(event: str, data: Dict[str, Any]) -> str:
    """
    Emit Server-Sent Event style logs used across the nudt projects.
    """
    json_str = json.dumps(data, ensure_ascii=False, default=_serialize)
    message = f"event: {event}\n" f"data: {json_str}\n"
    print(message, flush=True)
    return message


def sse_envelope(
    event: str,
    progress: Optional[float],
    message: str,
    log: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    resp_code: int = 0,
    resp_msg: str = "操作成功",
    callback_params: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "resp_code": resp_code,
        "resp_msg": resp_msg,
        "time_stamp": _now_string(),
        "data": {
            "event": event,
            "callback_params": callback_params or default_callback_params(),
            "progress": progress,
            "message": message,
            "log": log,
            "details": details or {},
        },
    }
    sse_print(event, payload)


def ensure_path(event: str, path: str, create: bool = False) -> None:
    """
    Validate or create a path, and emit SSE logs with envelope.
    """
    try:
        if create:
            os.makedirs(path, exist_ok=True)
        if os.path.exists(path):
            sse_envelope(
                event=event,
                progress=None,
                message="Path is ready.",
                log=f"[info] path ready: {path}",
                details={"file_name": path},
            )
        else:
            raise FileNotFoundError(path)
    except Exception as exc:
        sse_envelope(
            event=event,
            progress=None,
            message=f"{exc}",
            log=f"[error] {exc}",
            details={"file_name": path},
            resp_code=1,
            resp_msg="操作失败",
        )

