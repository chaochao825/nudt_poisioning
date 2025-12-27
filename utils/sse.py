import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List


def _serialize(value: Any):
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
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


class RunSummary:
    """Collect SSE payloads and flush them to disk as a JSON report."""

    def __init__(self, output_dir: Optional[str], filename: str = "sse_run_summary.json") -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.filename = filename
        self.events: List[Dict[str, Any]] = []

    def record(self, payload: Dict[str, Any]) -> None:
        self.events.append(payload)

    def flush(self, extra: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        if not self.output_dir:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / self.filename
        report = {
            "events": self.events,
        }
        if extra:
            report["summary"] = extra
        with target.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, default=_serialize)
        return target


_SUMMARY_WRITER: Optional[RunSummary] = None


def set_summary_writer(writer: Optional[RunSummary]) -> None:
    global _SUMMARY_WRITER
    _SUMMARY_WRITER = writer


def get_summary_writer() -> Optional[RunSummary]:
    return _SUMMARY_WRITER


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
) -> Dict[str, Any]:
    if details is None:
        details = {}
    
    # If this is a final event, mark it explicitly in details
    if event == "final_result" or progress == 100:
        details["is_final"] = True
        details["completion_timestamp"] = datetime.now().isoformat()

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
            "details": details,
        },
    }
    sse_print(event, payload)
    writer = get_summary_writer()
    if writer is not None:
        writer.record(payload)
    return payload


def emit_from_json(json_path: str, algorithm_type: str):
    """
    Read SSE messages from a JSON file and emit them with updated timestamps.
    """
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Try to find the messages list. Different JSONs might have different structures.
    messages = []
    if "attack_sse_messages" in data:
        if algorithm_type in data["attack_sse_messages"]:
            messages = data["attack_sse_messages"][algorithm_type].get("sse_messages", [])
    elif "defense_sse_messages" in data:
        if algorithm_type in data["defense_sse_messages"]:
            messages = data["defense_sse_messages"][algorithm_type].get("sse_messages", [])
    
    last_payload = None
    total_msgs = len(messages)
    for i, msg in enumerate(messages):
        # Update timestamp
        msg["time_stamp"] = _now_string()
        
        # Ensure details exist
        if "data" in msg:
            if "details" not in msg["data"]:
                msg["data"]["details"] = {}
            
            # Mark final if it's the last one
            if i == total_msgs - 1:
                msg["data"]["details"]["is_final"] = True
                msg["data"]["details"]["completion_timestamp"] = datetime.now().isoformat()
                if "progress" not in msg["data"] or msg["data"]["progress"] is None:
                    msg["data"]["progress"] = 100

        # Print and record
        event = msg.get("data", {}).get("event", "unknown")
        sse_print(event, msg)
        writer = get_summary_writer()
        if writer is not None:
            writer.record(msg)
        last_payload = msg
        
        # Simulate some delay
        time_delay = msg.get("delay", 0.5)
        import time
        time.sleep(min(time_delay, 0.5)) 
        
    return last_payload


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

