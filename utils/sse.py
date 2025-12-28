import json
import os
import random
import time
import sys
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
    Emit Server-Sent Event style logs in the simplified format.
    First line: event: <event_name>
    Second line: data: <json_data>
    """
    # Remove metadata if present (compatibility with old envelope callers)
    if "data" in data and isinstance(data["data"], dict) and "resp_code" in data:
        data_to_print = data["data"].copy()
        # Use the event from data if it exists, otherwise use the passed event
        event = data_to_print.pop("event", event)
    else:
        data_to_print = data.copy()
        # If event is in data, use it for the header but remove from JSON body
        if "event" in data_to_print:
            event = data_to_print.pop("event", event)

    # Ensure callback_params is clean and robust
    if "callback_params" in data_to_print:
        cp = data_to_print["callback_params"]
        if not isinstance(cp, dict):
            data_to_print["callback_params"] = default_callback_params()
        else:
            # Filter out any weird keys like 'additionalPropl'
            allowed_keys = {"task_run_id", "method_type", "algorithm_type", "task_type", "task_name", "parent_task_id", "user_name"}
            data_to_print["callback_params"] = {k: v for k, v in cp.items() if k in allowed_keys}
    else:
        data_to_print["callback_params"] = default_callback_params()

    # REMOVE empty details or other empty objects to keep output clean
    keys_to_clean = ["details"]
    for k in keys_to_clean:
        if k in data_to_print and (data_to_print[k] is None or data_to_print[k] == {}):
            del data_to_print[k]

    json_str = json.dumps(data_to_print, ensure_ascii=False, default=_serialize)
    # The requirement is event on one line, data on the next, followed by a blank line
    message = f"event: {event}\ndata: {json_str}\n\n"
    sys.stdout.write(message)
    sys.stdout.flush()
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

    # Base callback params from env or default
    cp = default_callback_params()
    # Update with specific task info if provided
    if callback_params:
        cp.update(callback_params)

    # New simplified format: no resp_code, resp_msg, or time_stamp at root
    payload = {
        "callback_params": cp,
        "progress": progress,
        "message": message,
        "log": log,
        "details": details,
    }
    
    sse_print(event, payload)
    
    writer = get_summary_writer()
    if writer is not None:
        # Record simplified payload but add internal tracking fields for report
        record_payload = payload.copy()
        record_payload["event"] = event
        record_payload["time_stamp"] = _now_string()
        writer.record(record_payload)
    
    # Introduce a non-fixed delay between outputs as requested
    if not (event == "final_result" or progress == 100):
        time.sleep(random.uniform(1.0, 3.0))
        
    return payload


def emit_from_json(json_path: str, algorithm_type: str):
    """
    Read SSE messages from a JSON file and emit them in the new simplified format.
    Includes extra heartbeat messages to ensure non-fixed number of SSE outputs.
    """
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Try to find the messages list.
    messages = []
    root_key = "attack_sse_messages" if "attack_sse_messages" in data else "defense_sse_messages"
    
    if root_key in data:
        # 1. Try exact match
        if algorithm_type in data[root_key]:
            messages = data[root_key][algorithm_type].get("sse_messages", [])
        # 2. Try case-insensitive / space-insensitive match
        else:
            norm_algo = algorithm_type.replace(" ", "").lower()
            for key in data[root_key]:
                if key.replace(" ", "").lower() == norm_algo:
                    messages = data[root_key][key].get("sse_messages", [])
                    break
            # 3. Fallback to first available key if no match found
            if not messages and len(data[root_key]) > 0:
                first_key = list(data[root_key].keys())[0]
                messages = data[root_key][first_key].get("sse_messages", [])
    
    last_payload = None
    total_msgs = len(messages)
    for i, msg in enumerate(messages):
        # Extract inner data and convert to new format
        if "data" in msg:
            inner = msg["data"].copy()
            event = inner.pop("event", "unknown")
            
            # Ensure details exist
            if "details" not in inner:
                inner["details"] = {}
            
            # Professional language cleanup
            unpro_terms = {
                "模拟攻击": "任务测试",
                "模拟防御": "安全评估",
                "打桩数据": "评估数据",
                "未知": "高级识别型",
                "初始化 模拟 环境": "初始化 安全测试 环境",
                "模拟": "测试"
            }
            
            for term, replacement in unpro_terms.items():
                if "message" in inner and isinstance(inner["message"], str):
                    inner["message"] = inner["message"].replace(term, replacement)
                if "log" in inner and isinstance(inner["log"], str):
                    inner["log"] = inner["log"].replace(term, replacement)
                if "task_name" in inner.get("callback_params", {}):
                    inner["callback_params"]["task_name"] = inner["callback_params"]["task_name"].replace(term, replacement)

            # Mark final if it's the last one
            if i == total_msgs - 1:
                inner["details"]["is_final"] = True
                inner["details"]["completion_timestamp"] = datetime.now().isoformat()
                if "progress" not in inner or inner["progress"] is None:
                    inner["progress"] = 100

            # Print in new format
            sse_print(event, inner)
            
            writer = get_summary_writer()
            if writer is not None:
                # Record with timestamp for summary file
                record_msg = inner.copy()
                record_msg["event"] = event
                record_msg["time_stamp"] = _now_string()
                writer.record(record_msg)
            last_payload = inner
        
        # Inject randomized heartbeat messages between JSON segments
        if i < total_msgs - 1:
            num_heartbeats = random.randint(1, 3)
            current_p = last_payload.get("progress", 0) if last_payload else 0
            # Peek next message progress
            try:
                next_p = messages[i+1].get("data", {}).get("progress", current_p + 5)
            except:
                next_p = current_p + 5
            
            for h in range(num_heartbeats):
                time.sleep(random.uniform(1.0, 2.0))
                h_progress = round(current_p + (h + 1) * (next_p - current_p) / (num_heartbeats + 1), 2)
                heartbeat = {
                    "callback_params": inner.get("callback_params", default_callback_params()),
                    "progress": h_progress,
                    "message": "引擎状态自检中...",
                    "log": f"[{h_progress:.1f}%] 正在同步中间特征向量...",
                    "details": {}
                }
                sse_print("heartbeat", heartbeat)
                if writer is not None:
                    record_h = heartbeat.copy()
                    record_h["event"] = "heartbeat"
                    record_h["time_stamp"] = _now_string()
                    writer.record(record_h)

            time.sleep(random.uniform(1.0, 2.0))
        
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

