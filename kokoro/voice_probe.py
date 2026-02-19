from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict
import json
import os

import torch

_LOCK = Lock()
_TRUE_VALUES = {"1", "true", "yes", "on"}


def probe_enabled() -> bool:
    return os.getenv("KOKORO_VOICE_PROBE", "").strip().lower() in _TRUE_VALUES


def probe_path() -> Path:
    raw = os.getenv("KOKORO_VOICE_PROBE_PATH", "logs/voice_probe.jsonl").strip()
    return Path(raw or "logs/voice_probe.jsonl")


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    x = t.detach().to("cpu")
    stats: Dict[str, Any] = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "numel": int(x.numel()),
    }
    if x.numel() == 0:
        return stats
    xf = x.float()
    finite = torch.isfinite(xf)
    finite_count = int(finite.sum().item())
    stats["finite_ratio"] = finite_count / int(xf.numel())
    if finite_count == 0:
        return stats
    xv = xf[finite]
    stats.update(
        {
            "min": _safe_float(xv.min()),
            "max": _safe_float(xv.max()),
            "mean": _safe_float(xv.mean()),
            "std": _safe_float(xv.std(unbiased=False)),
            "l2": _safe_float(torch.norm(xv)),
            "absmax": _safe_float(xv.abs().max()),
        }
    )
    return stats


def voice_pack_stats(pack: torch.Tensor) -> Dict[str, Any]:
    x = pack.detach().to("cpu")
    out: Dict[str, Any] = tensor_stats(x)
    if x.ndim >= 2 and x.shape[-1] == 256:
        y = x.reshape(-1, 256).float()
        norms = torch.norm(y, dim=-1)
        out["row_norm"] = {
            "min": _safe_float(norms.min()),
            "max": _safe_float(norms.max()),
            "mean": _safe_float(norms.mean()),
            "std": _safe_float(norms.std(unbiased=False)),
        }
        left = y[:, :128]
        right = y[:, 128:]
        out["split_l2"] = {
            "left_mean": _safe_float(torch.norm(left, dim=-1).mean()),
            "right_mean": _safe_float(torch.norm(right, dim=-1).mean()),
        }
    return out


def emit_probe(event: str, **payload: Any) -> None:
    if not probe_enabled():
        return
    dst = probe_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "pid": os.getpid(),
        **payload,
    }
    with _LOCK:
        with dst.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")
