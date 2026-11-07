"""Golden test data: ``<case>.json`` (manifest) + ``<case>.bin`` (raw little-endian arrays, 16-byte aligned).

Manifest: {"meta": {...}, "arrays": {name: {"dtype": "float32" | "int32" | "uint32", "shape": [...],
"offset": bytes, "count": elements}}}. Read by tests/helpers/golden.ts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DTYPES = {"float32": "<f4", "int32": "<i4", "uint32": "<u4"}


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def write_case(stem: Path, arrays: dict, meta: dict | None = None):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    index = {}
    for name, x in arrays.items():
        a = _np(x)
        kind = "float32" if a.dtype.kind == "f" else ("uint32" if a.dtype.kind == "u" else "int32")
        if a.dtype.kind == "b":
            kind = "uint32"
        a = np.ascontiguousarray(a, dtype=DTYPES[kind])
        blob += b"\0" * ((-len(blob)) % 16)
        index[name] = {"dtype": kind, "shape": list(a.shape), "offset": len(blob), "count": int(a.size)}
        blob += a.tobytes()
    stem.with_suffix(".bin").write_bytes(bytes(blob))
    stem.with_suffix(".json").write_text(json.dumps({"meta": meta or {}, "arrays": index}, indent=1) + "\n")


def read_case(stem: Path):
    stem = Path(stem)
    man = json.loads(stem.with_suffix(".json").read_text())
    buf = stem.with_suffix(".bin").read_bytes()
    out = {}
    for name, e in man["arrays"].items():
        out[name] = np.frombuffer(buf, DTYPES[e["dtype"]], e["count"], e["offset"]).reshape(e["shape"])
    return out, man["meta"]
