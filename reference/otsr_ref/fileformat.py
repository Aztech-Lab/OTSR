"""Read and write ``.otsr`` model files (format v1.0, specified in docs/file-format.md).

Layout, all little-endian:

    offset  size      field
    0       4         magic b"OTSR"
    4       2         version major (1)
    6       2         version minor (0)
    8       4         width W of the source image (u32)
    12      4         height H (u32)
    16      4         Gaussian count N (u32)
    20      4         flags (u32, 0; reserved)
    24      4         metadata length M in bytes (u32)
    28      4         reserved (u32, 0)
    32      M         metadata, UTF-8 JSON object (may be empty: M = 0)
    ...     0-3       zero padding to a multiple of 4
    ...     36 N      float32 payload, one C-contiguous block per parameter in PARAM_LAYOUT order:
                      uv (N, 2), log_s (N, 2), rot (N, 1), color (N, 3), log_w (N, 1)
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .spec import PARAM_LAYOUT

MAGIC = b"OTSR"
VERSION = (1, 0)
HEADER = struct.Struct("<4sHHIIIIII")


def save(path, params: dict, W: int, H: int, meta: dict | None = None):
    """params: dict of arrays / tensors as in PARAM_LAYOUT (any float dtype)."""
    arrs = []
    N = None
    for name, n in PARAM_LAYOUT:
        a = params[name]
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        a = np.ascontiguousarray(a.reshape(-1, n), dtype="<f4")
        N = a.shape[0] if N is None else N
        if a.shape[0] != N:
            raise ValueError(f"{name}: {a.shape[0]} rows, expected {N}")
        arrs.append(a)
    m = json.dumps(meta or {}, separators=(",", ":")).encode("utf-8") if meta else b""
    pad = (-(HEADER.size + len(m))) % 4
    with open(path, "wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION[0], VERSION[1], W, H, N, 0, len(m), 0))
        f.write(m + b"\0" * pad)
        for a in arrs:
            f.write(a.tobytes())


def load(path):
    """-> (params dict of float32 numpy arrays, W, H, meta dict)."""
    buf = Path(path).read_bytes()
    magic, vmaj, vmin, W, H, N, flags, mlen, _ = HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("not an .otsr file")
    if vmaj != VERSION[0]:
        raise ValueError(f"unsupported .otsr version {vmaj}.{vmin}")
    meta = json.loads(buf[HEADER.size:HEADER.size + mlen].decode("utf-8")) if mlen else {}
    off = HEADER.size + mlen + (-(HEADER.size + mlen)) % 4
    params = {}
    for name, n in PARAM_LAYOUT:
        params[name] = np.frombuffer(buf, dtype="<f4", count=N * n, offset=off).reshape(N, n).copy()
        off += 4 * N * n
    if off != len(buf):
        raise ValueError(f"size mismatch: payload ends at {off}, file has {len(buf)} bytes")
    return params, W, H, meta
