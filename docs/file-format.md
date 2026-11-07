# The `.otsr` model file (format 1.0)

A fitted model: the source image size, N Gaussians, optional metadata. Little-endian throughout.

| offset | size | field |
|---|---|---|
| 0 | 4 | magic `OTSR` (0x4F 0x54 0x53 0x52) |
| 4 | 2 | version major = 1 (u16) |
| 6 | 2 | version minor = 0 (u16) |
| 8 | 4 | width W of the source image (u32) |
| 12 | 4 | height H (u32) |
| 16 | 4 | Gaussian count N (u32) |
| 20 | 4 | flags (u32), 0 — reserved |
| 24 | 4 | metadata length M in bytes (u32) |
| 28 | 4 | reserved (u32), 0 |
| 32 | M | metadata: a UTF-8 JSON object, or nothing (M = 0) |
| 32 + M | 0–3 | zero padding to a multiple of 4 |
| … | 36 N | float32 payload, **planar**: `uv` (N × 2), `log_s` (N × 2), `rot` (N), `color` (N × 3), `log_w` (N) |

The parameters are those of method.md §1: `uv` in [0, 1]² (x, y), `log_s` in long-side units, `rot` in radians with
the kernel's rotation convention, `color` RGB as fitted (nominally [0, 1], not clamped), `log_w` the log weight.
The file size is 32 + M (+ padding) + 36 N bytes; a 2048 × 2048 image at 0.1 Gaussians per pixel gives ≈ 15 MB.

Metadata written by the web app (all optional): `generator`, `source` (file name), `width`, `height`, `count`,
`steps`, `psnr`, `seconds`, `adapter`. Readers must ignore unknown keys. A minor version bump adds fields to the
metadata or uses the flags; a major bump changes the payload.

Readers / writers: `src/core/fileformat.ts` (browser), `reference/otsr_ref/fileformat.py` (Python; returns numpy
arrays). `tests/golden/sample.otsr` is a 5-Gaussian sample written by Python and checked by the TypeScript tests
byte for byte.

Rendering a model outside OTSR only needs the image function of method.md §1 — `reference/otsr_ref/render.py`
(60 lines of PyTorch) is a complete renderer.
