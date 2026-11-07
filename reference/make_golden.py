"""Write the golden test data in tests/golden/ from the Python reference.

    python reference/make_golden.py            (any Python with torch >= 2.0 and numpy; CPU is enough)

Every case is deterministic (fixed seeds, CPU). Re-run it after changing anything in otsr_ref/ and commit the result.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from otsr_ref import edges as E  # noqa: E402
from otsr_ref import fileformat  # noqa: E402
from otsr_ref import fit as FIT  # noqa: E402
from otsr_ref import init as INIT  # noqa: E402
from otsr_ref import render as R  # noqa: E402
from otsr_ref import spec  # noqa: E402
from otsr_ref.goldenio import write_case  # noqa: E402

OUT = HERE.parent / "tests" / "golden"


def pack(p: dict) -> torch.Tensor:
    """dict -> (N, 9) in PARAM_LAYOUT order (the engine's per-Gaussian layout)."""
    return torch.cat([p[k].reshape(p[k].shape[0], n) for k, n in spec.PARAM_LAYOUT], 1)


def unpack(x: torch.Tensor) -> dict:
    out, o = {}, 0
    for k, n in spec.PARAM_LAYOUT:
        out[k] = x[:, o:o + n]
        o += n
    return out


def hwc(img: torch.Tensor) -> torch.Tensor:
    return img.permute(1, 2, 0).contiguous()


def synthetic_image(H: int, W: int, seed: int, ss: int = 4) -> torch.Tensor:
    """(3, H, W) float32: a smooth background with anti-aliased disks, a rotated box and a thin line."""
    g = torch.Generator().manual_seed(seed)
    U = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)  # noqa: E731
    yy, xx = torch.meshgrid((torch.arange(H * ss, dtype=torch.float64) + 0.5) / ss,
                            (torch.arange(W * ss, dtype=torch.float64) + 0.5) / ss, indexing="ij")
    c0, c1 = U(3, 1, 1), U(3, 1, 1)
    img = c0 + (c1 - c0) * (xx / W) + 0.08 * torch.sin(yy / H * 2 * math.pi * (1 + 2 * U(3, 1, 1)))
    m = min(H, W)
    for _ in range(4):
        cx, cy, r = U(1) * W, U(1) * H, (0.08 + 0.2 * U(1)) * m
        img = torch.where(((xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2)[None], U(3, 1, 1), img)
    cx, cy, th = U(1) * W, U(1) * H, U(1) * math.pi
    hx, hy = (0.1 + 0.2 * U(1)) * W, (0.05 + 0.1 * U(1)) * H
    u = (xx - cx) * torch.cos(th) + (yy - cy) * torch.sin(th)
    v = -(xx - cx) * torch.sin(th) + (yy - cy) * torch.cos(th)
    img = torch.where(((u.abs() < hx) & (v.abs() < hy))[None], U(3, 1, 1), img)
    nx_, ny_ = math.cos(0.3 + U(1).item()), math.sin(0.3 + U(1).item())
    dist = ((xx - W / 2) * nx_ + (yy - H / 2) * ny_).abs()
    img = torch.where((dist < 0.6)[None], U(3, 1, 1) * 0.2, img)
    return F.avg_pool2d(img[None], ss)[0].clamp(0, 1).float()


def random_params(N: int, H: int, W: int, seed: int, dtype=torch.float64) -> dict:
    """Varied Gaussians: mostly 0.3-4 px, 10 % large (8-20 px), 3 % at the 0.2 px clamp, some outside the frame."""
    g = torch.Generator().manual_seed(seed)
    U = lambda *s: torch.rand(*s, generator=g, dtype=dtype)  # noqa: E731
    L = max(H, W)
    s_px = 0.3 * (4 / 0.3) ** U(N, 2)
    big = U(N) < 0.1
    s_px[big] = 8 + 12 * U(int(big.sum()), 2)
    tiny = U(N) < 0.03
    s_px[tiny] = 0.2
    log_w = (U(N, 1) * 2 - 1) * 2
    log_w[U(N) < 0.02] = 6.0
    return {"uv": U(N, 2) * 1.04 - 0.02, "log_s": torch.log(s_px / L), "rot": (U(N, 1) * 2 - 1) * math.pi,
            "color": U(N, 3) * 1.3 - 0.15, "log_w": log_w}


def case_spec():
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.as_dict().items()}
    d["PARAM_LAYOUT"] = [[k, n] for k, n in spec.PARAM_LAYOUT]
    d["LR_SCALE_SAMPLES"] = [[it, 100, spec.lr_scale(it, 100)] for it in (0, 1, 25, 50, 99, 100)]
    d["N_LONG_SAMPLES"] = [[h, w, spec.n_long_for(h, w)] for h, w in ((45, 61), (512, 768), (1080, 1920),
                                                                      (2160, 3840), (300, 200))]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spec.json").write_text(json.dumps(d, indent=1) + "\n")


def case_render():
    H, W, N = 45, 61, 900
    p = random_params(N, H, W, seed=1)
    gt = synthetic_image(H, W, seed=2).double()
    q = {k: v.clone().requires_grad_(True) for k, v in p.items()}
    out, D = R.render(q, H, W, with_den=True)
    loss = R.mse(out, gt)
    loss.backward()
    grad = pack({k: v.grad for k, v in q.items()})
    mx, my, a, b, d, w = R.gaussian_terms(p, H, W)
    rx, ry = R.support_box(p, H, W)
    view = {"zoom": 2.5, "tx": -7.3, "ty": 4.1, "out_w": 100, "out_h": 70}
    with torch.no_grad():
        out_v = R.render(p, H, W, out_h=view["out_h"], out_w=view["out_w"], zoom=view["zoom"], tx=view["tx"],
                         ty=view["ty"])
    write_case(OUT / "render", {
        "params": pack(p), "gt": hwc(gt), "out": hwc(out.detach()), "D": D.detach(), "loss": loss.detach().view(1),
        "grad": grad, "terms": torch.stack([mx, my, a, b, d, w], 1), "support": torch.stack([rx, ry], 1),
        "out_view": hwc(out_v)}, {"H": H, "W": W, "N": N, "view": view, "dtype": "float64 reference"})


def case_adam():
    H, W, N, steps = 30, 50, 64, 10
    L = max(H, W)
    p = random_params(N, H, W, seed=11)
    p["uv"][:4] = torch.tensor([[-0.0199, 0.5], [1.0199, 0.5], [0.5, -0.0199], [0.5, 1.0199]], dtype=torch.float64)
    p["log_s"][4:6] = math.log(0.2 / L) + 1e-4
    p["log_s"][6:8] = spec.LOG_S_MAX - 1e-4
    p["log_w"][8:10] = torch.tensor([[7.9999], [-7.9999]], dtype=torch.float64)
    g = torch.Generator().manual_seed(12)
    grads = (torch.rand(3, N, 9, generator=g, dtype=torch.float64) * 2 - 1) * \
        10 ** (-7 + 4 * torch.rand(3, N, 9, generator=g, dtype=torch.float64))
    grads[:, :12] = torch.sign(grads[:, :12]) * 1e-3            # push the clamped ones over their bounds
    grads[:, 12:14] = 0.0                                       # zero gradients still move through momentum
    q = {k: v.clone().requires_grad_(True) for k, v in p.items()}
    opt = FIT.make_optimizer(q, L)
    base = [gr["lr"] for gr in opt.param_groups]
    after, ms, vs = [], [], []
    for it in range(3):
        for gr, b0 in zip(opt.param_groups, base):
            gr["lr"] = b0 * spec.lr_scale(it, steps)
        gd = unpack(grads[it])
        for k in q:
            q[k].grad = gd[k].clone()
        opt.step()
        FIT.clamp_(q, L)
        after.append(pack({k: v.detach() for k, v in q.items()}).clone())
        ms.append(pack({k: opt.state[q[k]]["exp_avg"] for k in q}).clone())
        vs.append(pack({k: opt.state[q[k]]["exp_avg_sq"] for k in q}).clone())
    write_case(OUT / "adam", {"params0": pack(p), "grads": grads, "params": torch.stack(after), "m": torch.stack(ms),
                              "v": torch.stack(vs)}, {"H": H, "W": W, "N": N, "steps_total": steps, "iters": 3})


def case_fit():
    H, W, steps = 40, 48, 40
    gt = synthetic_image(H, W, seed=3)
    p0 = INIT.ot_init(gt, K=180)
    p0d = {k: v.double() for k, v in p0.items()}
    t0 = time.time()
    p, curve = FIT.fit(gt.double(), p0d, steps=steps, log_every=10)
    print(f"  fit: {len(curve)} log points, {time.time() - t0:.1f} s, PSNR {curve[0]['psnr']:.3f} -> "
          f"{curve[-1]['psnr']:.3f} dB")
    write_case(OUT / "fit", {"gt": hwc(gt), "params0": pack(p0d), "params": pack(p),
                             "psnr": torch.tensor([c["psnr"] for c in curve]),
                             "loss": torch.tensor([c["loss"] for c in curve])},
               {"H": H, "W": W, "N": int(p0["uv"].shape[0]), "steps": steps, "log_steps": [c["step"] for c in curve]})


def cov_of(p: dict) -> torch.Tensor:
    """(N, 3) covariance (xx, xy, yy) in long-side units^2: rotation-sign agnostic comparison of shapes."""
    s2 = torch.exp(2 * p["log_s"])
    c, sn = torch.cos(p["rot"][:, 0]), torch.sin(p["rot"][:, 0])
    return torch.stack([c * c * s2[:, 0] + sn * sn * s2[:, 1], c * sn * (s2[:, 1] - s2[:, 0]),
                        sn * sn * s2[:, 0] + c * c * s2[:, 1]], 1)


def case_ot(name: str, H: int, W: int, seed: int, K=None, n_long=None):
    gt = synthetic_image(H, W, seed=seed)
    t0 = time.time()
    p, dbg = INIT.ot_init(gt, K=K, n_long=n_long, debug=True)
    L = max(H, W)
    T64, _ = INIT.transport_map(dbg["rho"], W / L, H / L, dtype=torch.float64)       # the precise map
    err32 = (dbg["T"] - T64).abs()
    print(f"  {name}: grid {tuple(dbg['rho'].shape)}, {dbg['gy']}x{dbg['gx']} Gaussians, {dbg['iters']} Sinkhorn "
          f"iterations, {time.time() - t0:.1f} s; float32 map vs float64: max {err32.max():.2e}, "
          f"mean {err32.mean():.2e} box units")
    write_case(OUT / name, {"gt": hwc(gt), "rho": dbg["rho"], "T": dbg["T"], "T64": T64, "params": pack(p),
                            "cov": cov_of(p), "J": dbg["J"]},
               {"H": H, "W": W, "K": K, "n_long": dbg["n_long"], "gy": dbg["gy"], "gx": dbg["gx"],
                "lam": dbg["lam"], "eps": dbg["eps"], "iters": dbg["iters"]})


def case_edges():
    base = synthetic_image(40, 52, seed=5)
    img = F.interpolate(base[None], scale_factor=4, mode="bilinear", align_corners=False)[0].clamp(0, 1)
    runs = [{"S": 4.0, "edge_px": 1.4}, {"S": 2.5, "edge_px": 0.8}, {"S": 8.0, "edge_px": 2.5}]
    arrays = {"img": hwc(img)}
    for i, r in enumerate(runs):
        arrays[f"out{i}"] = hwc(E.edge_field(img, r["S"], r["edge_px"]))
    write_case(OUT / "edges", arrays, {"H": img.shape[1], "W": img.shape[2], "runs": runs})


def case_format():
    p = random_params(5, 20, 30, seed=21, dtype=torch.float32)
    fileformat.save(OUT / "sample.otsr", p, 30, 20, {"generator": "otsr-ref 0.1.0", "note": "golden sample"})
    write_case(OUT / "sample", {"params": pack(p)}, {"W": 30, "H": 20, "N": 5,
                                                     "meta": {"generator": "otsr-ref 0.1.0", "note": "golden sample"}})


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.time()
    case_spec()
    case_render()
    case_adam()
    case_fit()
    case_ot("ot", 120, 160, seed=4)
    case_ot("ot_tall", 100, 70, seed=6, K=400, n_long=64)
    case_edges()
    case_format()
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"wrote {len(list(OUT.iterdir()))} files ({total / 1e6:.2f} MB) to {OUT} in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
