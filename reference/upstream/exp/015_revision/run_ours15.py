"""015: our per-image fitting on any image set / resolution / aspect ratio, with the switches of the ablations.

  layout (--inits)  random       i.i.d. uniform positions, isotropic, colours sampled from the image
                    random_rc    the same with random colours (GaussianImage-style start)
                    lattice      uniform lattice (= OT to the uniform density)
                    importance   Image-GS sampling, p ~ lam + (1-lam)|grad I| per pixel (jittered)
                    imp_pooled   i.i.d. sampling from exactly the OT target density (pooled grid)
                    kr           Knothe-Rosenblatt map of the lattice to that density (exact marginals, sheared cells)
                    ot           OT lattice (positions + shapes from the Jacobian)
                    ot_iso       OT positions, isotropic Gaussians of the same area
                    ot_cap<r>    OT positions and Jacobian shapes, axis ratio capped at r (same area), e.g. ot_cap4
                    ot_relayout  ot, --s1 steps, error-driven second OT, fresh attributes, remaining steps
                    imp_relayout importance, --s1 steps, i.i.d. resampling from the same error-driven density
                                 (the control for the re-layout: same density, sampled instead of transported)
  blending (--mode) wnorm (ours) | sum (GaussianImage) | topk (Image-GS)
  gradients         --grads exact (ours) | kernel (the gmod projection backward, wrong for scale / rotation)
  optimiser         --opt ours (per-parameter Adam learning rates, cosine) | single (Adam, lr 1e-3 for all, cosine)
                    | frozen (ours, but the positions never move: tests the layout itself)
Budget: --budget d10k / d50k (Gaussians per pixel of the 512^2 protocol x H x W), or K=<n>[,<n>], or
--K-from rows.jsonl (per image: K, or params / 9 for Bezier rows). Time includes the layout computation.
Rows -> <out>/rows.jsonl (resumable on set, image, K, config).

  python -u Agent/cc/exp/015_revision/run_ours15.py --out .../runs/x --images kodak:24 --budget d10k --inits ot
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CC = HERE.parents[1]
sys.path.insert(0, str(CC))
sys.path.insert(0, str(CC / "exp" / "011_ot_layout"))
sys.path.insert(0, str(HERE))

from common import render as R  # noqa: E402
from common.gauss import fit, init_params, lattice_dims, lattice_uv, render_params, sample_color  # noqa: E402
from common.ot import kr_map, lam_for_spacing, transport_map_f32 as transport_map  # noqa: E402
from ot_layout_test import imp_pooled_layout, ot_params_from_map, pooled_density, residual_monitor  # noqa: E402
import protocol15 as P  # noqa: E402

T_CHECK = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
S_CHECK = (0, 100, 300, 1000, 2000, 3000)


def cap_shapes(p, r):
    """Cap the axis ratio of the Jacobian shapes at r (log-area kept); r <= 1 leaves them unchanged."""
    if r is None or r <= 1:
        return p
    lr = math.log(r)
    m = p["log_s"].mean(1, keepdim=True)
    d = (p["log_s"][:, :1] - p["log_s"][:, 1:]).clamp(-lr, lr) / 2
    p["log_s"] = torch.cat([m + d, m - d], 1)
    return p


def make_init(gt, K, init, lam, mode, n_long=128, seed=0, ot_cap=0.0):
    """-> (params, rho or None, (gy, gx) or None)"""
    _, H, W = gt.shape
    L = max(H, W)
    dev = gt.device
    if init in ("importance", "imp_relayout"):
        p = init_params(gt, K, mode, how="importance", seed=seed)
        return p, (pooled_density(gt, lam, n_long) if init == "imp_relayout" else None), None
    if init == "lattice":
        return init_params(gt, K, mode, how="lattice"), None, None
    if init in ("random", "random_rc"):
        g = torch.Generator(device="cpu").manual_seed(seed)
        uv = torch.rand(K, 2, generator=g).to(dev)
        s = 0.6 * math.sqrt(H * W / K) / L
        col = torch.rand(K, 3, generator=g).to(dev) if init == "random_rc" else sample_color(gt, uv)
        p = {"uv": uv, "log_s": torch.full((K, 2), math.log(s), device=dev), "rot": torch.zeros(K, 1, device=dev),
             "color": col, "log_w": torch.zeros(K, 1, device=dev)}
    elif init == "imp_pooled":
        p = imp_pooled_layout(gt, K, lam, n_long, seed=seed)
    elif init in ("cvt", "cvt_cov"):
        p = cvt_layout(gt, K, lam, n_long, seed, shapes="cov" if init == "cvt_cov" else "iso")
    else:                                                  # kr, ot, ot_iso, ot_relayout
        gy, gx = lattice_dims(K, H, W)
        rho = pooled_density(gt, lam, n_long)
        T = kr_map(rho, W / L, H / L) if init == "kr" else transport_map(rho, W / L, H / L)[0]
        p = ot_params_from_map(gt, gy, gx, T, W / L, H / L)[0]
        if init == "ot_iso":
            p["log_s"] = p["log_s"].mean(1, keepdim=True).repeat(1, 2)
            p["rot"] = torch.zeros_like(p["rot"])
        if init.startswith("ot_cap"):                       # log s1 - log s2 <= log r, log-area kept
            p = cap_shapes(p, float(init[6:]))
        elif init in ("ot", "ot_relayout"):
            p = cap_shapes(p, ot_cap)
        if mode == "sum":
            p["color"] = p["color"] / (2 * math.pi * 0.36)
        return p, rho, (gy, gx)
    if mode == "sum":
        p["color"] = p["color"] / (2 * math.pi * 0.36)
    return p, None, None


_CVT = {}
CVT_WORKERS = -1


def cvt_layout(gt, K, lam, n_long, seed, shapes="iso", iters=30, ss=2):
    """Weighted Lloyd relaxation (centroidal Voronoi tessellation) of i.i.d. samples of the pooled density. The weights
    are rho^2 on a ss x ss supersampled pixel grid, since the point density of a weighted CVT in 2D follows weight^(1/2):
    the points then follow rho, like the OT lattice and the samples. shapes: 'iso' (0.6 sqrt(cell area)) or 'cov'
    (the cell's second moment, scaled so that a square cell of side a gets sigma = 0.6 a, as the lattice)."""
    from scipy.spatial import cKDTree
    _, H, W = gt.shape
    L = max(H, W)
    key = (id(gt), K, seed, n_long, lam)
    if key not in _CVT:
        _CVT.clear()
        rho = pooled_density(gt, lam, n_long)
        ny, nx = rho.shape
        ys, xs = np.mgrid[0:H * ss, 0:W * ss]
        pts = np.stack([(xs + 0.5) / ss, (ys + 0.5) / ss], -1).reshape(-1, 2)          # pixel units
        r = rho.detach().float().cpu().numpy()
        w = (r[np.minimum(ys * ny // (H * ss), ny - 1), np.minimum(xs * nx // (W * ss), nx - 1)] ** 2).reshape(-1)
        sites = sample_pooled(gt, K, rho, seed)["uv"].cpu().numpy() * np.array([W, H])
        for _ in range(iters):
            lab = cKDTree(sites).query(pts, workers=CVT_WORKERS)[1]
            sw = np.bincount(lab, weights=w, minlength=K)
            ok = sw > 0
            for d in (0, 1):
                sites[ok, d] = np.bincount(lab, weights=w * pts[:, d], minlength=K)[ok] / sw[ok]
        lab = cKDTree(sites).query(pts, workers=CVT_WORKERS)[1]
        n = np.bincount(lab, minlength=K).astype(np.float64)                          # samples per cell
        m1 = [np.bincount(lab, weights=pts[:, d], minlength=K) for d in (0, 1)]
        m2 = [np.bincount(lab, weights=pts[:, a] * pts[:, b], minlength=K) for a, b in ((0, 0), (0, 1), (1, 1))]
        _CVT[key] = (sites, n, m1, m2)
    sites, n, m1, m2 = _CVT[key]
    area = np.maximum(n, 1.0) / ss ** 2                                                # pixels
    s_iso = np.maximum(0.6 * np.sqrt(area), 0.35)
    s1, s2, th = s_iso.copy(), s_iso.copy(), np.zeros(K)
    if shapes == "cov":
        big = n >= 6
        nn = np.maximum(n, 1.0)
        mx, my = m1[0] / nn, m1[1] / nn
        cxx = m2[0] / nn - mx ** 2 + 1 / (12 * ss ** 2)
        cxy = m2[1] / nn - mx * my
        cyy = m2[2] / nn - my ** 2 + 1 / (12 * ss ** 2)
        tr, det = cxx + cyy, np.maximum(cxx * cyy - cxy ** 2, 1e-12)
        disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0.0))
        l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-12)
        a = np.sqrt(12 * 0.36)                                                          # sigma = 0.6 a for side a
        s1 = np.where(big, np.maximum(a * np.sqrt(l1), 0.35), s_iso)
        s2 = np.where(big, np.maximum(a * np.sqrt(l2), 0.35), s_iso)
        th = np.where(big, -0.5 * np.arctan2(2 * cxy, cxx - cyy), 0.0)    # kernel angle: first axis (c, -s)
    dev = gt.device
    uv = torch.from_numpy(sites / np.array([W, H])).float().to(dev)
    log_s = torch.log(torch.from_numpy(np.stack([s1, s2], 1) / L).float().to(dev))
    return {"uv": uv, "log_s": log_s, "rot": torch.from_numpy(th).float().to(dev)[:, None],
            "color": sample_color(gt, uv), "log_w": torch.zeros(K, 1, device=dev)}


def sample_pooled(gt, K, rho, seed):
    """K i.i.d. positions from a pooled density (cell by rho, uniform inside), isotropic by local spacing."""
    _, H, W = gt.shape
    L = max(H, W)
    ny, nx = rho.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.multinomial(rho.flatten().float().cpu(), K, replacement=True, generator=g).to(gt.device)
    r, c = idx // nx, idx % nx
    jit = torch.rand(K, 2, generator=g).to(gt.device)
    uv = torch.stack([(c.float() + jit[:, 0]) / nx, (r.float() + jit[:, 1]) / ny], -1)
    sp = torch.sqrt((H / ny) * (W / nx) / (K * rho.flatten()[idx].float()))
    s = (0.6 * sp / L).clamp(min=0.35 / L).unsqueeze(1).repeat(1, 2)
    return {"uv": uv, "log_s": torch.log(s), "rot": torch.zeros(K, 1, device=gt.device),
            "color": sample_color(gt, uv), "log_w": torch.zeros(K, 1, device=gt.device)}


def opt_kwargs(opt, L):
    if opt == "ours":
        return {}
    if opt == "frozen":                    # positions fixed at the layout; shapes, colours, weights fitted
        return dict(lr_px=0.0)
    if opt == "single":                    # one Adam learning rate for every parameter group (uv in [0,1] units)
        return dict(lr_px=1e-3 * L, lr_logs=1e-3, lr_rot=1e-3, lr_col=1e-3, lr_logw=1e-3)
    raise ValueError(opt)


def checkpoints(curve):
    """curve: [(time_s, psnr, step)]"""
    out = {f"psnr_{t:g}s": P.psnr_at([(a, b) for a, b, _ in curve], t) for t in T_CHECK}
    for s in S_CHECK:
        v = [b for a, b, st in curve if st <= s]
        out[f"psnr_step{s}"] = v[-1] if v else None
    return out


def save_params(path, **ps):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{n}_{k}": v.detach().float().cpu().numpy() for n, p in ps.items()
                                 for k, v in p.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--budget", default="d10k")
    ap.add_argument("--K-from", default="")
    ap.add_argument("--inits", default="ot_relayout")
    ap.add_argument("--mode", default="wnorm", choices=["wnorm", "sum", "topk"])
    ap.add_argument("--grads", default="exact", choices=["exact", "kernel"])
    ap.add_argument("--opt", default="ours", choices=["ours", "single", "frozen"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--s1", type=int, default=300)
    ap.add_argument("--n-long", default="128", help="transport grid cells on the long side, or 'auto' = "
                    "clip(L / 4, 128, 256): ~4 px cells up to 1024 px (007: 256 > 128 at native 1-2K, 512 unstable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hr-radius", type=float, default=5 / 3, help="culling radius multiplier (x 3 sigma) for the S x renders of sr_x<S>/... sets; the fit itself always culls at 3 sigma")
    ap.add_argument("--aa-px", type=float, default=0.0, help="pixel-footprint std (output pixels) in fitting and rendering; sets sr_x<S>/... are also rendered at S x their size")
    ap.add_argument("--ot-cap", type=float, default=0.0, help="cap the axis ratio of OT shapes (0 = none)")
    ap.add_argument("--save-params", default="", help="comma list of sets (or 'all') whose params are saved")
    ap.add_argument("--save-renders", default="", help="comma list of sets (or 'all') whose renders are saved")
    ap.add_argument("--snap-steps", default="", help="steps (multiples of 25) at which params are added to the npz")
    args = ap.parse_args()
    dev = "cuda:0"
    from common.gpu import guard
    guard()
    R.KERNEL_GRADS = args.grads == "kernel"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    cfg = {"mode": args.mode, "grads": args.grads, "opt": args.opt, "steps": args.steps, "n_long": args.n_long,
           "ot_cap": args.ot_cap,
           "seed": args.seed, "aa_px": args.aa_px}
    done = P.done_keys(rows_path, ("set", "image", "K", "init"))
    k_of = {}
    if args.K_from:
        for line in open(args.K_from, encoding="utf-8"):
            r = json.loads(line)
            if r.get("failed"):
                continue
            tag = f"{r['mode']}{r['curves']}" if r.get("curves") is not None else None     # Bezier: closed512 ...
            k_of[(r["set"], r["image"], tag)] = int(r["K"]) if "K" in r else int(round(r["params"] / 9))
    sp = set(args.save_params.split(",")) if args.save_params else set()
    sr = set(args.save_renders.split(",")) if args.save_renders else set()
    warm = True
    for s, f in P.images(args.images):
        st = P.stem(f)
        gt = P.load_gt(f, dev)
        _, H, W = gt.shape
        L = max(H, W)
        nl = min(256, max(128, round(L / 4))) if args.n_long == "auto" else int(args.n_long)
        if warm:                   # untimed warm-up of the OT solver and the renderer
            for n in sorted({64, 128, 256, nl}):
                transport_map(pooled_density(gt, 0.3, n), W / L, H / L)
            fit(gt, init_params(gt, 4096, args.mode, how="importance"), mode=args.mode, steps=20, log_every=10)
            torch.cuda.synchronize()
            warm = False
        if k_of:
            Ks = sorted({(k, c) for (ss, im, c), k in k_of.items() if ss == s and im == st}, key=lambda x: x[0])
        elif args.budget.startswith("K="):
            Ks = [(int(k), None) for k in args.budget[2:].split(",")]
        else:
            Ks = [(P.budget(H, W, args.budget), None)]
        for K, curves in Ks:
            gy, gx = lattice_dims(K, H, W)
            lam = lam_for_spacing(math.sqrt(H * W / (gy * gx)))
            for init in args.inits.split(","):
                if (s, st, K, init) in done:
                    continue
                torch.cuda.synchronize()
                t0 = time.time()
                p0, rho, grid = make_init(gt, K, init, lam, args.mode, nl, args.seed, args.ot_cap)
                torch.cuda.synchronize()
                t_init = time.time() - t0
                okw = opt_kwargs(args.opt, L)
                okw["aa_px"] = args.aa_px
                snaps = {}
                want = {int(x) for x in args.snap_steps.split(",")} if args.snap_steps else set()

                def snap(off):
                    def cb(it, p_, out_):
                        if off + it in want and off + it not in snaps:
                            snaps[off + it] = {k: v.detach().clone() for k, v in p_.items()}
                    return cb if want else None
                if init in ("ot_relayout", "imp_relayout"):
                    pa, ca, sa = fit(gt, p0, mode=args.mode, steps=args.s1, log_every=25, callback=snap(0), **okw)
                    with torch.no_grad():
                        rend = render_params(pa, H, W, args.mode, aa_px=args.aa_px).clamp(0, 1)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    rho2 = residual_monitor(gt, rend, rho)
                    if init == "ot_relayout":
                        T2, _ = transport_map(rho2, W / L, H / L)
                        p2 = cap_shapes(ot_params_from_map(gt, grid[0], grid[1], T2, W / L, H / L)[0], args.ot_cap)
                    else:
                        p2 = sample_pooled(gt, K, rho2, args.seed + 1)
                    if args.mode == "sum":
                        p2["color"] = p2["color"] / (2 * math.pi * 0.36)
                    torch.cuda.synchronize()
                    t_re = time.time() - t1
                    if want:
                        snaps["relayout"] = {k: v.detach().clone() for k, v in p2.items()}
                        if args.s1 in snaps:                # end of phase 1; step s1 = the re-laid-out start
                            snaps["prerelayout"] = snaps.pop(args.s1)
                    p, cb, sb = fit(gt, p2, mode=args.mode, steps=args.steps - args.s1, log_every=25,
                                    callback=snap(args.s1), **okw)
                    curve = [(t_init + c["time_s"], c["psnr"], c["step"]) for c in ca] + \
                            [(t_init + sa + t_re + c["time_s"], c["psnr"], args.s1 + c["step"]) for c in cb]
                else:
                    p, cc, _ = fit(gt, p0, mode=args.mode, steps=args.steps, log_every=25, callback=snap(0), **okw)
                    curve = [(t_init + c["time_s"], c["psnr"], c["step"]) for c in cc]
                with torch.no_grad():
                    pred = render_params(p, H, W, args.mode, aa_px=args.aa_px).clamp(0, 1)
                N = p["uv"].shape[0]
                row = {"set": s, "image": st, "H": H, "W": W, "K": K, "N": N, "init": init, "lam": lam, "n_grid": nl,
                       "curves": curves, **cfg, "params_per_gaussian": 9 if args.mode == "wnorm" else 8,
                       "gpu": torch.cuda.get_device_name(0), "t_init": t_init, "total_s": curve[-1][0],
                       "init_psnr": curve[0][1], **checkpoints(curve), **P.score(pred, gt),
                       "curve": [c for i, c in enumerate(curve) if i % 2 == 0 or i == len(curve) - 1]}
                P.append(rows_path, row)
                tag = f"K{K}_{init}"
                if P.wanted(args.save_params, s):
                    save_params(out / "params" / f"{s.replace('/', '_')}_{st}_{tag}.npz", p0=p0, final=p,
                                **{f"snap{k}": v for k, v in snaps.items()})
                m_sr = re.match(r"sr_x(\d+)/", s)
                if m_sr:                           # super-resolution: render the LR fit at S x its size
                    S = int(m_sr.group(1))
                    with torch.no_grad():
                        R.RADIUS_MULT = args.hr_radius
                        hr = render_params(p, H * S, W * S, args.mode, aa_px=args.aa_px).clamp(0, 1)
                        R.RADIUS_MULT = 1.0
                    P.save_png(out / "hr" / f"{s.replace('/', '_')}_{st}_{tag}.png", hr)
                    del hr
                if P.wanted(args.save_renders, s):
                    P.save_png(out / "renders" / f"{s.replace('/', '_')}_{st}_{tag}.png", pred)
                print(f"{s}/{st} {H}x{W} K={K} {init} [{args.mode},{args.grads},{args.opt}]: init "
                      f"{row['init_psnr']:.2f} 1s {row['psnr_1s']:.2f} final {row['psnr']:.2f} "
                      f"(line {row['psnr_edge']:.2f}) SSIM {row['ssim']:.4f} {row['total_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
