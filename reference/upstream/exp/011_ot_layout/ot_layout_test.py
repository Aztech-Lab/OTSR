"""011: does an optimal-transport layout help Gaussians? First tests, no learning.

Test A (layout): per-image Adam (weight-normalised, 2000 steps, same settings as 001) on the six 512-px images of 001,
N = 1024 / 4096 / 16384, from
  lattice      uniform lattice
  importance   Image-GS sampling p ~ lam + (1-lam)|grad I| at pixel level (random, jittered)       [001's best]
  imp_pooled   the same random sampling, but from the density pooled to the 128-long-side OT grid
  ot           the uniform lattice moved by the entropic-OT (~Brenier) map to that pooled density; covariance of
               each Gaussian (0.6 h)^2 J J^T with J the Jacobian of the map (the lattice cell, deformed)
  ot_lam01     the same with a sharper density (lam = 0.1)
Test B (encoder): 005 long-run encoder (EMA, res_cond 0) at native 2048^2 (H1 anime-1..4, h = 8 / 4): one pass on the
uniform lattice vs one pass with OT-warped anchors (density from the image, no retraining), then Adam 1000 steps from
each and from the importance initialisation.

  python -u Agent/cc/exp/011_ot_layout/ot_layout_test.py --test A --out Agent/cc/exp/011_ot_layout/runs/layout_A
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CC = HERE.parents[1]
sys.path.insert(0, str(CC))
sys.path.insert(0, str(CC / "exp" / "001_fit_ceiling"))
sys.path.insert(0, str(CC / "exp" / "006_learned_refine"))

from common import data, ffeval, metrics, viz  # noqa: E402
from common.gauss import fit, from_encoder, init_params, lattice_dims, lattice_uv, render_params, sample_color  # noqa: E402
from common.ot import ot_grid_shape, transport_map, warp_points  # noqa: E402
from common.runlog import RunLogger, save_json  # noqa: E402


def pixel_density(gt, lam):
    """Exactly init_params' importance density (per pixel, sums to 1)."""
    y = 0.299 * gt[0] + 0.587 * gt[1] + 0.114 * gt[2]
    gxm = F.pad((y[:, 1:] - y[:, :-1]).abs(), (0, 1))
    gym = F.pad((y[1:, :] - y[:-1, :]).abs(), (0, 0, 0, 1))
    g = gxm + gym
    return lam / g.numel() + (1 - lam) * g / g.sum().clamp_min(1e-12)


def pooled_density(gt, lam, n_long=128):
    _, H, W = gt.shape
    ny, nx = ot_grid_shape(H, W, n_long)
    p = F.adaptive_avg_pool2d(pixel_density(gt, lam)[None, None], (ny, nx))[0, 0]
    return p / p.sum()


def cov_to_scale_rot(C):
    """C (n, 2, 2) SPD in long-side units -> scales (n, 2) and kernel angle (n, 1): Sigma = R diag(s^2) R^T with
    R = [[c, s], [-s, c]], first principal axis (c, -s)."""
    evals, evecs = torch.linalg.eigh(C)                     # ascending
    v1 = evecs[..., :, 1]                                   # eigenvector of the larger eigenvalue
    s = torch.sqrt(evals.clamp_min(1e-12)).flip(-1)         # (s1 = larger, s2)
    theta = torch.atan2(-v1[:, 1], v1[:, 0]).unsqueeze(1)
    return s, theta


def ot_layout(gt, N, lam, n_long=128, sigma_rel=0.6):
    """Uniform lattice moved by the OT map to the pooled importance density. Returns fit params and diagnostics."""
    _, H, W = gt.shape
    L = max(H, W)
    gy, gx = lattice_dims(N, H, W)
    uv0 = lattice_uv(gy, gx, gt.device)
    rho = pooled_density(gt, lam, n_long)
    t0 = time.time()
    T, info = transport_map(rho, W / L, H / L)
    torch.cuda.synchronize()
    t_ot = time.time() - t0
    uv, J = warp_points(uv0, T, W / L, H / L)
    h = math.sqrt(H * W / uv0.shape[0]) / L
    C = (sigma_rel * h) ** 2 * (J @ J.transpose(1, 2))
    s, theta = cov_to_scale_rot(C)
    s = s.clamp(min=0.35 / L)
    p = {"uv": uv.float().clone(), "log_s": torch.log(s).float(), "rot": theta.float(),
         "color": sample_color(gt, uv.float()).clone(), "log_w": torch.zeros(uv.shape[0], 1, device=gt.device)}
    detJ = torch.linalg.det(J)
    return p, {"t_ot": t_ot, "marginal_err": info["marginal_err"], "eps": info["eps"],
               "detJ_min": float(detJ.min()), "detJ_max": float(detJ.max()), "folds": int((detJ <= 0).sum()),
               "gy": gy, "gx": gx, "T": T, "rho": rho}


def imp_pooled_layout(gt, N, lam, n_long=128, sigma_rel=0.6, seed=0):
    """Random sampling from the pooled density (cell chosen by rho, uniform inside the cell)."""
    _, H, W = gt.shape
    L = max(H, W)
    rho = pooled_density(gt, lam, n_long)
    ny, nx = rho.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.multinomial(rho.flatten().cpu(), N, replacement=True, generator=gen).to(gt.device)
    r, c = idx // nx, idx % nx
    jit = torch.rand(N, 2, generator=gen).to(gt.device)
    uv = torch.stack([(c.float() + jit[:, 0]) / nx, (r.float() + jit[:, 1]) / ny], -1)
    cell_area_px = (H / ny) * (W / nx)
    spacing_px = torch.sqrt(cell_area_px / (N * rho.flatten()[idx]))
    s = (sigma_rel * spacing_px / L).clamp(min=0.35 / L).unsqueeze(1).repeat(1, 2)
    return {"uv": uv, "log_s": torch.log(s), "rot": torch.zeros(N, 1, device=gt.device),
            "color": sample_color(gt, uv), "log_w": torch.zeros(N, 1, device=gt.device)}


def psnr_of(p, gt):
    with torch.no_grad():
        out = render_params(p, gt.shape[1], gt.shape[2], "wnorm").clamp(0, 1)
    return float(metrics.psnr(out, gt)[0]), out


def lattice_lines(uv, gy, gx, H, W, crop, up):
    """Draw the deformed lattice (neighbour links) inside a crop -> PIL image overlay helper data."""
    y0, x0, ch, cw = crop
    P = uv.view(gy, gx, 2).cpu().numpy() * np.array([W, H])
    segs = []
    for a, b in ((P[:, :-1], P[:, 1:]), (P[:-1, :], P[1:, :])):
        for p, q in zip(a.reshape(-1, 2), b.reshape(-1, 2)):
            if x0 <= p[0] < x0 + cw and y0 <= p[1] < y0 + ch:
                segs.append(((p[0] - x0) * up, (p[1] - y0) * up, (q[0] - x0) * up, (q[1] - y0) * up))
    return segs


def test_a(args, out, log, dev):
    imgs = []
    for setname, nm in [("h1_anime8", "anime-1_2k"), ("h1_anime8", "anime-3_2k"), ("h1_anime8", "anime-6_2k"),
                        ("danboo_test100", "danboo_0"), ("danboo_test100", "danboo_7"), ("poster_hold", None)]:
        items = data.eval_set(setname)
        it = items[0] if nm is None else next(x for x in items if x["name"] == nm)
        arr = data.load_rgb(it["path"])
        h, w = arr.shape[:2]
        s = 512 / max(h, w)
        imgs.append((it["name"], data.to_tensor(data.resize_rgb(arr, round(w * s), round(h * s))).to(dev)))
    inits = ["lattice", "importance", "imp_pooled", "ot", "ot_lam01"]
    rows = []
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    for name, gt in imgs:
        _, H, W = gt.shape
        for N in (1024, 4096, 16384):
            p0s, diag = {}, {}
            p0s["lattice"] = init_params(gt, N, "wnorm", how="lattice")
            p0s["importance"] = init_params(gt, N, "wnorm", how="importance", seed=0)
            p0s["imp_pooled"] = imp_pooled_layout(gt, N, 0.3)
            p0s["ot"], diag["ot"] = ot_layout(gt, N, 0.3)
            p0s["ot_lam01"], diag["ot_lam01"] = ot_layout(gt, N, 0.1)
            fitted = {}
            for k in inits:
                p0 = p0s[k]
                ps0, _ = psnr_of(p0, gt)
                p, curve, secs = fit(gt, p0, mode="wnorm", steps=args.steps, log_every=50)
                ps, outimg = psnr_of(p, gt)
                rp = metrics.region_psnr(outimg, gt)
                r = {"image": name, "N": N, "init": k, "psnr_init": ps0, "psnr": ps, "psnr_edge": float(rp["edge"][0]),
                     "curve": [(c["step"], c["psnr"]) for c in curve], "seconds": secs}
                if k in diag:
                    r.update({kk: v for kk, v in diag[k].items() if kk not in ("T", "rho")})
                rows.append(r)
                fitted[k] = (p0, p, outimg)
                print(f"{name} N={N} {k:10s}: init {ps0:.2f} -> {ps:.2f} (edge {r['psnr_edge']:.2f})"
                      + (f" | OT {diag[k]['t_ot'] * 1000:.0f} ms, folds {diag[k]['folds']}, "
                         f"detJ {diag[k]['detJ_min']:.2f}..{diag[k]['detJ_max']:.2f}" if k in diag else ""), flush=True)
            with open(out / "rows.jsonl", "a", encoding="utf-8") as f:
                for r in rows[-len(inits):]:
                    f.write(json.dumps(r) + "\n")
            if N == 4096:   # layout figure: density, deformed lattice, initial ellipses, fitted renders
                crop = viz.pick_crops(gt, size=min(128, H // 3, W // 3), k=1)[0]
                up = 3
                y0, x0, ch, cw = crop
                z = lambda im: viz.to_np(im)[y0:y0 + ch, x0:x0 + cw].repeat(up, 0).repeat(up, 1)
                tiles, titles = [z(gt)], ["GT"]
                rho = diag["ot"]["rho"]
                dimg = F.interpolate(rho[None, None] / rho.max(), size=(H, W), mode="nearest")[0].expand(3, -1, -1)
                tiles.append(z(dimg))
                titles.append("target density (lam .3)")
                from PIL import ImageDraw
                base = viz.Image.fromarray(z(gt)).convert("RGB")
                dr = ImageDraw.Draw(base)
                for sgm in lattice_lines(p0s["ot"]["uv"], diag["ot"]["gy"], diag["ot"]["gx"], H, W, crop, up):
                    dr.line(sgm, fill=(255, 40, 40), width=1)
                tiles.append(base)
                titles.append("OT-warped lattice")
                for k in ("importance", "ot"):
                    p0 = p0s[k]
                    tiles.append(viz.ellipses(gt, p0["uv"], torch.exp(p0["log_s"]), p0["rot"], H, W, crop=crop, upscale=up))
                    titles.append(f"init ellipses: {k}")
                for k in ("lattice", "importance", "ot"):
                    tiles.append(z(fitted[k][2]))
                    rr = next(r for r in rows if r["image"] == name and r["N"] == N and r["init"] == k)
                    titles.append(f"fit {k}: {rr['psnr']:.2f}dB")
                viz.grid(tiles, titles, ncol=4, max_w=3200).save(fig / f"layout_{name}_N{N}.png")
    # summaries + curves
    summ = {}
    for N in (1024, 4096, 16384):
        summ[N] = {}
        for k in inits:
            R = [r for r in rows if r["N"] == N and r["init"] == k]
            summ[N][k] = {"psnr_init": float(np.mean([r["psnr_init"] for r in R])), "psnr": float(np.mean([r["psnr"] for r in R])),
                          "psnr_edge": float(np.mean([r["psnr_edge"] for r in R]))}
        steps = [s for s, _ in rows[0]["curve"]]
        viz.plot_curves({k: (steps, [float(np.mean([dict(r["curve"])[s] for r in rows if r["N"] == N and r["init"] == k]))
                                     for s in steps]) for k in inits},
                        "Adam step", f"PSNR (mean of 6, N={N})", fig / f"curves_N{N}.png", logx=False, marker="")
    save_json(out / "summary.json", summ)
    print(json.dumps(summ, indent=1))
    log.save_summary({"summary": summ})


def residual_monitor(gt, render, rho1, gamma=0.5, blur_px=4.0):
    """Error-equidistribution monitor: rho2 ~ rho1 * (E / mean E + 0.1)^gamma with E the blurred squared residual,
    pooled to the OT grid of rho1."""
    e = ((render - gt) ** 2).mean(0, keepdim=True)[None]
    r = int(math.ceil(3 * blur_px))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=gt.device, dtype=e.dtype) / blur_px) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1)
    e = F.conv2d(F.pad(e, (r, r, 0, 0), mode="replicate"), k)
    e = F.conv2d(F.pad(e, (0, 0, r, r), mode="replicate"), k.transpose(-1, -2))
    E = F.adaptive_avg_pool2d(e, rho1.shape)[0, 0]
    rho2 = rho1 * (E / E.mean() + 0.1) ** gamma
    return rho2 / rho2.sum()


def ot_params_from_map(gt, gy, gx, T, ex, ey, sigma_rel=0.6):
    _, H, W = gt.shape
    L = max(H, W)
    uv, J = warp_points(lattice_uv(gy, gx, gt.device), T, ex, ey)
    h = math.sqrt(H * W / (gy * gx)) / L
    s, theta = cov_to_scale_rot((sigma_rel * h) ** 2 * (J @ J.transpose(1, 2)))
    return {"uv": uv.float().clone(), "log_s": torch.log(s.clamp(min=0.35 / L)).float(), "rot": theta.float(),
            "color": sample_color(gt, uv.float()).clone(), "log_w": torch.zeros(uv.shape[0], 1, device=gt.device)}, uv, J


def test_c(args, out, log, dev):
    """Error-driven OT re-layout (the core of an OT refinement step), per-image optimisation at 512 px:
      ot1        OT layout from the gradient density (lam 0.1 for N <= 4096, else 0.3), Adam S1 + S steps
      ot2_fresh  OT layout -> Adam S1 -> monitor from the residual -> new OT layout, fresh attributes -> Adam S
      ot2_warm   ... the fitted Gaussians are moved along T2(T1^-1): positions follow the lattice, covariances by
                 J2 J1^-1, colours kept (Adam then adapts them) -> Adam S"""
    imgs = []
    for setname, nm in [("h1_anime8", "anime-1_2k"), ("h1_anime8", "anime-3_2k"), ("h1_anime8", "anime-6_2k"),
                        ("danboo_test100", "danboo_0"), ("danboo_test100", "danboo_7"), ("poster_hold", None)]:
        items = data.eval_set(setname)
        it = items[0] if nm is None else next(x for x in items if x["name"] == nm)
        arr = data.load_rgb(it["path"])
        h, w = arr.shape[:2]
        s = 512 / max(h, w)
        imgs.append((it["name"], data.to_tensor(data.resize_rgb(arr, round(w * s), round(h * s))).to(dev)))
    S1, S = args.s1, args.steps
    rows = []
    for name, gt in imgs:
        _, H, W = gt.shape
        L = max(H, W)
        ex, ey = W / L, H / L
        for N in (1024, 4096, 16384):
            lam = 0.1 if N <= 4096 else 0.3
            gy, gx = lattice_dims(N, H, W)
            rho1 = pooled_density(gt, lam)
            T1, _ = transport_map(rho1, ex, ey)
            p1, uv1, J1 = ot_params_from_map(gt, gy, gx, T1, ex, ey)
            res = {}
            p, curve, _ = fit(gt, p1, mode="wnorm", steps=S1 + S, log_every=50)
            res["ot1"] = {"psnr": psnr_of(p, gt)[0], "curve": [(c["step"], c["psnr"]) for c in curve]}
            # pass 1 (shared by both two-pass variants)
            pa, curve_a, _ = fit(gt, p1, mode="wnorm", steps=S1, log_every=50)
            ps_a, rend_a = psnr_of(pa, gt)
            rho2 = residual_monitor(gt, rend_a, rho1, gamma=args.gamma)
            T2, _ = transport_map(rho2, ex, ey)
            p2, uv2, J2 = ot_params_from_map(gt, gy, gx, T2, ex, ey)
            pf, curve_f, _ = fit(gt, p2, mode="wnorm", steps=S, log_every=50)
            res["ot2_fresh"] = {"psnr": psnr_of(pf, gt)[0], "after_pass1": ps_a,
                                "curve": [(c["step"] + S1, c["psnr"]) for c in curve_f]}
            # warm: move each fitted Gaussian with its lattice cell; covariance by A = J2 J1^-1
            A = (J2 @ torch.linalg.inv(J1)).double()
            from common.render import conics_from_scale_rot
            with torch.no_grad():
                s_old = torch.exp(pa["log_s"])
                rot_old = pa["rot"]
                conic = conics_from_scale_rot(s_old, rot_old)            # inverse covariance (a, b, d), long-side units
                Ci = torch.stack([torch.stack([conic[:, 0], conic[:, 1]], -1), torch.stack([conic[:, 1], conic[:, 2]], -1)], -2)
                C_old = torch.linalg.inv(Ci.double())
                C_new = A @ C_old @ A.transpose(1, 2)
                s_new, th_new = cov_to_scale_rot(C_new)
                d_cell = (pa["uv"].double() - uv1) * torch.tensor([ex, ey], device=dev, dtype=torch.float64)
                d_new = (A @ d_cell.unsqueeze(-1)).squeeze(-1) / torch.tensor([ex, ey], device=dev, dtype=torch.float64)
                pw = {"uv": (uv2 + d_new).clamp(0, 1).float(), "log_s": torch.log(s_new.clamp(min=0.35 / L)).float(),
                      "rot": th_new.float(), "color": pa["color"].detach().clone(), "log_w": pa["log_w"].detach().clone()}
            ps_w0, _ = psnr_of(pw, gt)
            pwf, curve_w, _ = fit(gt, pw, mode="wnorm", steps=S, log_every=50)
            res["ot2_warm"] = {"psnr": psnr_of(pwf, gt)[0], "after_move": ps_w0,
                               "curve": [(c["step"] + S1, c["psnr"]) for c in curve_w]}
            row = {"image": name, "N": N, "pass1": ps_a, **res}
            rows.append(row)
            with open(out / "rows_c.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{name} N={N}: ot1 {res['ot1']['psnr']:.2f} | pass1 {ps_a:.2f} -> fresh {res['ot2_fresh']['psnr']:.2f}"
                  f" | warm (after move {ps_w0:.2f}) {res['ot2_warm']['psnr']:.2f}", flush=True)
    summ = {}
    for N in (1024, 4096, 16384):
        R = [r for r in rows if r["N"] == N]
        summ[N] = {k: float(np.mean([r[k]["psnr"] for r in R])) for k in ("ot1", "ot2_fresh", "ot2_warm")}
        summ[N]["pass1"] = float(np.mean([r["pass1"] for r in R]))
    save_json(out / "summary_c.json", summ)
    print(json.dumps(summ, indent=1))
    log.save_summary({"summary_c": summ})


def test_b(args, out, log, dev):
    from train_refine import load_encoder
    enc, _, _ = load_encoder("Agent/cc/exp/005_decoder2",
                             "Agent/cc/exp/005_decoder2/runs/long_2stage_b8_h2-24_s100k/ema_100000.pt", dev)
    enc.res_cond = 0.0
    rows = []
    for it, arr in ffeval.load_set("h1_anime8")[:4]:
        x = data.to_tensor(arr).to(dev)[None]
        H, W = x.shape[-2:]
        L = max(H, W)
        for h in (8.0, 4.0):
            N = round(H * W / (h * h))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                feats = enc.features(x)
                g_lat = enc.decode(feats, x, N)
            gy, gx = g_lat["lattice"]
            rho = pooled_density(x[0], 0.3)
            T, info = transport_map(rho, W / L, H / L)
            uv_ot, J = warp_points(lattice_uv(gy, gx, dev), T, W / L, H / L)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                g_ot = enc.decode(feats, x, N, anchors=uv_ot.float().view(1, gy, gx, 2))
            res = {}
            for k, g in (("enc", g_lat), ("enc_ot_anchors", g_ot)):
                g1 = {kk: (v[0].float() if torch.is_tensor(v) and v.dim() > 1 else v) for kk, v in g.items()}
                p0 = from_encoder(g1)
                ps0, _ = psnr_of(p0, x[0])
                p, curve, _ = fit(x[0], p0, mode="wnorm", steps=args.steps_b, lr_px=0.1, lr_logs=0.01, lr_rot=0.01,
                                  lr_col=0.005, lr_logw=0.01, log_every=50)
                res[k] = {"psnr_init": ps0, "psnr": psnr_of(p, x[0])[0], "curve": [(c["step"], c["psnr"]) for c in curve]}
            p0 = init_params(x[0], N, "wnorm", how="importance", seed=0)
            ps0, _ = psnr_of(p0, x[0])
            p, curve, _ = fit(x[0], p0, mode="wnorm", steps=args.steps_b, log_every=50)
            res["importance"] = {"psnr_init": ps0, "psnr": psnr_of(p, x[0])[0], "curve": [(c["step"], c["psnr"]) for c in curve]}
            row = {"image": it["name"], "h": h, "N": N, **res}
            rows.append(row)
            with open(out / "rows_b.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{it['name']} h={h}: " + " | ".join(f"{k} {v['psnr_init']:.2f} -> {v['psnr']:.2f}" for k, v in res.items()),
                  flush=True)
    summ = {}
    for h in (8.0, 4.0):
        R = [r for r in rows if r["h"] == h]
        summ[h] = {k: {"psnr_init": float(np.mean([r[k]["psnr_init"] for r in R])), "psnr": float(np.mean([r[k]["psnr"] for r in R]))}
                   for k in ("enc", "enc_ot_anchors", "importance")}
        steps = [s for s, _ in R[0]["enc"]["curve"]]
        viz.plot_curves({k: (steps, [float(np.mean([dict(r[k]["curve"])[s] for r in R])) for s in steps])
                         for k in ("enc", "enc_ot_anchors", "importance")},
                        "Adam step", f"PSNR (H1 2048^2, 4 images, h={h:g})", out / "figures" / f"b_curves_h{h:g}.png",
                        logx=False, marker="")
    save_json(out / "summary_b.json", summ)
    print(json.dumps(summ, indent=1))
    log.save_summary({"summary_b": summ})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="A", choices=["A", "B", "C"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--steps-b", type=int, default=1000)
    ap.add_argument("--s1", type=int, default=300, help="test C: steps before the error-driven re-layout")
    ap.add_argument("--gamma", type=float, default=0.5, help="test C: monitor exponent on the residual energy")
    args = ap.parse_args()
    dev = "cuda:0"
    from common.gpu import guard
    guard()
    out = Path(args.out)
    with RunLogger(out, vars(args) | {"name": out.name, "experiment": "011_ot_layout"}, code_root=HERE) as log:
        (out / "figures").mkdir(parents=True, exist_ok=True)
        {"A": test_a, "B": test_b, "C": test_c}[args.test](args, out, log, dev)


if __name__ == "__main__":
    main()
