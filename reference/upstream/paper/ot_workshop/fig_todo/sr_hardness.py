"""每个高斯学一个温度（硬度），拟合时一起学；渲染时一个全局旋钮 κ 控制整体锐度（附加功能，SR，定性）。
参考 SAD（每个站点一个 τ_i，softmax(−τ_i d_i)）。我们的 wnorm：ℓ_i = log w_i − d_i²/2，第 i 个高斯的温度 τ_i = exp(κ·log_t_i)：
  softmax(τ_i ℓ_i) ⇔ Σ_i/τ_i、w_i^τ_i；κ = 0 就是原来的渲染（log_t 不起作用），κ = 1 是学到的锐度，κ > 1 更锐。
怎么学 log_t：同一批高斯两种渲染都要对上原图——
  ① 原渲染（κ = 0），原分辨率逐像素采样（就是平时的拟合）；
  ② 锐渲染（κ 在 kappa_train 里随机取），在随机窗口里按 s 倍超采样、再按像素面积平均回原分辨率（像素 = 面积）。
  边缘：原图里约 1 像素的抗锯齿过渡 = 硬边按像素面积平均的结果，② 要它变硬 → τ 学大；
  渐变：变硬会出色带，② 会罚 → τ 留在 1。哪些边界该锐化、哪些不变，由数据决定。
  κ 随机取，整个旋钮范围都被训练过（边缘变硬后不能起锯齿 / 串珠）。
  相邻高斯 τ 不一样时分界线会挪，所以照 SAD 的 tau diffusion：log_t 的梯度在 kNN 邻居间扩散，邻居一起变。
先有一个普通拟合（params 给现成的，或 None = 自己用 OT 拟合），再加 log_t 微调。改 PARAMS 后运行（或双击 sr_hardness.bat）。
输出：fig_todo/sr_hardness/<图名>_x<S>.png（ROI 对比，每行一个 ROI）、<图名>_tau.png（学到的 τ 图）、拟合缓存 .npz。
"""
import math
import os
import sys
import time
from pathlib import Path

PARAMS = dict(
    image="../../../../../data/subsets/DanbooRegion_test_24/120.image.png",   # 相对 fig_todo，或绝对路径（线稿干净的平涂图）
    params=None,                  # 起点（现成拟合的 .npz，final_*）；None = 自己拟合（OT 初始化，结果缓存）
    # 软画风的例子：image="../../../cache/rev15/native/igs_anime/anime-5_2k.png",
    #             params="../../../exp/015_revision/runs/rev/ours/params/native_igs_anime_anime-5_2k_K160000_ot_relayout.npz"
    K_per_px=0.1,                 # params = None 时：每像素高斯数
    fit_steps=3000,               # params = None 时：拟合步数
    steps=2000,                   # 微调步数
    warmup=200,                   # 学习率预热步数（已收敛的拟合，直接大步会先掉点）
    s_train=4,                    # ② 的超采样倍数
    crop=192,                     # ② 的窗口边长（原图像素）
    n_crops=2,                    # 每步几个窗口
    lam=0.3,                      # ② 的权重（1.0：原分辨率掉 3.4 dB；0.3：掉 1.6 dB，锐度差不多）
    kappa_train=(1.0, 4.0),       # ② 的 κ 随机范围
    diffuse_lam=0.5,              # τ 梯度扩散强度（SAD 用 0.005→0.05）
    diffuse_iters=4,
    knn=8,                        # 扩散用的邻居数
    lr_t=0.05,                    # log_t 学习率
    lr_scale=0.3,                 # 其余参数的学习率 = 原拟合默认值 × lr_scale（已收敛，小步微调）
    tau_max=16.0,                 # 学到的 τ 上限
    scale=8,                      # 展示：放大倍数
    kappas=(0, 1, 2, 4, 6),       # 展示：全局旋钮
    tau_cap=32.0,                 # 展示：τ_i^κ 的上限（太大会数值下溢出黑点）
    rois=("edges", "smooth"),     # 每行一个 ROI："edges" / "smooth" / (y0, x0)
    roi_size=56,
    tile_px=448,
    font_px=18,
    gpu="GPU-1416b4f8-0b51-4e4d-ae4e-7fb363c07e93",   # 5070 Ti
    refit=False,                  # True：忽略缓存重新微调
)
os.environ["CUDA_VISIBLE_DEVICES"] = PARAMS["gpu"]
HERE = Path(__file__).resolve().parent
CC = HERE.parents[2]
for p in (CC, CC / "exp" / "011_ot_layout", CC / "exp" / "015_revision", HERE, HERE.parent / "figs15"):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from common import render as R  # noqa: E402
from common.gauss import fit, lattice_dims, render_params  # noqa: E402
from common.ot import lam_for_spacing  # noqa: E402
from fig_qual15 import pick_crop  # noqa: E402
from gmod.gsplat.rasterize_sum import rasterize_gaussians_sum  # noqa: E402
from sr_temperature import pick_smooth  # noqa: E402

B = R.BLOCK


def render_hard(p, H, W, kappa, s=1, win=None, down=True, cull=1.0, tau_cap=None, eps=1e-20):
    """wnorm with per-Gaussian temperature tau_i = exp(kappa log_t_i); see render_t."""
    t = torch.exp(kappa * p["log_t"].reshape(-1, 1))
    if tau_cap:
        t = t.clamp(max=tau_cap)
    return render_t(p, t, H, W, s, win, down, cull, eps)


def render_t(p, t, H, W, s=1, win=None, down=True, cull=1.0, eps=1e-20):
    """wnorm with per-Gaussian temperature t (N, 1): Sigma_i / t_i, w_i^t_i, culled at cull x 3 sigma of the
    wider of the tempered and untempered Gaussian. win = (X0, Y0, C): only that native window. Rendered at s x;
    down=True averages back to native pixels (pixel = area)."""
    uv, log_s = p["uv"], p["log_s"]
    if win is not None:
        X0, Y0, C = win
        uv = (uv * uv.new_tensor([W, H]) - uv.new_tensor([X0, Y0])) / C
        log_s = log_s + math.log(max(H, W) / C)
        H = W = C
    Hs, Ws = H * s, W * s
    s_px = torch.exp(log_s) * max(Hs, Ws)
    s_eff = s_px / torch.sqrt(t)
    xys = (uv - uv.new_tensor([0.5 / Ws, 0.5 / Hs])) * uv.new_tensor([float(Ws), float(Hs)])
    conics = R.conics_from_scale_rot(s_eff, p["rot"])
    with torch.no_grad():                       # integer culling data, as the kernel would compute them
        rad = torch.ceil(3.0 * cull * torch.maximum(s_px, s_eff).amax(1))
        ntx, nty = (Ws + B - 1) // B, (Hs + B - 1) // B
        cx, cy = xys[:, 0], xys[:, 1]
        tx0 = torch.clamp(torch.trunc((cx - rad) / B), 0, ntx)
        tx1 = torch.clamp(torch.trunc((cx + rad) / B + 1), 0, ntx)
        ty0 = torch.clamp(torch.trunc((cy - rad) / B), 0, nty)
        ty1 = torch.clamp(torch.trunc((cy + rad) / B + 1), 0, nty)
        nth = ((tx1 - tx0) * (ty1 - ty0)).to(torch.int32)
        radii = torch.where(nth > 0, rad, torch.zeros_like(rad)).to(torch.int32)
    lw = t * p["log_w"].reshape(-1, 1)
    w = torch.exp((lw - lw.detach().median()).clamp(-80, 80))    # softmax-invariant shift; w^t stays in float range
    acc = rasterize_gaussians_sum(xys.contiguous(), radii, conics, nth, torch.cat([p["color"] * w, w], 1),
                                  Hs, Ws, B, B, topk_norm=False)
    img = (acc[..., :3] / (acc[..., 3:] + eps)).reshape(Hs, Ws, 3).permute(2, 0, 1)
    return F.avg_pool2d(img[None], s)[0] if (down and s > 1) else img


def psnr(a, b):
    return -10 * math.log10(max(float(((a.clamp(0, 1) - b) ** 2).mean()), 1e-12))


def finetune(P, p0, gt):
    _, H, W = gt.shape
    L, C = max(H, W), P["crop"]
    p = {k: v.detach().clone().requires_grad_(True) for k, v in p0.items()}
    k_ = P["lr_scale"]
    opt = torch.optim.Adam([{"params": [p["uv"]], "lr": 0.25 / L * k_}, {"params": [p["log_s"]], "lr": 0.02 * k_},
                            {"params": [p["color"]], "lr": 0.01 * k_}, {"params": [p["rot"]], "lr": 0.02 * k_},
                            {"params": [p["log_w"]], "lr": 0.02 * k_}, {"params": [p["log_t"]], "lr": P["lr_t"]}],
                           betas=(0.9, 0.99))
    base = [g["lr"] for g in opt.param_groups]
    g = torch.Generator().manual_seed(0)
    from scipy.spatial import cKDTree
    xy = (p0["uv"] * p0["uv"].new_tensor([W, H])).cpu().numpy()
    nb = torch.from_numpy(cKDTree(xy).query(xy, k=P["knn"] + 1)[1][:, 1:]).long().cuda()
    lo, hi = P["kappa_train"]
    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(P["steps"]):
        k = (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * it / P["steps"]))) * min(1.0, (it + 1) / P["warmup"])
        for gr, b in zip(opt.param_groups, base):
            gr["lr"] = b * k
        l_plain = F.mse_loss(render_params(p, H, W, "wnorm"), gt)
        l_sharp = 0.0
        for _ in range(P["n_crops"]):
            X0 = int(torch.randint(0, W - C + 1, (1,), generator=g))
            Y0 = int(torch.randint(0, H - C + 1, (1,), generator=g))
            kp = lo + (hi - lo) * float(torch.rand(1, generator=g))
            out = render_hard(p, H, W, kp, s=P["s_train"], win=(X0, Y0, C), tau_cap=P["tau_cap"], eps=1e-30)
            l_sharp = l_sharp + F.mse_loss(out, gt[:, Y0:Y0 + C, X0:X0 + C]) / P["n_crops"]
        loss = l_plain + P["lam"] * l_sharp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gr0 = p["log_t"].grad.clone()            # SAD-style tau diffusion: Jacobi steps of (I + lam L) g = g_raw
        gt_ = gr0
        for _ in range(P["diffuse_iters"]):
            gt_ = (gr0 + P["diffuse_lam"] * gt_[nb].sum(1)) / (1 + P["diffuse_lam"] * P["knn"])
        p["log_t"].grad = gt_
        opt.step()
        with torch.no_grad():
            p["uv"].clamp_(-0.02, 1.02)
            p["log_s"].clamp_(math.log(0.2 / L), math.log(0.5))
            p["log_w"].clamp_(-8, 8)
            p["log_t"].clamp_(0.0, math.log(P["tau_max"]))
        if it % 100 == 0 or it == P["steps"] - 1:
            tq = torch.exp(p["log_t"].detach()).flatten()
            print(f"  step {it:5d}  plain {-10 * math.log10(float(l_plain)):.2f} dB  sharp(crops) "
                  f"{-10 * math.log10(float(l_sharp)):.2f} dB  tau median {float(tq.median()):.2f} "
                  f"p90 {float(tq.quantile(0.9)):.2f} max {float(tq.max()):.2f}", flush=True)
    torch.cuda.synchronize()
    print(f"fine-tuned {P['steps']} steps in {time.time() - t0:.1f}s on {torch.cuda.get_device_name(0)}")
    return {k: v.detach() for k, v in p.items()}


def tau_map(p, H, W, P, out_png, pil):
    """The learned log tau, splatted with the plain renderer, next to the image."""
    from matplotlib import cm
    q = dict(p)
    top = max(float(p["log_t"].flatten().quantile(0.995)), 1e-3)   # colour range = 1 .. 99.5th percentile of tau
    lt = (p["log_t"].reshape(-1, 1) / top).clamp(0, 1)
    q["color"] = lt.repeat(1, 3)
    with torch.no_grad():
        m = render_params(q, H, W, "wnorm")[0].clamp(0, 1).cpu().numpy()
    rgb = (cm.magma(m)[..., :3] * 255 + 0.5).astype(np.uint8)
    T = 900
    sc = T / max(H, W)
    a = pil.resize((round(W * sc), round(H * sc)), Image.LANCZOS)
    b = Image.fromarray(rgb).resize(a.size, Image.LANCZOS)
    cv = Image.new("RGB", (a.width * 2 + 24 + 80, a.height + 40), "white")
    cv.paste(a, (8, 32))
    cv.paste(b, (a.width + 16, 32))
    d = ImageDraw.Draw(cv)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    d.text((8, 6), "Image", fill="black", font=font)
    d.text((a.width + 16, 6), "Learned temperature τ_i (log scale)", fill="black", font=font)
    x0 = a.width * 2 + 24
    for i in range(a.height):                   # colour bar
        v = 1 - i / (a.height - 1)
        d.line([(x0, 32 + i), (x0 + 16, 32 + i)], fill=tuple(int(255 * c) for c in cm.magma(v)[:3]))
    d.text((x0 + 20, 28), f"{math.exp(top):.2f}", fill="black", font=font)
    d.text((x0 + 20, 32 + a.height - 18), "1", fill="black", font=font)
    cv.save(out_png)


def main(P=PARAMS):
    img = Path(P["image"])
    img = img if img.is_absolute() else (HERE / img).resolve()
    a = np.asarray(Image.open(img).convert("RGB"), np.float32) / 255
    H, W = a.shape[:2]
    L, S, c = max(H, W), P["scale"], P["roi_size"]
    out = HERE / "sr_hardness"
    out.mkdir(exist_ok=True)
    gt = torch.from_numpy(a).permute(2, 0, 1).contiguous().cuda()
    if P["params"]:
        f = Path(P["params"])
        z = np.load(f if f.is_absolute() else (HERE / f).resolve())
        p0 = {k[6:]: torch.from_numpy(z[k]).cuda() for k in z.files if k.startswith("final_")}
    else:
        K = int(round(P["K_per_px"] * H * W))
        fc = out / f"{img.stem}_K{K}_ot_fit{P['fit_steps']}.npz"
        if fc.exists():
            z = np.load(fc)
            p0 = {k: torch.from_numpy(z[k]).cuda() for k in z.files}
        else:
            import run_ours15 as RO
            gy, gx = lattice_dims(K, H, W)
            lam = lam_for_spacing(math.sqrt(H * W / (gy * gx)))
            pi, _, _ = RO.make_init(gt, K, "ot", lam, "wnorm", min(256, max(128, round(L / 4))))
            p0, _, sec = fit(gt, pi, mode="wnorm", steps=P["fit_steps"])
            np.savez_compressed(fc, **{k: v.cpu().numpy() for k, v in p0.items()})
            print(f"fitted K={K:,} in {sec:.1f}s")
    p0["log_t"] = torch.zeros_like(p0["log_w"])
    cache = out / (f"{img.stem}_K{p0['uv'].shape[0]}_hard_s{P['s_train']}_n{P['steps']}"
                   f"_k{P['kappa_train'][1]:g}_d{P['diffuse_lam']:g}_l{P['lam']:g}.npz")
    if cache.exists() and not P["refit"]:
        z = np.load(cache)
        p = {k: torch.from_numpy(z[k]).cuda() for k in z.files}
        print("loaded", cache.name)
    else:
        p = finetune(P, p0, gt)
        np.savez_compressed(cache, **{k: v.cpu().numpy() for k, v in p.items()})
    with torch.no_grad():
        print(f"native plain PSNR: start {psnr(render_params(p0, H, W, 'wnorm'), gt):.2f} dB -> "
              f"after {psnr(render_params(p, H, W, 'wnorm'), gt):.2f} dB")
        for kp in P["kappas"]:
            print(f"kappa {kp:g}: native area-sampled (s=2) PSNR "
                  f"{psnr(render_hard(p, H, W, kp, s=2, tau_cap=P['tau_cap'], eps=1e-30), gt):.2f} dB")
    pil = Image.fromarray((a * 255 + 0.5).astype(np.uint8))
    tau_map(p, H, W, P, out / f"{img.stem}_tau.png", pil)
    m = 8
    rows = []
    for r in P["rois"]:
        y0, x0 = pick_crop(a, c) if r == "edges" else pick_smooth(a, c) if r == "smooth" else r
        X0, Y0, C = x0 - m, y0 - m, c + 2 * m
        cut = lambda im: im.crop((m * S, m * S, (m + c) * S, (m + c) * S))
        win = pil.crop((X0, Y0, X0 + C, Y0 + C))
        cols = [(f"Input (nearest ×{S})", cut(win.resize((C * S, C * S), Image.NEAREST))),
                (f"Bicubic ×{S}", cut(win.resize((C * S, C * S), Image.BICUBIC))),
                (f"Lanczos ×{S}", cut(win.resize((C * S, C * S), Image.LANCZOS)))]
        with torch.no_grad():
            for kp in P["kappas"]:
                im = render_hard(p, H, W, kp, s=S, win=(X0, Y0, C), down=False, cull=5 / 3,
                                 tau_cap=P["tau_cap"], eps=1e-30).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                cols.append((f"Ours, κ = {kp:g}" + (" (plain)" if kp == 0 else ""),
                             cut(Image.fromarray((im * 255 + 0.5).astype(np.uint8)))))
        rows.append(((y0, x0), cols))
        print(f"ROI '{r}': y0,x0,size = {y0},{x0},{c}")
    T, pad, fp = P["tile_px"], 8, P["font_px"]
    try:
        font = ImageFont.truetype("arial.ttf", fp)
    except OSError:
        font = ImageFont.load_default()
    n, top = len(rows[0][1]), fp + 12
    canvas = Image.new("RGB", (pad + (n + 1) * (T + pad), top + len(rows) * (T + pad)), "white")
    d = ImageDraw.Draw(canvas)
    s_ = T / L
    d.text((pad, 4), f"Image {W}×{H}, K = {p['uv'].shape[0]:,}", fill="black", font=font)
    for i, ((y0, x0), cols) in enumerate(rows):
        yy = top + i * (T + pad)
        th = pil.resize((round(W * s_), round(H * s_)), Image.LANCZOS)
        ImageDraw.Draw(th).rectangle([x0 * s_, y0 * s_, (x0 + c) * s_, (y0 + c) * s_], outline=(255, 40, 40), width=2)
        canvas.paste(th, (pad + (T - th.width) // 2, yy + (T - th.height) // 2))
        for j, (t, im) in enumerate(cols):
            x = pad + (j + 1) * (T + pad)
            canvas.paste(im.resize((T, T), Image.NEAREST if c * S < T else Image.LANCZOS), (x, yy))
            if i == 0:
                d.text((x, 4), t, fill="black", font=font)
            im.save(out / f"{img.stem}_x{S}_r{i}_{j}.png")
    f = out / f"{img.stem}_x{S}.png"
    canvas.save(f)
    print("written", f)


if __name__ == "__main__":
    main()
