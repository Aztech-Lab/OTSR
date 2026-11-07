"""各区域的温度怎么随分辨率变（附加功能，SR，分析）。
同一批高斯（原图上拟合好，冻结位置 / 形状 / 颜色 / 权重），只学温度 τ（Σ/τ、w^τ）去对上原图按像素面积缩小的 1/2、1/4、1/8：
  原图里约 1 像素宽的边缘（宽度由像素决定）：缩小 r 倍后宽 1/r 倍 → τ ∝ r；本来就是渐变的：宽度不随分辨率变 → τ 不变。
温度场：field = "grid"：每层一张粗网格（cell_px 一格 = 一个区域），网格上的 log τ 双线性插值到各高斯；
        field = "gaussian"：每个高斯每层一个 τ（参数比 1/8 层的数据还多，欠定，学出来是噪声，只作对照）。
每个区域拟合 log τ(r) = b · log r（锐化指数 b：1 = 跟着像素走，放大时该锐化；0 = 本来就软，不动），
放大 S 倍时 τ = S^(κ b)，κ 是全局旋钮（κ = 1 = 按规律外推）。
另外两个检查：
  · 留一层验证（有真值）：原图按面积缩小 S 倍当输入，在输入的 1/2、1/4、1/8 上学 b，外推回原图分辨率和原图比。
  · 不同高斯预算下 b 怎么变。
改 PARAMS 后运行（或双击 sr_mres_tau.bat）。输出在 fig_todo/sr_mres_tau/：
  <图名>_maps.png（各层温度 + b）、<图名>_curves.png（温度曲线、b 和线稿密度的关系、各层 PSNR）、
  <图名>_x<S>.png（原图直接放大）、<图名>_holdout_x<S>.png（有真值的验证）、<图名>_budgets.png、summary.json。
"""
import json
import math
import os
import sys
from pathlib import Path

PARAMS = dict(
    image="../../../../../data/subsets/DanbooRegion_test_24/120.image.png",   # 相对 fig_todo，或绝对路径
    K_per_px=0.1,                 # 每像素高斯数（原图拟合，OT 初始化，结果缓存）
    fit_steps=3000,
    levels=(1, 2, 3),             # 学温度的层：缩小 2^l
    field="grid",                 # "grid"（每个区域一个温度）/ "gaussian"（每个高斯一个，对照）
    cell_px=32,                   # grid：一个区域的边长（原图像素）
    grid_steps=600,
    steps=1500,                   # gaussian：学温度的步数
    lr_t=0.05,
    warmup=100,
    diffuse_lam=0.5,              # gaussian：τ 梯度扩散（SAD 的 tau diffusion）
    diffuse_iters=4,
    knn=8,
    edge_thr=0.1,                 # 线稿像素：5×5 邻域内最大亮度差 > edge_thr
    scale=8,                      # 原图直接放大的倍数（没有真值，定性）
    kappas=(0, 0.5, 1, 1.5),      # 全局旋钮
    tau_cap=32.0,
    rois=("edges", "smooth"),     # "edges" / "smooth" / (y0, x0)
    roi_size=56,                  # 原图直接放大的 ROI（原图像素）
    holdout_S=(2, 4),             # 留一层验证的放大倍数；() = 不跑
    roi_hold=112,                 # 验证图的 ROI（原图像素）
    budgets=(0.025, 0.05, 0.1, 0.2),   # 不同预算下的 b；() = 不跑
    tile_px=448,
    font_px=18,
    gpu="GPU-1416b4f8-0b51-4e4d-ae4e-7fb363c07e93",   # 5070 Ti
)
os.environ["CUDA_VISIBLE_DEVICES"] = PARAMS["gpu"]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import sr_hardness  # noqa: E402,F401  (sets up the import paths)
os.environ["CUDA_VISIBLE_DEVICES"] = PARAMS["gpu"]
from sr_hardness import psnr, render_t  # noqa: E402
from common.gauss import fit, lattice_dims, render_params  # noqa: E402
from common.ot import lam_for_spacing  # noqa: E402
from fig_qual15 import pick_crop  # noqa: E402
from sr_temperature import pick_smooth  # noqa: E402

OUT = HERE / "sr_mres_tau"
LN2 = math.log(2)


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).contiguous().cuda()


def to_pil(x):
    return Image.fromarray((x.clamp(0, 1).nan_to_num(0).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))


def base_fit(gt, K_per_px, steps, tag, use_cache=True, callback=None):
    _, H, W = gt.shape
    K = int(round(K_per_px * H * W))
    f = OUT / "fits" / f"{tag}_{W}x{H}_K{K}.npz"
    if use_cache and f.exists():
        z = np.load(f)
        return {k: torch.from_numpy(z[k]).cuda() for k in z.files}
    import run_ours15 as RO
    gy, gx = lattice_dims(K, H, W)
    lam = lam_for_spacing(math.sqrt(H * W / (gy * gx)))
    p0, _, _ = RO.make_init(gt, K, "ot", lam, "wnorm", min(256, max(128, round(max(H, W) / 4))))
    p, _, sec = fit(gt, p0, mode="wnorm", steps=steps, log_every=20, callback=callback)
    if use_cache:
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **{k: v.cpu().numpy() for k, v in p.items()})
    with torch.no_grad():
        print(f"  fitted {tag} {W}x{H} K={K:,} in {sec:.1f}s: {psnr(render_params(p, H, W, 'wnorm'), gt):.2f} dB")
    return p


def pyramid(gt, levels):
    return {l: F.avg_pool2d(gt[None], 2 ** l)[0] for l in levels}


def knn(p, H, W, k):
    from scipy.spatial import cKDTree
    xy = (p["uv"] * p["uv"].new_tensor([W, H])).cpu().numpy()
    return torch.from_numpy(cKDTree(xy).query(xy, k=k + 1)[1][:, 1:]).long().cuda()


def sample_grid(G, uv):
    """(1, L, gh, gw) field -> (N, L) at the Gaussian centres (bilinear)."""
    return F.grid_sample(G, (uv * 2 - 1).view(1, 1, -1, 2), mode="bilinear", padding_mode="border",
                         align_corners=False)[0, :, 0].T


def probe(p, gts, H, W, P, mode):
    """log tau per level for frozen Gaussians. mode: "global" (1, L), "grid" (G: 1, L, gh, gw) or "gaussian" (N, L).
    Returns (log tau at the Gaussians (N, L), the field itself)."""
    lv = sorted(gts)
    N = p["uv"].shape[0]
    if mode == "grid":
        X = torch.zeros(1, len(lv), H // P["cell_px"], W // P["cell_px"], device="cuda", requires_grad=True)
        at = lambda X: sample_grid(X, p["uv"])
        steps = P["grid_steps"]
    else:
        X = torch.zeros(N if mode == "gaussian" else 1, len(lv), device="cuda", requires_grad=True)
        at = (lambda X: X) if mode == "gaussian" else (lambda X: X.expand(N, -1))
        steps = P["steps"] if mode == "gaussian" else 300
    opt = torch.optim.Adam([X], lr=P["lr_t"], betas=(0.9, 0.99))
    nb = knn(p, H, W, P["knn"]) if mode == "gaussian" else None
    for it in range(steps):
        k = (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * it / steps))) * min(1.0, (it + 1) / P["warmup"])
        opt.param_groups[0]["lr"] = P["lr_t"] * k
        lt = at(X)
        loss = 0.0
        for j, l in enumerate(lv):
            loss = loss + F.mse_loss(render_t(p, torch.exp(lt[:, j:j + 1]), H >> l, W >> l), gts[l])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if nb is not None:                   # SAD-style tau diffusion: Jacobi steps of (I + lam L) g = g_raw
            g0 = X.grad.clone()
            g = g0
            for _ in range(P["diffuse_iters"]):
                g = (g0 + P["diffuse_lam"] * g[nb].sum(1)) / (1 + P["diffuse_lam"] * P["knn"])
            X.grad = g
        opt.step()
        with torch.no_grad():
            X.clamp_(math.log(1 / 64), math.log(4))
    return at(X).detach(), X.detach()


def slope(lt, levels):
    """b in log tau = b log r (through tau(1) = 1), least squares over the levels; lt: (..., L) on the last axis."""
    x = lt.new_tensor([-l * LN2 for l in levels])
    return (lt * x).sum(-1, keepdim=True) / (x * x).sum()


def level_psnrs(p, gts, H, W, fields):
    N = p["uv"].shape[0]
    with torch.no_grad():
        full = render_params(p, H, W, "wnorm").clamp(0, 1)
        rows = {}
        for j, l in enumerate(sorted(gts)):
            h, w = H >> l, W >> l
            rows[l] = {"point τ = 1": psnr(render_t(p, torch.ones(N, 1, device="cuda"), h, w), gts[l])}
            for name, lt in fields.items():
                rows[l][name] = psnr(render_t(p, torch.exp(lt[:, j:j + 1]), h, w), gts[l])
            rows[l]["area (ideal)"] = psnr(F.avg_pool2d(full[None], 2 ** l)[0], gts[l])
    return rows


def edge_density(a, P, cell):
    """Fraction of line-art pixels in every cell (gh, gw)."""
    from scipy.ndimage import maximum_filter
    y = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    g = np.abs(np.diff(y, axis=0, append=y[-1:])) + np.abs(np.diff(y, axis=1, append=y[:, -1:]))
    e = (maximum_filter(g, size=5) > P["edge_thr"]).astype(np.float32)
    H, W = e.shape
    return e[:H // cell * cell, :W // cell * cell].reshape(H // cell, cell, W // cell, cell).mean((1, 3))


def font(px):
    try:
        return ImageFont.truetype("arial.ttf", px)
    except OSError:
        return ImageFont.load_default()


def grid_figure(rows, header, thumb, roi_boxes, P, out_png, foot=None):
    """rows: list of tile lists [(title, PIL)]; thumbnail with the ROI box of each row in column 0."""
    T, pad, fp = P["tile_px"], 8, P["font_px"]
    f, sf = font(fp), font(fp - 3)
    n, top = len(rows[0]), fp + 12
    fh = (fp + 6) if foot else 0
    canvas = Image.new("RGB", (pad + (n + 1) * (T + pad), top + fp + 8 + len(rows) * (T + pad) + fh), "white")
    d = ImageDraw.Draw(canvas)
    d.text((pad, 4), header, fill="black", font=f)
    top += fp + 8
    s_ = T / max(thumb.size)
    for i, row in enumerate(rows):
        yy = top + i * (T + pad)
        th = thumb.resize((round(thumb.width * s_), round(thumb.height * s_)), Image.LANCZOS)
        y0, x0, c = roi_boxes[i]
        ImageDraw.Draw(th).rectangle([x0 * s_, y0 * s_, (x0 + c) * s_, (y0 + c) * s_], outline=(255, 40, 40), width=2)
        canvas.paste(th, (pad + (T - th.width) // 2, yy + (T - th.height) // 2))
        for j, (t, im) in enumerate(row):
            x = pad + (j + 1) * (T + pad)
            canvas.paste(im.resize((T, T), Image.NEAREST if im.width < T else Image.LANCZOS), (x, yy))
            if i == 0:
                d.text((x, top - fp - 6), t, fill="black", font=f)
    if foot:
        for j, t in enumerate(foot):
            if t:
                d.text((pad + (j + 1) * (T + pad), top + len(rows) * (T + pad) - 2), t, fill="black", font=sf)
    canvas.save(out_png)


def roi_of(r, a, c):
    return pick_crop(a, c) if r == "edges" else pick_smooth(a, c) if r == "smooth" else tuple(r)


def splat(p, val, H, W, long=1000):
    sc = long / max(H, W)
    q = dict(p)
    q["color"] = val.reshape(-1, 1).float().repeat(1, 3)
    with torch.no_grad():
        return render_params(q, round(H * sc), round(W * sc), "wnorm")[0].cpu().numpy()


def field_img(p, lt_col, Xf, j, H, W, P):
    """What to show for level j: the grid itself, or the per-Gaussian values splatted."""
    if P["field"] == "grid":
        return Xf[0, j].cpu().numpy()
    return splat(p, lt_col, H, W)


def fig_maps(a, p, lt, Xf, b_img, levels, P, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = a.shape[:2]
    n = len(levels) + 2
    fig, ax = plt.subplots(1, n, figsize=(3.4 * n, 3.4 * H / W + 0.6), constrained_layout=True)
    ax[0].imshow(a)
    ax[0].set_title("Image")
    ext = (0, W, H, 0)
    for j, l in enumerate(levels):
        im = ax[j + 1].imshow(field_img(p, lt[:, j], Xf, j, H, W, P) / LN2, cmap="magma", vmin=-3.5, vmax=0.5,
                              extent=ext, interpolation="bilinear")
        ax[j + 1].set_title(f"log2 τ at 1/{2 ** l}")
    fig.colorbar(im, ax=ax[1:n - 1], shrink=0.8, label="log2 τ")
    im = ax[-1].imshow(b_img, cmap="turbo", vmin=0, vmax=1.2, extent=ext, interpolation="bilinear")
    ax[-1].set_title("sharpening exponent b")
    fig.colorbar(im, ax=ax[-1], shrink=0.8)
    for x in ax:
        x.axis("off")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_curves(Lc, bc, dens, lt_glob, levels, lp, out_png):
    """Lc: (cells, L) log tau per region; bc: (cells,) b; dens: (cells,) line-art density."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    xs = [0] + [-l for l in levels]
    bins = [(0, 0.02, "#9e9e9e", "almost no line art"), (0.02, 0.15, "#2c7fb8", "some line art"),
            (0.15, 1.01, "#d7301f", "dense line art")]
    for lo, hi, col, name in bins:
        m = (dens >= lo) & (dens < hi)
        if m.sum() < 5:
            continue
        q = np.percentile(Lc[m] / LN2, [25, 50, 75], axis=0)
        ax[0].plot(xs, [0] + list(q[1]), "o-", color=col, label=f"{name} ({m.mean() * 100:.0f}% of regions)")
        ax[0].fill_between(xs, [0] + list(q[0]), [0] + list(q[2]), color=col, alpha=0.15)
    ax[0].plot(xs, [0] + list(lt_glob / LN2), "k--", label="one τ per level")
    ax[0].plot(xs, xs, ":", color="0.5", label="b = 1 (τ ∝ r)")
    ax[0].axhline(0, color="0.7", lw=0.8)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(["1"] + [f"1/{2 ** l}" for l in levels])
    ax[0].set_xlabel("resolution r")
    ax[0].set_ylabel("log2 τ per region (median, IQR)")
    ax[0].legend(fontsize=8)
    ax[1].scatter(dens, bc, s=4, alpha=0.35, color="0.3")
    edges = np.linspace(0, max(dens.max(), 1e-3), 11)
    mid, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dens >= lo) & (dens <= hi)
        if m.sum() >= 5:
            mid.append((lo + hi) / 2)
            med.append(np.median(bc[m]))
    ax[1].plot(mid, med, "o-", color="#d7301f", label="median")
    ax[1].set_xlabel("line-art pixel fraction in the region")
    ax[1].set_ylabel("sharpening exponent b")
    ax[1].legend(fontsize=8)
    names = list(next(iter(lp.values())).keys())
    w = 0.8 / len(names)
    for i, nm in enumerate(names):
        ax[2].bar(np.arange(len(levels)) + (i - (len(names) - 1) / 2) * w, [lp[l][nm] for l in levels], w, label=nm)
    ax[2].set_xticks(range(len(levels)))
    ax[2].set_xticklabels([f"1/{2 ** l}" for l in levels])
    ax[2].set_ylabel("PSNR vs area-downscaled image (dB)")
    ax[2].set_ylim(min(min(v.values()) for v in lp.values()) - 2, max(max(v.values()) for v in lp.values()) + 1)
    ax[2].legend(fontsize=8)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def tau_S(b, S, kappa, cap):
    return torch.exp(kappa * b.clamp(0, 1.5) * math.log(S)).clamp(max=cap)


def learn_b(p, gt_pyr, H, W, P):
    """b at the Gaussians (N, 1), log tau at the Gaussians (N, L) and the field (grid or per Gaussian)."""
    lt, Xf = probe(p, gt_pyr, H, W, P, P["field"])
    if P["field"] == "grid":
        bg = slope(Xf[0].permute(1, 2, 0), list(P["levels"]))[..., 0]      # (gh, gw)
        return sample_grid(bg[None, None], p["uv"]), lt, Xf, bg
    return slope(lt, list(P["levels"])), lt, Xf, None


def main(P=PARAMS):
    OUT.mkdir(exist_ok=True)
    f = Path(P["image"])
    f = f if f.is_absolute() else (HERE / f).resolve()
    im = Image.open(f).convert("RGB")
    m = 2 ** max(P["levels"]) * max(list(P["holdout_S"]) + [1])
    m = m * P["cell_px"] // math.gcd(m, P["cell_px"])
    W, H = im.width - im.width % m, im.height - im.height % m
    a = np.asarray(im.crop((0, 0, W, H)), np.float32) / 255
    stem = f.stem.replace(".", "_")
    gt = to_t(a)
    pil = im.crop((0, 0, W, H))
    summ = {"image": f.name, "W": W, "H": H, "field": P["field"]}
    lv = list(P["levels"])

    # 1. temperature per level, the same Gaussians
    print(f"[1] {f.name} {W}x{H}: temperatures at 1/{', 1/'.join(str(2 ** l) for l in lv)} ({P['field']})")
    p = base_fit(gt, P["K_per_px"], P["fit_steps"], stem)
    gts = pyramid(gt, lv)
    b, lt, Xf, bg = learn_b(p, gts, H, W, P)
    lt_glob, _ = probe(p, gts, H, W, P, "global")
    fields = {"one τ per level": lt_glob, ("τ per region" if P["field"] == "grid" else "τ per Gaussian"): lt}
    lp = level_psnrs(p, gts, H, W, fields)
    for l in lv:
        print(f"  1/{2 ** l}: " + "  ".join(f"{k} {v:.2f}" for k, v in lp[l].items()))
    cell = P["cell_px"]
    dens = edge_density(a, P, cell).ravel()
    if P["field"] == "grid":
        Lc = Xf[0].permute(1, 2, 0).reshape(-1, len(lv)).cpu().numpy()
        bc = bg.cpu().numpy().ravel()
        b_img = bg.cpu().numpy()
    else:                                    # per Gaussian: average into the same cells for the curves
        ij = np.clip((p["uv"].cpu().numpy() * [W // cell, H // cell]).astype(int), 0, [W // cell - 1, H // cell - 1])
        idx = ij[:, 1] * (W // cell) + ij[:, 0]
        cnt = np.maximum(np.bincount(idx, minlength=len(dens)), 1)
        Lc = np.stack([np.bincount(idx, lt[:, j].cpu().numpy(), len(dens)) / cnt for j in range(len(lv))], 1)
        bc = np.bincount(idx, b[:, 0].cpu().numpy(), len(dens)) / cnt
        b_img = splat(p, b[:, 0], H, W)
    for lo, hi, name in ((0, 0.02, "no line art"), (0.02, 0.15, "some line art"), (0.15, 1.01, "dense line art")):
        mm = (dens >= lo) & (dens < hi)
        if mm.any():
            print(f"  regions with {name:15s} {mm.mean() * 100:5.1f}%: b median {np.median(bc[mm]):.2f} "
                  f"(IQR {np.percentile(bc[mm], 25):.2f}..{np.percentile(bc[mm], 75):.2f})")
    tg = lt_glob[0].cpu().numpy()
    print("  one tau per level: " + ", ".join(f"1/{2 ** l}: {math.exp(v):.3f}" for l, v in zip(lv, tg)))
    summ["level_psnr"] = {f"1/{2 ** l}": lp[l] for l in lv}
    summ["tau_global"] = {f"1/{2 ** l}": math.exp(float(v)) for l, v in zip(lv, tg)}
    summ["b_vs_line_density"] = {"corr": float(np.corrcoef(dens, bc)[0, 1]) if bc.std() > 0 else 0.0}
    print(f"  corr(b, line-art density) = {summ['b_vs_line_density']['corr']:.2f}")
    fig_maps(a, p, lt, Xf, b_img, lv, P, OUT / f"{stem}_maps.png")
    fig_curves(Lc, bc, dens, tg, lv, lp, OUT / f"{stem}_curves.png")

    # 2. upscale the original directly, tau = S^(kappa b)
    S, c, mg = P["scale"], P["roi_size"], 8
    rows, boxes = [], []
    for r in P["rois"]:
        y0, x0 = roi_of(r, a, c)
        X0, Y0, C = x0 - mg, y0 - mg, c + 2 * mg
        cut = lambda q: q.crop((mg * S, mg * S, (mg + c) * S, (mg + c) * S))
        win = pil.crop((X0, Y0, X0 + C, Y0 + C))
        row = [(f"Input (nearest ×{S})", cut(win.resize((C * S, C * S), Image.NEAREST))),
               (f"Bicubic ×{S}", cut(win.resize((C * S, C * S), Image.BICUBIC))),
               (f"Lanczos ×{S}", cut(win.resize((C * S, C * S), Image.LANCZOS)))]
        with torch.no_grad():
            for kp in P["kappas"]:
                q = render_t(p, tau_S(b, S, kp, P["tau_cap"]), H, W, s=S, win=(X0, Y0, C), down=False,
                             cull=5 / 3, eps=1e-30)
                row.append((f"Ours, κ = {kp:g}" + (" (plain)" if kp == 0 else ""), cut(to_pil(q))))
        rows.append(row)
        boxes.append((y0, x0, c))
    grid_figure(rows, f"{f.name} {W}×{H}, K = {p['uv'].shape[0]:,}; τ = {S}^(κ b), b per "
                      f"{'region' if P['field'] == 'grid' else 'Gaussian'}", pil, boxes, P, OUT / f"{stem}_x{S}.png")

    # 3. holdout: input = original / S (area), b learned on the input's own pyramid, rendered back at S x
    from protocol15 import lpips_net
    summ["holdout"] = {}
    for Sh in P["holdout_S"]:
        X = F.avg_pool2d(gt[None], Sh)[0]
        h, w = H // Sh, W // Sh
        print(f"[3] holdout x{Sh}: input {w}x{h}")
        pX = base_fit(X, P["K_per_px"], P["fit_steps"], f"{stem}_in{Sh}")
        PX = dict(P, cell_px=max(P["cell_px"] // Sh, 2 ** max(lv)))
        bX = learn_b(pX, pyramid(X, lv), h, w, PX)[0]
        Xp = to_pil(X)
        preds = {"Bicubic": to_t(np.asarray(Xp.resize((W, H), Image.BICUBIC), np.float32) / 255),
                 "Lanczos": to_t(np.asarray(Xp.resize((W, H), Image.LANCZOS), np.float32) / 255)}
        with torch.no_grad():
            for kp in P["kappas"]:
                preds[f"Ours, κ = {kp:g}" + (" (plain)" if kp == 0 else "")] = render_t(
                    pX, tau_S(bX, Sh, kp, P["tau_cap"]), H, W, cull=5 / 3, eps=1e-30).clamp(0, 1).nan_to_num(0)
            met = {k: (psnr(v, gt), float(lpips_net(v, gt, "alex"))) for k, v in preds.items()}
        for k, (ps, lpv) in met.items():
            print(f"  {k:22s} PSNR {ps:6.2f}  LPIPS {lpv:.4f}")
        summ["holdout"][f"x{Sh}"] = {k: {"psnr": ps, "lpips": lpv} for k, (ps, lpv) in met.items()}
        ch = P["roi_hold"]
        rows, boxes = [], []
        for r in P["rois"]:
            y0, x0 = roi_of(r, a, ch)
            row = [("Ground truth", pil.crop((x0, y0, x0 + ch, y0 + ch)))]
            row += [(k, to_pil(v[:, y0:y0 + ch, x0:x0 + ch])) for k, v in preds.items()]
            rows.append(row)
            boxes.append((y0, x0, ch))
        foot = [""] + [f"{met[k][0]:.2f} dB · LPIPS {met[k][1]:.4f}" for k in preds]
        grid_figure(rows, f"Holdout ×{Sh}: input = {W}×{H} / {Sh} (area); whole-image scores below", pil, boxes, P,
                    OUT / f"{stem}_holdout_x{Sh}.png", foot=foot)

    # 4. b under different budgets
    if P["budgets"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        print("[4] budgets")
        res = []
        for kpp in P["budgets"]:
            pk = base_fit(gt, kpp, P["fit_steps"], stem)
            bk, _, _, bgk = learn_b(pk, gts, H, W, P)
            bck = bgk.cpu().numpy().ravel() if bgk is not None else None
            img = bgk.cpu().numpy() if bgk is not None else splat(pk, bk[:, 0], H, W)
            q = {name: np.percentile(bck[(dens >= lo) & (dens < hi)], [25, 50, 75])
                 for lo, hi, name in ((0, 0.02, "no line art"), (0.02, 0.15, "some line art"),
                                      (0.15, 1.01, "dense line art"))
                 if bck is not None and ((dens >= lo) & (dens < hi)).sum() >= 5}
            res.append((kpp, pk["uv"].shape[0], img, q))
            print(f"  {kpp:g}/px (K={pk['uv'].shape[0]:,}): " + "  ".join(f"{nm} b {v[1]:.2f}" for nm, v in q.items()))
        n = len(res)
        fig, ax = plt.subplots(1, n + 1, figsize=(3.4 * (n + 1), 3.4 * H / W + 0.7), constrained_layout=True)
        for i, (kpp, K, img, _) in enumerate(res):
            imh = ax[i].imshow(img, cmap="turbo", vmin=0, vmax=1.2, extent=(0, W, H, 0), interpolation="bilinear")
            ax[i].set_title(f"b, {kpp:g} Gaussians/px\n(K = {K:,})")
            ax[i].axis("off")
        fig.colorbar(imh, ax=ax[:n], shrink=0.8, label="sharpening exponent b")
        xs = [r_[0] for r_ in res]
        for nm, col in (("no line art", "#9e9e9e"), ("some line art", "#2c7fb8"), ("dense line art", "#d7301f")):
            qq = [r_[3].get(nm) for r_ in res]
            if all(v is not None for v in qq):
                qq = np.array(qq)
                ax[n].plot(xs, qq[:, 1], "o-", color=col, label=nm)
                ax[n].fill_between(xs, qq[:, 0], qq[:, 2], color=col, alpha=0.15)
        ax[n].set_xscale("log")
        ax[n].set_xlabel("Gaussians per pixel")
        ax[n].set_ylabel("b per region (median, IQR)")
        ax[n].legend(fontsize=8)
        fig.savefig(OUT / f"{stem}_budgets.png", dpi=150)
        plt.close(fig)
        summ["budgets"] = {f"{r_[0]:g}": {nm: float(v[1]) for nm, v in r_[3].items()} for r_ in res}
    with open(OUT / "summary.json", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summ) + "\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
