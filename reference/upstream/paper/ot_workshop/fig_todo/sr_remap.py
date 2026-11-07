"""v2 边缘处理：放大 S 倍后，让真正的边缘过渡带保持固定的输出像素宽度（像 Real-ESRGAN），内部渐变不动。

field（默认）—— 和矢量字体的距离场放大（Green, SIGGRAPH 2007）同一原理：
  在半径 ≈ window × S 的窗口里找亮度最暗、最亮的两个点，取它们的实际颜色 D、B；
  t = (Y − Y_min) / (Y_max − Y_min) 是一张局部归一化的距离场，0.5 处就是边界；
  α = smoothstep(½ − w, ½ + w, t)，w = ½ · edge_px · |∇t|（t 在一个输出像素上的变化量，即 fwidth）
  → 过渡带正好 edge_px 个输出像素，与放大倍数无关；输出 = D + (B − D) · α，三个通道共用一条曲线（不出色点）。
  edge_px 是平滑度：默认 1.4，小 = 更硬，大 = 更柔和。
sigmoid —— 旧版（软切换，Kramer–Bruckner 切换算子的平滑版）：每个通道单独用温度为 τ 的 S 形曲线拉陡，
  过渡带约 4.4 · S / τ 像素，会随倍数变宽。
两种都只在局部亮度反差 > c0 的地方生效（> c1 完全生效），内部渐变原样保留；纯渲染后处理，实时。
输入应是干净、连续的放大图（高斯直接渲染）；用在 Lanczos 上会把振铃一起放大。
"""
import torch
import torch.nn.functional as F

LUMA = (0.299, 0.587, 0.114)


def _smoothstep(e0, e1, x):
    u = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return u * u * (3 - 2 * u)


def _gate(c, c0, c1):
    return _smoothstep(torch.as_tensor(c0), torch.as_tensor(max(c1, c0 + 1e-6)), c)


def edge_field(img, S, edge_px=1.4, window=0.75, c0=0.08, c1=0.2):
    """img: (3, H, W) in [0, 1] at the S x output resolution. Edges become edge_px output pixels wide."""
    r = max(1, int(round(S * window)))
    _, H, W = img.shape
    y = (img * img.new_tensor(LUMA).view(3, 1, 1)).sum(0, keepdim=True)[None]          # (1, 1, H, W)
    ymax, imax = F.max_pool2d(y, 2 * r + 1, 1, r, return_indices=True)
    ymin, imin = F.max_pool2d(-y, 2 * r + 1, 1, r, return_indices=True)
    ymin = -ymin
    flat = img.reshape(3, -1)
    B = flat[:, imax.view(-1)].view(3, H, W)                                          # colour of the brightest point
    D = flat[:, imin.view(-1)].view(3, H, W)                                          # colour of the darkest point
    rng = (ymax - ymin)[0]
    t = ((y[0] - ymin[0]) / (rng + 1e-6)).clamp(0, 1)
    gx = torch.zeros_like(t)
    gy = torch.zeros_like(t)
    gx[..., 1:-1] = (t[..., 2:] - t[..., :-2]) / 2
    gy[..., 1:-1, :] = (t[..., 2:, :] - t[..., :-2, :]) / 2
    w = (0.5 * edge_px * (gx.abs() + gy.abs())).clamp(1e-3, 0.5)
    a = _smoothstep(0.5 - w, 0.5 + w, t)
    sharp = D + (B - D) * a
    return (img + _gate(rng, c0, c1) * (sharp - img)).clamp(0, 1)


def remap(img, S, tau=12.0, window=0.75, c0=0.08, c1=0.2):
    """Old soft toggle mapping (per channel, sigmoid of temperature tau)."""
    r = max(1, int(round(S * window)))
    x = img[None]
    lo = -F.max_pool2d(-x, 2 * r + 1, 1, r)
    hi = F.max_pool2d(x, 2 * r + 1, 1, r)
    rng = hi - lo
    t = ((x - lo) / (rng + 1e-6)).clamp(0, 1)
    s0, s1 = torch.sigmoid(torch.tensor(-tau / 2)), torch.sigmoid(torch.tensor(tau / 2))
    sharp = lo + rng * (torch.sigmoid(tau * (t - 0.5)) - s0) / (s1 - s0)
    w = ((rng.amax(1, keepdim=True) - c0) / max(c1 - c0, 1e-6)).clamp(0, 1)
    w = w * w * (3 - 2 * w)
    return (x + w * (sharp - x))[0].clamp(0, 1)


def sharpen(img, S, mode="field", edge_px=1.4, tau=12.0, window=0.75, c0=0.08, c1=0.2):
    return edge_field(img, S, edge_px, window, c0, c1) if mode == "field" else remap(img, S, tau, window, c0, c1)
