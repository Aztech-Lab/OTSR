"""The figures of README.md, drawn in the OTSR design (docs/assets/*.svg, a light and a dark file each).

    python tools/readme-figures.py

Hand-authored SVG with CSS animations (they play inside GitHub's README images; they stop under
prefers-reduced-motion). No web fonts: the same system stacks as the app. The hero shows the start screen of the
app with its boot log; the pipeline figure shows the four stages of the method.
"""
import io
import math
import os
from string import Template

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets')

LIGHT = dict(BG='#ebebe8', BG2='#e2e2de', DOT='rgba(20,22,20,0.16)', INK='#111312', TEXT='#24261f', MUTED='#6f726a',
             FAINT='#a3a59d', LINE='#c7c8c0', LINES='#24261f', PANEL='rgba(246,246,243,0.55)', PANELS='#f4f4f1',
             HEAD='#1a1c1a', HEADT='#e8e8e2', COPPER='#b8793c', COPPER2='#d9a25f', OK='#4f7a4a',
             SOLID='#1a1c1a', SOLIDT='#f2f1ea', PALE='#ffffff', PALET='#111312', ONCOPPER='#ffffff', GLOW='rgba(184,121,60,0.55)')
DARK = dict(BG='#0f1112', BG2='#151819', DOT='rgba(232,232,226,0.12)', INK='#f1f1ea', TEXT='#e2e3dc', MUTED='#9ba09a',
            FAINT='#5e635f', LINE='#2b302f', LINES='#e2e3dc', PANEL='rgba(21,24,25,0.55)', PANELS='#161919',
            HEAD='#e6e6df', HEADT='#121413', COPPER='#d9a25f', COPPER2='#f0c389', OK='#8fbf7a',
            SOLID='#e6e6df', SOLIDT='#121413', PALE='#e6e6df', PALET='#121413', ONCOPPER='#111312', GLOW='rgba(217,162,95,0.5)')

SANS = "system-ui,-apple-system,'Segoe UI',Roboto,'PingFang SC','Microsoft YaHei',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,'Cascadia Mono','SF Mono',Menlo,Consolas,'DejaVu Sans Mono',monospace"


def chamfer(x, y, w, h, c=8, tl=True, tr=True, br=True, bl=True):
    """A rectangle with cut corners, as polygon points (the app's clip-path)."""
    pts = []
    pts += [(x + c, y), ] if tl else [(x, y)]
    pts += [(x + w - c, y), (x + w, y + c)] if tr else [(x + w, y)]
    pts += [(x + w, y + h - c), (x + w - c, y + h)] if br else [(x + w, y + h)]
    pts += [(x + c, y + h), (x, y + h - c)] if bl else [(x, y + h)]
    if tl:
        pts += [(x, y + c)]
    return ' '.join(f'{px:.1f},{py:.1f}' for px, py in pts)


def bracket(cx, cy, half, leg=22, sw=2):
    """Four corner brackets around a square, the instrument's frame."""
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * half, cy + sy * half
            out.append(f'<path d="M{x:.0f} {y + sy * -leg:.0f} V{y:.0f} H{x + sx * -leg:.0f}" stroke="$COPPER" stroke-width="{sw}" fill="none"/>')
    return '\n'.join(out)


COMMON_STYLE = """
  .sans { font-family: @SANS@; }
  .mono { font-family: @MONO@; }
  .spin { transform-origin: 300px 262px; animation: spin 90s linear infinite; }
  .spinr { transform-origin: 300px 262px; animation: spin 140s linear infinite reverse; }
  .blink { animation: blink 1.1s steps(1) infinite; }
  .d1 { animation: dot 1.6s steps(1) infinite; }
  .d2 { animation: dot 1.6s steps(1) .4s infinite; }
  .d3 { animation: dot 1.6s steps(1) .8s infinite; }
  .glow { animation: glow 2.6s ease-in-out infinite; }
  .star { animation: star 4s ease-in-out infinite; }
  .boot { animation: boot 11s steps(1) infinite; }
  .seg { animation: seg 11s steps(1) infinite; }
  .gauss { animation: grow 3.2s ease-in-out infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes blink { 0%, 49% { opacity: 1; } 50%, 100% { opacity: 0; } }
  @keyframes dot { 0%, 24% { opacity: 0; } 25%, 100% { opacity: 1; } }
  @keyframes glow { 0%, 100% { opacity: 1; } 50% { opacity: .55; } }
  @keyframes star { 0%, 100% { opacity: .25; } 50% { opacity: .75; } }
  @keyframes grow { 0%, 100% { transform: scale(.55); opacity: .35; } 50% { transform: scale(1); opacity: .9; } }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
""".replace('@SANS@', SANS).replace('@MONO@', MONO)


def hero_svg():
    W, H = 1280, 520
    cx, cy = 300, 262
    boot_lines = ['SYS.INIT // OTSR 0.1.4', 'WEBGPU ADAPTER // YOUR GPU', 'KERNELS // 16 PIPELINES COMPILED', 'SYSTEM CHECK // READY',
                  'READY // DROP, PASTE OR OPEN AN IMAGE']
    # the boot log types itself: line k appears at (6 + 7k)% of the loop, all lines leave at 94%
    boot_css = ''.join(f'  .b{k} {{ animation-delay: 0s; }} @keyframes boot{k} {{ 0%, {6 + 7 * k}% {{ opacity: 0; }} {6 + 7 * k + 0.01}%, 94% {{ opacity: 1; }} 94.01%, 100% {{ opacity: 0; }} }}\n  .b{k} {{ animation: boot{k} 11s linear infinite; }}\n' for k in range(len(boot_lines)))
    seg_css = ''.join(f'  @keyframes seg{k} {{ 0%, {40 + 4 * k}% {{ opacity: .25; }} {40 + 4 * k + 0.01}%, 94% {{ opacity: 1; }} 94.01%, 100% {{ opacity: .25; }} }}\n  .s{k} {{ animation: seg{k} 11s linear infinite; }}\n' for k in range(12))
    stars = [(120, 90, 18, 0), (470, 420, 14, 1.3), (1130, 120, 22, 2.1), (1010, 440, 16, .7), (720, 60, 12, 3.2), (900, 480, 10, 1.9)]
    star_svg = '\n'.join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="$COPPER2" filter="url(#blur)" class="star" style="animation-delay:{d}s"/>'
                         f'<path d="M{x - r * 1.6:.0f} {y} H{x + r * 1.6:.0f} M{x} {y - r * 1.6:.0f} V{y + r * 1.6:.0f}" stroke="$COPPER2" stroke-width=".8" opacity=".5" class="star" style="animation-delay:{d}s"/>' for x, y, r, d in stars)
    # the strip of Gaussians (bottom right): elongated along a gentle wave, each one breathing in turn
    gauss = []
    cols = ['$COPPER', '$COPPER2', '$OK', '$MUTED', '$TEXT']
    for i in range(14):
        gx = 620 + i * 38
        gy = 428 + 10 * math.sin(i / 2.2)
        ang = -math.degrees(math.atan(10 / 2.2 * math.cos(i / 2.2) / 38)) * 1.6
        rx, ry = 20 - 5 * (i % 3), 5 + 2 * (i % 2)
        gauss.append(f'<g transform="translate({gx:.0f} {gy:.1f})"><g class="gauss" style="animation-delay:{i * 0.23:.2f}s"><ellipse rx="{rx}" ry="{ry}" transform="rotate({ang:.0f})" fill="{cols[i % 5]}"/></g></g>')
    gauss_svg = '\n'.join(gauss)
    boot_svg = '\n'.join(f'<text x="600" y="{300 + 19 * k}" class="mono b{k}" font-size="11" letter-spacing="1" fill="{"$TEXT" if k == len(boot_lines) - 1 else "$MUTED"}"><tspan fill="$COPPER">› </tspan>{line}{"<tspan fill=\"$COPPER\" class=\"blink\"> ▌</tspan>" if k == len(boot_lines) - 1 else ""}</text>' for k, line in enumerate(boot_lines))
    segs = '\n'.join(f'<rect x="{1000 + k * 14}" y="496" width="10" height="4" fill="$COPPER" class="s{k}"/>' for k in range(12))
    chips = []
    x = 600
    for t in ('ANY RESOLUTION · ANY ASPECT RATIO', 'ON-DEVICE · NOTHING UPLOADED', 'NO NEURAL WEIGHTS'):
        w = round(len(t) * 6.9 + 26)
        chips.append(f'<polygon points="{chamfer(x, 212, w, 28, 6)}" fill="none" stroke="$COPPER" stroke-width="1"/><text x="{x + w / 2:.0f}" y="230" class="mono" font-size="9.5" font-weight="700" letter-spacing="1.2" text-anchor="middle" fill="$COPPER">{t}</text>')
        x += w + 14
    chip_svg = '\n'.join(chips)
    hatch = ''.join(f'<rect x="{600 + i * 4}" y="170" width="{1 if i % 3 else 2}" height="12" fill="$FAINT"/>' for i in range(0, 20))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">
  <title id="t">OTSR — optimal-transport Gaussian super-resolution in the browser</title>
  <desc id="d">The start screen of OTSR: the instrument with its rings waiting for an image, the wordmark, three claims (any resolution, on-device, no neural weights), the boot log reporting the GPU ready, and a strip of anisotropic Gaussians.</desc>
  <style>{COMMON_STYLE}{boot_css}{seg_css}  </style>
  <defs>
    <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="11" cy="11" r="1.1" fill="$DOT"/></pattern>
    <pattern id="grid" width="110" height="110" patternUnits="userSpaceOnUse"><path d="M110 0 V110 M0 110 H110" stroke="$LINE" stroke-width=".6" opacity=".5"/></pattern>
    <filter id="blur" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="9"/></filter>
    <filter id="textglow" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="{W}" height="{H}" fill="$BG"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <rect width="{W}" height="{H}" fill="url(#dots)"/>
  {star_svg}
  <!-- the instrument -->
  <g fill="none" stroke="$COPPER">
    <circle cx="{cx}" cy="{cy}" r="236" stroke-width="1" stroke-dasharray="3 9" opacity=".9" class="spin"/>
    <circle cx="{cx}" cy="{cy}" r="196" stroke-width="2.5" stroke-dasharray="190 1050" stroke-linecap="round" class="spinr"/>
    <circle cx="{cx}" cy="{cy}" r="150" stroke-width=".8" stroke-dasharray="1 6" opacity=".8" class="spin"/>
    <circle cx="{cx}" cy="{cy}" r="104" stroke-width=".8" opacity=".5"/>
    <path d="M{cx} {cy - 236} v10 M{cx} {cy + 236} v-10 M{cx - 236} {cy} h10 M{cx + 236} {cy} h-10" stroke-width="1.5"/>
  </g>
  {bracket(cx, cy, 170)}
  <!-- input detecting -->
  <g class="glow" filter="url(#textglow)">
    <rect x="{cx - 76}" y="156" width="7" height="7" fill="$COPPER" class="blink"/>
    <text x="{cx - 62}" y="163" class="mono" font-size="11" letter-spacing="2.2" fill="$COPPER">INPUT DETECTING<tspan class="d1">.</tspan><tspan class="d2">.</tspan><tspan class="d3">.</tspan></text>
  </g>
  <text x="{cx}" y="212" class="sans" font-size="34" font-weight="900" letter-spacing="5" text-anchor="middle" fill="$INK">DROP IMAGE</text>
  <text x="{cx}" y="238" class="sans" font-size="12" letter-spacing="5" text-anchor="middle" fill="$MUTED">拖入 · 貼上 · 選擇</text>
  <!-- the three actions -->
  <polygon points="{chamfer(170, 262, 130, 44, 8, tl=False, tr=False, br=False)}" fill="$PALE"/>
  <text x="186" y="281" class="mono" font-size="11" font-weight="700" letter-spacing="1.5" fill="$PALET">LOAD SAMPLE</text>
  <text x="186" y="296" class="sans" font-size="9" letter-spacing="1.5" fill="$PALET" opacity=".75">載入樣本</text>
  <text x="284" y="289" class="mono" font-size="13" fill="$PALET">↓</text>
  <polygon points="{chamfer(300, 262, 130, 44, 8, tl=False, bl=False, br=False)}" fill="$COPPER"/>
  <text x="316" y="281" class="mono" font-size="11" font-weight="700" letter-spacing="1.5" fill="$ONCOPPER">OPEN IMAGE</text>
  <text x="316" y="296" class="sans" font-size="9" letter-spacing="1.5" fill="$ONCOPPER" opacity=".8">開啟圖片</text>
  <text x="411" y="289" class="mono" font-size="13" fill="$ONCOPPER">→</text>
  <polygon points="{chamfer(170, 318, 260, 40, 8)}" fill="$PANEL" stroke="$LINES" stroke-width="1"/>
  <text x="186" y="335" class="mono" font-size="11" font-weight="700" letter-spacing="1.5" fill="$INK">TECHNICAL REPORT</text>
  <text x="186" y="349" class="sans" font-size="9" letter-spacing="1.5" fill="$MUTED">技術報告 · 原理與過程</text>
  <text x="408" y="343" class="mono" font-size="13" fill="$INK">↗</text>
  <text x="{cx}" y="380" class="mono" font-size="9" letter-spacing="1.8" text-anchor="middle" fill="$MUTED" text-decoration="underline">SOURCE ON GITHUB · 原始碼 ↗</text>
  <text x="{cx - 230}" y="52" class="mono" font-size="9" letter-spacing="1.2" fill="$MUTED">// ANY SIZE</text>
  <text x="{cx - 230}" y="66" class="mono" font-size="9" letter-spacing="1.2" fill="$MUTED">// ANY ASPECT RATIO</text>
  <text x="{cx + 230}" y="52" class="mono" font-size="9" letter-spacing="1.2" fill="$MUTED" text-anchor="end">ON-DEVICE</text>
  <text x="{cx + 230}" y="66" class="mono" font-size="9" letter-spacing="1.2" fill="$MUTED" text-anchor="end">NOTHING UPLOADED</text>
  <!-- the wordmark -->
  <text x="596" y="150" class="sans" font-size="98" font-weight="900" letter-spacing="10" fill="$INK">OTSR</text>
  <polygon points="{chamfer(902, 92, 88, 30, 6, tl=False, br=False)}" fill="$COPPER"/>
  <text x="946" y="112" class="mono" font-size="12" font-weight="700" letter-spacing="2" text-anchor="middle" fill="$ONCOPPER">V0.1.4</text>
  {hatch}
  <text x="690" y="181" class="mono" font-size="12" letter-spacing="3" fill="$TEXT">OPTIMAL-TRANSPORT GAUSSIAN SUPER-RESOLUTION</text>
  {chip_svg}
  <!-- the boot log -->
  {boot_svg}
  <!-- the image as Gaussians -->
  {gauss_svg}
  <text x="600" y="466" class="mono" font-size="9.5" letter-spacing="1.6" fill="$MUTED">THE IMAGE AS A MESH OF ANISOTROPIC GAUSSIANS · DENSE ALONG EDGES, SPARSE WHERE FLAT</text>
  <!-- the status strip -->
  <path d="M40 478 H1240" stroke="$LINE" stroke-width="1"/>
  <rect x="40" y="495" width="6" height="6" fill="$OK"/>
  <text x="54" y="501" class="mono" font-size="10" letter-spacing="1.4" fill="$MUTED">IDLE</text>
  <text x="140" y="501" class="mono" font-size="10" letter-spacing="1.4" fill="$MUTED">SYS.READY — awaiting input</text>
  {segs}
  <text x="1240" y="501" class="mono" font-size="10" letter-spacing="1.4" fill="$MUTED" text-anchor="end">GPU · YOURS</text>
</svg>
'''
    return svg


def pipeline_svg():
    W, H = 1280, 420
    px = [70, 370, 670, 970]  # panel x; panels 220 wide, the drawing 200 × 200 at y 76
    top = 76
    out = []
    # ---- 01 image: a pixel grid with an edge running through it -------------------------------------------------
    x0 = px[0] + 10
    cells = []
    n = 10
    s = 20
    for i in range(n):
        for j in range(n):
            cxp, cyp = x0 + j * s + s / 2, top + i * s + s / 2
            d = (cxp - x0) + (cyp - top) - 200  # signed distance to the diagonal (scaled)
            k = max(0, min(1, 0.5 - d / 40))
            tone = f'rgba(184,121,60,{0.12 + 0.55 * k:.2f})'
            cells.append(f'<rect x="{x0 + j * s}" y="{top + i * s}" width="{s}" height="{s}" fill="{tone}"/>')
    grid1 = ''.join(f'<path d="M{x0 + j * s} {top} V{top + 200}" stroke="$LINE" stroke-width=".6"/>' for j in range(n + 1)) + \
            ''.join(f'<path d="M{x0} {top + i * s} H{x0 + 200}" stroke="$LINE" stroke-width=".6"/>' for i in range(n + 1))
    out.append(f'<g>{"".join(cells)}{grid1}<polygon points="{chamfer(x0, top, 200, 200, 10)}" fill="none" stroke="$LINES" stroke-width="1"/></g>')
    # ---- 02 OT mesh: a uniform mesh morphing onto the edge ------------------------------------------------------
    x0 = px[1] + 10
    m = 7

    def warp(u, v):
        # pull the point toward the diagonal u + v = 1 along its normal: cells get dense and thin there
        d = (u + v - 1) / math.sqrt(2)
        pull = 0.62 * math.exp(-(d / 0.26) ** 2)
        d2 = d * (1 - pull)
        shift = (d2 - d) / math.sqrt(2)
        return u + shift, v + shift

    def pts(line, warped):
        res = []
        for t in range(0, 21):
            u, v = (line / m, t / 20) if line <= m else (t / 20, (line - m - 1) / m)
            if warped:
                u, v = warp(u, v)
            res.append(f'{x0 + u * 200:.1f} {top + v * 200:.1f}')
        return 'M' + ' L'.join(res)

    mesh = []
    for line in range(0, 2 * m + 2):
        a, b = pts(line, False), pts(line, True)
        mesh.append(f'<path d="{a}" fill="none" stroke="$COPPER" stroke-width=".9" opacity=".85"><animate attributeName="d" dur="7s" repeatCount="indefinite" calcMode="spline" keyTimes="0;0.15;0.55;0.75;1" keySplines=".4 0 .2 1;.4 0 .2 1;.4 0 .2 1;.4 0 .2 1" values="{a};{b};{b};{a};{a}"/></path>')
    nodes = []
    for i in range(0, m + 1, 2):
        for j in range(0, m + 1, 2):
            u, v = i / m, j / m
            wu, wv = warp(u, v)
            nodes.append(f'<rect width="3" height="3" fill="$COPPER2"><animate attributeName="x" dur="7s" repeatCount="indefinite" calcMode="spline" keyTimes="0;0.15;0.55;0.75;1" keySplines=".4 0 .2 1;.4 0 .2 1;.4 0 .2 1;.4 0 .2 1" values="{x0 + u * 200 - 1.5:.1f};{x0 + wu * 200 - 1.5:.1f};{x0 + wu * 200 - 1.5:.1f};{x0 + u * 200 - 1.5:.1f};{x0 + u * 200 - 1.5:.1f}"/><animate attributeName="y" dur="7s" repeatCount="indefinite" calcMode="spline" keyTimes="0;0.15;0.55;0.75;1" keySplines=".4 0 .2 1;.4 0 .2 1;.4 0 .2 1;.4 0 .2 1" values="{top + v * 200 - 1.5:.1f};{top + wv * 200 - 1.5:.1f};{top + wv * 200 - 1.5:.1f};{top + v * 200 - 1.5:.1f};{top + v * 200 - 1.5:.1f}"/></rect>')
    out.append(f'<g><path d="M{x0} {top + 200} L{x0 + 200} {top}" stroke="$FAINT" stroke-width="8" opacity=".25"/>{"".join(mesh)}{"".join(nodes)}<polygon points="{chamfer(x0, top, 200, 200, 10)}" fill="none" stroke="$LINES" stroke-width="1"/></g>')
    # ---- 03 Gaussians: elongated along the edge, round and sparse away from it ------------------------------------
    x0 = px[2] + 10
    cols = ['$COPPER', '$COPPER2', '$OK', '$MUTED', '$TEXT']
    g = []
    k = 0
    for t in range(0, 13):
        u = 0.06 + t * 0.075
        v = 1 - u
        for off in (-0.045, 0.045):
            uu, vv = u + off / math.sqrt(2), v + off / math.sqrt(2)
            g.append(f'<g transform="translate({x0 + uu * 200:.1f} {top + vv * 200:.1f})"><g class="gauss" style="animation-delay:{(k % 7) * 0.4:.1f}s"><ellipse rx="11" ry="3.4" transform="rotate(-45)" fill="{cols[k % 5]}" opacity=".85"/></g></g>')
            k += 1
    for i in range(5):
        for j in range(5):
            u, v = 0.1 + i * 0.2, 0.1 + j * 0.2
            if abs(u + v - 1) < 0.22:
                continue
            g.append(f'<g transform="translate({x0 + u * 200:.1f} {top + v * 200:.1f})"><g class="gauss" style="animation-delay:{((i + j) % 5) * 0.5:.1f}s"><ellipse rx="13" ry="11" transform="rotate({(i * 37 + j * 53) % 90})" fill="{cols[(i + j) % 5]}" opacity=".45"/></g></g>')
    out.append(f'<g>{"".join(g)}<polygon points="{chamfer(x0, top, 200, 200, 10)}" fill="none" stroke="$LINES" stroke-width="1"/></g>')
    # ---- 04 any scale: bicubic against OTSR at 8×, the app's split view with a scan line ----------------------------
    x0 = px[3] + 10
    out.append(f'''<g>
    <clipPath id="left"><rect x="{x0}" y="{top}" width="100" height="200"/></clipPath>
    <clipPath id="right"><rect x="{x0 + 100}" y="{top}" width="100" height="200"/></clipPath>
    <clipPath id="frame"><polygon points="{chamfer(x0, top, 200, 200, 10)}"/></clipPath>
    <linearGradient id="soft" gradientUnits="userSpaceOnUse" x1="{x0 + 70}" y1="{top + 70}" x2="{x0 + 130}" y2="{top + 130}"><stop offset="0" stop-color="$COPPER" stop-opacity=".67"/><stop offset="1" stop-color="$COPPER" stop-opacity=".12"/></linearGradient>
    <g clip-path="url(#frame)">
      <rect x="{x0}" y="{top}" width="200" height="200" fill="$COPPER" opacity=".12"/>
      <g clip-path="url(#left)"><rect x="{x0}" y="{top}" width="200" height="200" fill="url(#soft)"/>{''.join(f'<rect x="{x0 + j * 25}" y="{top + i * 25}" width="25" height="25" fill="none" stroke="$BG" stroke-width=".8" opacity=".5"/>' for i in range(8) for j in range(4))}</g>
      <g clip-path="url(#right)"><polygon points="{x0},{top} {x0 + 200},{top} {x0},{top + 200}" fill="$COPPER" opacity=".55"/></g>
      <path d="M{x0 + 100} {top} V{top + 200}" stroke="$BG" stroke-width="3"/>
      <path d="M{x0 + 100} {top} V{top + 200}" stroke="$LINES" stroke-width="1"/>
      <rect x="{x0 + 87}" y="{top + 88}" width="26" height="24" fill="$PANELS" stroke="$LINES" stroke-width="1"/>
      <text x="{x0 + 100}" y="{top + 105}" class="mono" font-size="11" text-anchor="middle" fill="$INK">‹›</text>
      <rect x="{x0}" width="200" height="2" fill="$COPPER2" opacity=".9"><animate attributeName="y" values="{top};{top + 200};{top + 200}" keyTimes="0;0.8;1" dur="3.5s" repeatCount="indefinite"/></rect>
      <rect x="{x0}" width="200" height="30" fill="url(#scan)"><animate attributeName="y" values="{top - 30};{top + 170};{top + 170}" keyTimes="0;0.8;1" dur="3.5s" repeatCount="indefinite"/></rect>
    </g>
    <polygon points="{chamfer(x0, top, 200, 200, 10)}" fill="none" stroke="$LINES" stroke-width="1"/>
    <rect x="{x0 + 8}" y="{top + 8}" width="58" height="16" fill="$PANELS" opacity=".9"/><text x="{x0 + 37}" y="{top + 20}" class="mono" font-size="9" font-weight="700" letter-spacing="1" text-anchor="middle" fill="$INK">BICUBIC</text>
    <rect x="{x0 + 134}" y="{top + 8}" width="58" height="16" fill="$PANELS" opacity=".9"/><text x="{x0 + 163}" y="{top + 20}" class="mono" font-size="9" font-weight="700" letter-spacing="1" text-anchor="middle" fill="$INK">OTSR</text>
    <text x="{x0 + 192}" y="{top + 192}" class="mono" font-size="9" letter-spacing="1" text-anchor="end" fill="$INK">×8</text>
  </g>''')
    heads = [('01', 'IMAGE', '影像'), ('02', 'OT MESH', '傳輸網格'), ('03', 'GAUSSIANS', '高斯'), ('04', 'ANY SCALE', '任意倍率')]
    caps = [('An edge-importance density is', 'computed from the pixels.'),
            ('Sinkhorn on the GPU moves a uniform', 'mesh onto it; cells pack along edges.'),
            ('Each cell becomes one anisotropic', 'Gaussian; Adam fits all nine parameters.'),
            ('A continuous function: viewed or', 'exported at any size, edges kept crisp.')]
    head_svg = ''.join(f'<text x="{x + 10}" y="58" class="mono" font-size="10" letter-spacing="1.5" fill="$COPPER">{n}</text><text x="{x + 34}" y="58" class="sans" font-size="15" font-weight="800" letter-spacing="1.5" fill="$INK">{en}</text><text x="{x + 210}" y="58" class="sans" font-size="11" letter-spacing="2" text-anchor="end" fill="$MUTED">{zh}</text><path d="M{x + 10} 66 H{x + 210}" stroke="$LINE"/>' for (n, en, zh), x in zip(heads, px))
    cap_svg = ''.join(f'<text x="{x + 10}" y="304" class="sans" font-size="11.5" fill="$TEXT">{a}</text><text x="{x + 10}" y="321" class="sans" font-size="11.5" fill="$TEXT">{b}</text>' for (a, b), x in zip(caps, px))
    arrows = ''.join(f'<path d="M{x + 226} 176 H{x + 294}" stroke="$COPPER" stroke-width="1.4" stroke-dasharray="4 7" class="flow"/><path d="M{x + 288} 171 L{x + 296} 176 L{x + 288} 181" fill="none" stroke="$COPPER" stroke-width="1.4"/>' for x in px[:3])
    labels = ''.join(f'<text x="{x + 260}" y="166" class="mono" font-size="8.5" letter-spacing="1" text-anchor="middle" fill="$MUTED">{t}</text>' for x, t in zip(px[:3], ['SINKHORN', 'JACOBIAN', 'ADAM']))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">
  <title id="t">How OTSR works, in four stages</title>
  <desc id="d">An image with an edge; a uniform mesh that optimal transport packs along the edge; one anisotropic Gaussian per mesh cell, elongated along the edge; and the result shown at 8× next to bicubic, with a crisp edge.</desc>
  <style>{COMMON_STYLE}  .flow {{ animation: flow 1.2s linear infinite; }}
  @keyframes flow {{ to {{ stroke-dashoffset: -11; }} }}
  </style>
  <defs>
    <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="11" cy="11" r="1.1" fill="$DOT"/></pattern>
    <linearGradient id="scan" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="$COPPER2" stop-opacity="0"/><stop offset="1" stop-color="$COPPER2" stop-opacity=".45"/></linearGradient>
  </defs>
  <rect width="{W}" height="{H}" fill="$BG"/>
  <rect width="{W}" height="{H}" fill="url(#dots)"/>
  {head_svg}
  {''.join(out)}
  {arrows}{labels}
  {cap_svg}
  <text x="640" y="392" class="mono" font-size="9.5" letter-spacing="1.8" text-anchor="middle" fill="$MUTED">EVERY STAGE RUNS ON YOUR OWN GPU THROUGH WEBGPU · NO SERVER · NO UPLOAD · NO NEURAL NETWORK WEIGHTS</text>
</svg>
'''
    return svg


def write(name, svg):
    for theme, tok in (('light', LIGHT), ('dark', DARK)):
        path = os.path.join(OUT, f'{name}-{theme}.svg')
        io.open(path, 'w', encoding='utf-8', newline='\n').write(Template(svg).substitute(tok))
        print(f'{path}: {os.path.getsize(path) / 1024:.1f} kB')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    write('hero', hero_svg())
    write('pipeline', pipeline_svg())
