// The one region-of-interest tool: a rectangle in source pixels, drawn and edited on the viewer's overlay and used by
// every feature that needs a region (zoom to region, comparison figures, local metrics, overlays).

export interface Roi {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type RoiHandle = 'nw' | 'ne' | 'sw' | 'se' | 'move';

export interface RoiOptions {
  /** Keep the region square (comparison figures use square crops). */
  square: boolean;
  /** Smallest side in source pixels. */
  minSize: number;
}

export interface RoiTransform {
  /** Output (canvas device) pixels per source pixel. */
  zoom: number;
  offsetX: number;
  offsetY: number;
  /** Source image size, to clamp the region. */
  W: number;
  H: number;
}

const HANDLE_PX = 10; // hit radius of corner handles in device pixels

export function clampRoi(r: Roi, W: number, H: number, minSize: number): Roi {
  const w = Math.max(minSize, Math.min(Math.round(r.w), W));
  const h = Math.max(minSize, Math.min(Math.round(r.h), H));
  const x = Math.max(0, Math.min(Math.round(r.x), W - w));
  const y = Math.max(0, Math.min(Math.round(r.y), H - h));
  return { x, y, w, h };
}

/** A sensible default region: a centred square of a quarter of the short side (at least 32 px). */
export function defaultRoi(W: number, H: number): Roi {
  const s = Math.max(32, Math.min(Math.round(Math.min(W, H) / 4), 512));
  return clampRoi({ x: (W - s) / 2, y: (H - s) / 2, w: s, h: s }, W, H, 8);
}

export class RoiTool {
  roi: Roi | null = null;
  opts: RoiOptions = { square: true, minSize: 8 };
  onChange: ((roi: Roi | null, final: boolean) => void) | null = null;
  private drag: { handle: RoiHandle | 'new'; startX: number; startY: number; start: Roi } | null = null;

  /** Which part of the region is under a source-pixel position, if any. */
  hitTest(sx: number, sy: number, t: RoiTransform): RoiHandle | null {
    const r = this.roi;
    if (!r) return null;
    const tol = HANDLE_PX / t.zoom;
    const corners: [RoiHandle, number, number][] = [['nw', r.x, r.y], ['ne', r.x + r.w, r.y], ['sw', r.x, r.y + r.h], ['se', r.x + r.w, r.y + r.h]];
    for (const [name, cx, cy] of corners) if (Math.abs(sx - cx) <= tol && Math.abs(sy - cy) <= tol) return name;
    if (sx >= r.x && sx < r.x + r.w && sy >= r.y && sy < r.y + r.h) return 'move';
    return null;
  }

  cursorFor(handle: RoiHandle | null): string {
    switch (handle) {
      case 'nw': case 'se': return 'nwse-resize';
      case 'ne': case 'sw': return 'nesw-resize';
      case 'move': return 'move';
      default: return 'crosshair';
    }
  }

  /** Starts a drag at a source-pixel position: edits the existing region or begins a new one. */
  begin(sx: number, sy: number, t: RoiTransform): void {
    const handle = this.hitTest(sx, sy, t);
    if (handle && this.roi) {
      this.drag = { handle, startX: sx, startY: sy, start: { ...this.roi } };
    } else {
      this.drag = { handle: 'new', startX: sx, startY: sy, start: { x: sx, y: sy, w: 0, h: 0 } };
      this.roi = clampRoi({ x: sx, y: sy, w: this.opts.minSize, h: this.opts.minSize }, t.W, t.H, this.opts.minSize);
    }
  }

  move(sx: number, sy: number, t: RoiTransform): void {
    const d = this.drag;
    if (!d) return;
    const dx = sx - d.startX;
    const dy = sy - d.startY;
    const s = d.start;
    let r: Roi;
    if (d.handle === 'move') {
      r = { x: s.x + dx, y: s.y + dy, w: s.w, h: s.h };
    } else if (d.handle === 'new') {
      let w = Math.abs(dx);
      let hh = Math.abs(dy);
      if (this.opts.square) w = hh = Math.max(w, hh);
      r = { x: dx < 0 ? s.x - w : s.x, y: dy < 0 ? s.y - hh : s.y, w, h: hh };
    } else {
      // corner handles: the opposite corner stays fixed
      const left = d.handle === 'nw' || d.handle === 'sw';
      const top = d.handle === 'nw' || d.handle === 'ne';
      let x0 = left ? s.x + dx : s.x;
      let y0 = top ? s.y + dy : s.y;
      let x1 = left ? s.x + s.w : s.x + s.w + dx;
      let y1 = top ? s.y + s.h : s.y + s.h + dy;
      if (this.opts.square) {
        const side = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0));
        if (left) x0 = x1 - side; else x1 = x0 + side;
        if (top) y0 = y1 - side; else y1 = y0 + side;
      }
      r = { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) };
    }
    this.roi = clampRoi(r, t.W, t.H, this.opts.minSize);
    this.onChange?.(this.roi, false);
  }

  end(): void {
    if (!this.drag) return;
    this.drag = null;
    this.onChange?.(this.roi, true);
  }

  /** Abandons the drag (a second finger turned it into a pinch): the region goes back to what it was before begin(). */
  cancel(before: Roi | null): void {
    if (!this.drag) return;
    this.drag = null;
    this.set(before);
  }

  get dragging(): boolean {
    return this.drag !== null;
  }

  /** Keyboard nudge (arrow keys): moves by `step` source pixels. */
  nudge(dx: number, dy: number, t: RoiTransform): void {
    if (!this.roi) return;
    this.roi = clampRoi({ ...this.roi, x: this.roi.x + dx, y: this.roi.y + dy }, t.W, t.H, this.opts.minSize);
    this.onChange?.(this.roi, true);
  }

  set(roi: Roi | null, t?: RoiTransform): void {
    this.roi = roi && t ? clampRoi(roi, t.W, t.H, this.opts.minSize) : roi;
    this.onChange?.(this.roi, true);
  }

  /** Draws the region on a 2-D overlay whose pixels are device pixels. */
  draw(g: CanvasRenderingContext2D, t: RoiTransform, active: boolean, dpr: number): void {
    const r = this.roi;
    if (!r) return;
    const x = t.offsetX + r.x * t.zoom;
    const y = t.offsetY + r.y * t.zoom;
    const w = r.w * t.zoom;
    const hh = r.h * t.zoom;
    g.save();
    g.lineWidth = 2 * dpr;
    g.strokeStyle = 'rgba(0,0,0,0.6)';
    g.strokeRect(x - dpr, y - dpr, w + 2 * dpr, hh + 2 * dpr);
    g.strokeStyle = active ? '#ff4d4f' : '#ff8a8c';
    g.setLineDash(active ? [] : [6 * dpr, 4 * dpr]);
    g.strokeRect(x, y, w, hh);
    g.setLineDash([]);
    if (active) {
      g.fillStyle = '#ff4d4f';
      const s = 6 * dpr;
      for (const [cx, cy] of [[x, y], [x + w, y], [x, y + hh], [x + w, y + hh]]) g.fillRect(cx - s / 2, cy - s / 2, s, s);
    }
    const label = `${r.w} × ${r.h} px`;
    g.font = `${12 * dpr}px system-ui, sans-serif`;
    const tw = g.measureText(label).width + 10 * dpr;
    const ly = y - 20 * dpr < 0 ? y + hh + 4 * dpr : y - 20 * dpr;
    g.fillStyle = 'rgba(0,0,0,0.65)';
    g.fillRect(x, ly, tw, 16 * dpr);
    g.fillStyle = '#fff';
    g.fillText(label, x + 5 * dpr, ly + 12 * dpr);
    g.restore();
  }
}
