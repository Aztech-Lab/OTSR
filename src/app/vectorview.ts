// The vector view outside of a fit: the current model drawn in Gaussian-id colours (random colour x luma per
// Gaussian) through the display-only recolour kernel. Owns a recoloured copy of the parameters and the viewer's
// pre-render hook while it is on; released when the model changes or a fit starts.

import { bindGroup, grid1d, runPass, type GpuContext, type Pipelines } from '../engine';
import type { AppModel, Store } from './store';
import type { Viewer } from './viewer';

const FX_BYTES = 32;

export class VectorView {
  private model: AppModel | null = null;
  private display: GPUBuffer | null = null;
  private uni: GPUBuffer | null = null;
  private bg: GPUBindGroup | null = null;
  private seed = 0;
  on = false;

  constructor(private readonly viewer: Viewer, private readonly ctx: GpuContext, private readonly pipes: Pipelines, private readonly store: Store) {
    store.subscribe((s, changed) => { if (changed.has('model') && s.model !== this.model) this.release(); });
  }

  /** Switches the view on (needs a model, not during a job) or off; returns the state in effect. */
  set(on: boolean): boolean {
    const m = this.store.state.model;
    if (!on || !m || this.store.busy) {
      this.release();
      return false;
    }
    if (this.model !== m) {
      this.release();
      const d = this.ctx.device;
      this.model = m;
      this.seed = (Math.random() * 1e9) >>> 0;
      this.display = d.createBuffer({ label: 'vector.display', size: m.N * 36, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      this.uni = d.createBuffer({ label: 'vector.fx', size: FX_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.bg = bindGroup(d, this.pipes.recolour, { 0: this.uni, 1: m.params, 2: this.display }, 'recolour');
      const ab = new ArrayBuffer(FX_BYTES);
      new Uint32Array(ab).set([m.N, this.seed]);
      new Float32Array(ab).set([1, 0, 0], 2); // amount 1, size_log 0, scan_u 0
      d.queue.writeBuffer(this.uni, 0, ab);
    }
    this.viewer.beforeRender = (enc) => { if (this.bg && this.model) runPass(enc, this.pipes.recolour, this.bg, grid1d(this.model.N), 'recolour'); };
    this.viewer.overrideParams(this.display);
    this.on = true;
    return true;
  }

  release(): void {
    if (this.on) {
      this.viewer.beforeRender = null;
      if (this.viewer.hasModel) this.viewer.overrideParams(null);
    }
    this.display?.destroy();
    this.uni?.destroy();
    this.display = null;
    this.uni = null;
    this.bg = null;
    this.model = null;
    this.on = false;
  }
}
