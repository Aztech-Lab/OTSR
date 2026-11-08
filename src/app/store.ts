// The application state: one store, one subscribe. Preferences persist in localStorage (best effort).

import type { AdapterInfo } from '../engine';
import type { ExportFormat, LoadedImage } from './image';
import type { Roi } from './roi';

export type Mode = 'simple' | 'research';
export type Preset = 'fast' | 'standard' | 'best' | 'custom';
export type Theme = 'auto' | 'light' | 'dark';
export type CompareMode = 'off' | 'split';
export type ViewMode = 'otsr' | 'raw' | 'bicubic' | 'bilinear' | 'nearest';
export type Loss = 'l2' | 'l1';

/** Research overlays drawn on the viewer: 1-sigma Gaussian outlines and the |error| map of the fit. */
export interface Layers {
  gaussians: boolean;
  error: boolean;
  /** The model in Gaussian-id colours (random colour x luma per Gaussian). */
  vector: boolean;
  /** The optimal-transport lattice of the initialisation over the image. */
  ot: boolean;
}

/** The running fit, for the live readout. */
export interface LiveFit {
  step: number;
  steps: number;
  psnr: number;
  /** Steps per second. */
  rate: number;
  /** Estimated seconds left (NaN until known). */
  left: number;
}

/** A region where the Gaussians changed most between two snapshots of the fit (the monitor panel). */
export interface ActiveRegion {
  x: number;
  y: number;
  w: number;
  h: number;
  /** Relative activity 0..1 (1 = the most active region). */
  score: number;
  label: string;
}

export interface ErrorStats {
  /** Mean absolute error, 8-bit units. */
  mae: number;
  /** Largest per-pixel error, 8-bit units. */
  max: number;
  /** Display gain of the map (error x gain -> 0..1). */
  gain: number;
}

export interface Prefs {
  mode: Mode;
  preset: Preset;
  kPerPx: number;
  steps: number;
  edges: boolean;
  edgePx: number;
  loss: Loss;
  scale: number;
  exportFormat: ExportFormat;
  /** Quality of the lossy export formats, 0.5..1. */
  exportQuality: number;
  compare: CompareMode;
  compareWith: 'bicubic' | 'bilinear' | 'nearest';
  theme: Theme;
  motion: boolean;
  /** The fit animation (OT lattice, watercolour, scan reveal); off = live render while fitting. */
  fitAnim: boolean;
  /** The starlight over the stage. */
  stars: boolean;
  /** The control console on the left is open. */
  panelLeft: boolean;
  /** Measured Gaussian-steps per second of this machine, for time estimates (0 = unknown). */
  throughput: number;
  sections: Record<string, boolean>;
}

export interface AppModel {
  N: number;
  W: number;
  H: number;
  params: GPUBuffer;
  /** CPU copy, valid when no fit is running. */
  cpu: Float32Array | null;
  meta: Record<string, unknown>;
  /** True once the model was exported or saved (used for the unsaved-work hint). */
  saved: boolean;
  /** The transport map of the initialisation (ny x nx x 2, box units), for the OT overlay. */
  ot?: { T: Float32Array; grid: { ny: number; nx: number } } | null;
}

export interface JobState {
  name: string;
  /** 0..1, or -1 for indeterminate. */
  fraction: number;
  text: string;
  cancel: () => void;
}

export interface FitSummary {
  psnr: number;
  seconds: number;
  steps: number;
  count: number;
  cancelled: boolean;
}

export interface AppState {
  prefs: Prefs;
  adapter: AdapterInfo | null;
  image: LoadedImage | null;
  sourceBuf: GPUBuffer | null;
  model: AppModel | null;
  job: JobState | null;
  lastFit: FitSummary | null;
  roi: Roi | null;
  tool: 'pan' | 'roi';
  viewMode: ViewMode;
  zoom: number;
  status: { text: string; error: boolean };
  /** Last progress lines of the running / finished job (newest last). */
  log: string[];
  live: LiveFit | null;
  /** PSNR curve of the last / running fit. */
  curve: { step: number; psnr: number }[];
  layers: Layers;
  errorStats: ErrorStats | null;
  /** The monitor panel on the right is open. */
  monitor: boolean;
  regions: ActiveRegion[];
  /** Tile entries of the last fit (for the exact GPU memory figure), 0 = unknown. */
  fitEntries: number;
}

export const DEFAULT_PREFS: Prefs = {
  mode: 'simple',
  preset: 'standard',
  kPerPx: 0.1,
  steps: 3000,
  edges: true,
  edgePx: 1.4,
  loss: 'l2',
  scale: 4,
  exportFormat: 'png',
  exportQuality: 0.92,
  compare: 'split',
  compareWith: 'bicubic',
  theme: 'light',
  motion: true,
  fitAnim: true,
  stars: true,
  panelLeft: true,
  throughput: 0,
  sections: {},
};

const PREFS_KEY = 'otsr.prefs.v1';

function loadPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (!raw) return { ...DEFAULT_PREFS };
    const p = JSON.parse(raw) as Partial<Prefs>;
    return { ...DEFAULT_PREFS, ...p, sections: { ...(p.sections ?? {}) } };
  } catch {
    return { ...DEFAULT_PREFS };
  }
}

type Listener = (state: AppState, changed: Set<keyof AppState>) => void;

export class Store {
  state: AppState;
  private readonly listeners = new Set<Listener>();
  private pending: Set<keyof AppState> | null = null;

  constructor() {
    this.state = {
      prefs: loadPrefs(),
      adapter: null,
      image: null,
      sourceBuf: null,
      model: null,
      job: null,
      lastFit: null,
      roi: null,
      tool: 'pan',
      viewMode: 'bicubic',
      zoom: 1,
      status: { text: '', error: false },
      log: [],
      live: null,
      curve: [],
      layers: { gaussians: false, error: false, vector: false, ot: false },
      errorStats: null,
      monitor: false,
      regions: [],
      fitEntries: 0,
    };
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  /** Merges `patch` and notifies listeners once per micro-task. */
  set(patch: Partial<AppState>): void {
    Object.assign(this.state, patch);
    const keys = Object.keys(patch) as (keyof AppState)[];
    if (!this.pending) {
      this.pending = new Set(keys);
      queueMicrotask(() => {
        const changed = this.pending!;
        this.pending = null;
        for (const l of this.listeners) l(this.state, changed);
      });
    } else {
      for (const k of keys) this.pending.add(k);
    }
  }

  setPrefs(patch: Partial<Prefs>): void {
    const prefs = { ...this.state.prefs, ...patch };
    this.set({ prefs });
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    } catch {
      /* private mode or quota: preferences simply do not persist */
    }
  }

  setStatus(text: string, error = false): void {
    this.set({ status: { text, error } });
  }

  get busy(): boolean {
    return this.state.job !== null;
  }
}
