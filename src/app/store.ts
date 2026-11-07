// The application state: one store, one subscribe. Preferences persist in localStorage (best effort).

import type { AdapterInfo } from '../engine';
import type { LoadedImage } from './image';
import type { Roi } from './roi';

export type Mode = 'simple' | 'research';
export type Preset = 'fast' | 'standard' | 'best' | 'custom';
export type Theme = 'auto' | 'light' | 'dark';
export type CompareMode = 'off' | 'split';
export type ViewMode = 'otsr' | 'raw' | 'bicubic' | 'bilinear' | 'nearest';

export interface Prefs {
  mode: Mode;
  preset: Preset;
  kPerPx: number;
  steps: number;
  edges: boolean;
  edgePx: number;
  scale: number;
  compare: CompareMode;
  compareWith: 'bicubic' | 'bilinear' | 'nearest';
  theme: Theme;
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
}

export const DEFAULT_PREFS: Prefs = {
  mode: 'simple',
  preset: 'standard',
  kPerPx: 0.1,
  steps: 3000,
  edges: true,
  edgePx: 1.4,
  scale: 4,
  compare: 'split',
  compareWith: 'bicubic',
  theme: 'auto',
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
