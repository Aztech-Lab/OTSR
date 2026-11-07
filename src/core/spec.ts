// The OTSR model specification: every constant shared with the Python reference (reference/otsr_ref/spec.py).
// tests/unit/spec.test.ts checks these values against tests/golden/spec.json, which the reference writes.

// --- image function ---------------------------------------------------------------------------------------------
export const K_SIGMA = 4; // support radius in Mahalanobis units
export const Q_CUT = 0.5 * K_SIGMA * K_SIGMA; // q = 0.5 d^T Sigma^-1 d is cut at 8
export const ALPHA_SHIFT = Math.exp(-Q_CUT); // alpha(q) = max(exp(-q) - exp(-Q_CUT), 0): continuous at the cut
export const EPS = 1e-4; // out = sum w a c / (sum w a + EPS)
export const TILE = 16; // tile edge in output pixels (engine only)

// --- parameter layout: 9 floats per Gaussian, in this order -----------------------------------------------------
export const PARAM_LAYOUT = [
  ['uv', 2],
  ['log_s', 2],
  ['rot', 1],
  ['color', 3],
  ['log_w', 1],
] as const;
export type ParamName = (typeof PARAM_LAYOUT)[number][0];
export const FLOATS_PER_GAUSSIAN = 9;
/** Offset of each parameter inside the 9 floats of one Gaussian. */
export const OFFSET = { uv: 0, log_s: 2, rot: 4, color: 5, log_w: 8 } as const;

// --- fitting ----------------------------------------------------------------------------------------------------
export const STEPS = 3000;
export const LR = { uv: 0.25, log_s: 0.02, rot: 0.02, color: 0.01, log_w: 0.02 } as const; // uv: in long-side pixels
export const FINAL_LR_FRAC = 0.05;
export const BETAS = [0.9, 0.99] as const;
export const ADAM_EPS = 1e-8;
export const UV_CLAMP = [-0.02, 1.02] as const;
export const LOG_S_MIN_PX = 0.2; // log_s in [log(0.2 / L), log(0.5)]
export const LOG_S_MAX = Math.log(0.5);
export const LOG_W_CLAMP = [-8, 8] as const;

// --- initialisation ---------------------------------------------------------------------------------------------
export const K_PER_PX = 0.1;
export const SIGMA_REL = 0.6;
export const MIN_SCALE_PX = 0.35;
export const SINKHORN_SHRINK = 0.6;
export const SINKHORN_ITERS_PER_STAGE = 6;
export const SINKHORN_FINAL_ITERS = 30;
export const LUMA = [0.299, 0.587, 0.114] as const;

// --- v3 distance-field edges ------------------------------------------------------------------------------------
export const EDGE_PX = 1.4;
export const EDGE_WINDOW = 0.75;
export const EDGE_C0 = 0.08;
export const EDGE_C1 = 0.2;

/** Cosine factor applied to every base learning rate at iteration `it` (set before the step). */
export function lrScale(it: number, steps: number): number {
  const frac = it / Math.max(1, steps);
  return FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * 0.5 * (1 + Math.cos(Math.PI * frac));
}

/** Python's round(): halves go to the even neighbour. */
export function pyRound(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** Transport-grid cells on the long side. */
export function nLongFor(H: number, W: number): number {
  return Math.min(256, Math.max(128, pyRound(Math.max(H, W) / 4)));
}

/** Window radius of the edge pass at magnification S. */
export function edgeWindowRadius(S: number, window = EDGE_WINDOW): number {
  return Math.max(1, pyRound(S * window));
}
