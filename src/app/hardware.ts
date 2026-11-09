// What the start screen tells about this machine: whether WebGPU works here, the GPU the browser gives the page and
// its limits, the memory the browser reports, and what that means for OTSR — the largest image one fit can take and
// the tier to start with. Browsers do not report the size of the GPU memory (it would help identify the machine): the
// driver's memory heaps are shown where Chrome's WebGPU developer flag exposes them; otherwise the card says so and
// shows the largest buffer the GPU allows and the system memory, which phones, Apple and Intel integrated GPUs share
// with the GPU. Nothing here leaves the page.

import { fmtBytes, fmtSeconds, h } from './dom';
import { MAX_PIXELS } from './image';
import { PRESETS, fitSecondsEstimate, gaussianCount, maxFitPixels } from './presets';
import type { Prefs } from './store';

/** A phone: a touch screen whose short side is under 600 CSS px. Phones start on the Fast tier, draw the viewer at
 * most at 2x device pixels and hold the starlight still while a job runs. */
export function isPhone(): boolean {
  if (typeof matchMedia !== 'function' || typeof screen === 'undefined') return false;
  return matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 600;
}

export interface HardwareReport {
  browser: string;
  os: string;
  /** The GPU's name: the adapter's description, the WebGL renderer, or vendor + architecture. */
  gpu: string;
  vendor: string;
  architecture: string;
  fallback: boolean;
  /** The GPU shares the system memory (phones, Apple, Intel integrated); null when it cannot be told. */
  unified: boolean | null;
  /** Device-local memory from the driver's heaps (Chrome's WebGPU developer flag only), bytes. */
  vram: number | null;
  /** navigator.deviceMemory in GB (Chromium only; rounded, at most 8). */
  ram: number | null;
  maxBuffer: number;
  maxBinding: number;
  features: string[];
  threads: number;
  screen: string;
}

interface UaData {
  brands?: { brand: string; version: string }[];
  platform?: string;
  getHighEntropyValues?: (hints: string[]) => Promise<{ platformVersion?: string; model?: string }>;
}

/** Browser and operating system, from the user-agent client hints where the browser has them (Chromium), else the
 * user-agent string. */
async function browserAndOs(): Promise<{ browser: string; os: string }> {
  const ua = navigator.userAgent;
  const uad = (navigator as unknown as { userAgentData?: UaData }).userAgentData;
  let browser = '';
  const brand = uad?.brands?.find((b) => !/not.?a.?brand|chromium/i.test(b.brand)) ?? uad?.brands?.find((b) => /chromium/i.test(b.brand));
  if (brand) browser = `${brand.brand.replace(/^(Google|Microsoft) /, '')} ${brand.version}`;
  let r: RegExpExecArray | null;
  if (!browser) {
    if ((r = /Edg(?:A|iOS)?\/(\d+)/.exec(ua))) browser = `Edge ${r[1]}`;
    else if ((r = /SamsungBrowser\/(\d+)/.exec(ua))) browser = `Samsung Internet ${r[1]}`;
    else if ((r = /(?:Chrome|CriOS)\/(\d+)/.exec(ua))) browser = `Chrome ${r[1]}`;
    else if ((r = /(?:Firefox|FxiOS)\/(\d+)/.exec(ua))) browser = `Firefox ${r[1]}`;
    else if ((r = /Version\/(\d+(?:\.\d+)?).*Safari/.exec(ua))) browser = `Safari ${r[1]}`;
    else browser = 'unknown browser';
  }
  let os = '';
  if (uad?.platform) {
    os = uad.platform;
    try {
      const hi = await Promise.race([uad.getHighEntropyValues?.(['platformVersion', 'model']), new Promise<undefined>((res) => setTimeout(() => res(undefined), 400))]);
      const version = hi?.platformVersion ?? '';
      const major = Number(version.split('.')[0]);
      if (os === 'Windows' && major > 0) os = major >= 13 ? 'Windows 11' : 'Windows 10';
      else if ((os === 'Android' || os === 'macOS') && major > 0) os = `${os} ${version.split('.').slice(0, os === 'macOS' ? 2 : 1).join('.')}`;
      if (hi?.model) os += ` · ${hi.model}`;
    } catch { /* the plain platform name is enough */ }
  } else if ((r = /Android (\d+(?:\.\d+)?)/.exec(ua))) os = `Android ${r[1]}`;
  else if ((r = /(?:iPhone|iPad|iPod).*? OS (\d+)_(\d+)/.exec(ua))) os = `iOS ${r[1]}.${r[2]}`;
  else if (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1) os = 'iPadOS';
  else if (/Windows NT/.test(ua)) os = 'Windows';
  else if (/Mac OS X/.test(ua)) os = 'macOS';
  else if (/CrOS/.test(ua)) os = 'ChromeOS';
  else if (/Linux/.test(ua)) os = 'Linux';
  return { browser, os };
}

/** The GPU's model name as WebGL reports it (WebGPU adapters usually leave the description empty). */
function webglRenderer(): string {
  try {
    const c = document.createElement('canvas');
    const gl = (c.getContext('webgl2') ?? c.getContext('webgl')) as WebGLRenderingContext | null;
    if (!gl) return '';
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const raw = String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER) ?? '');
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return cleanRenderer(raw);
  } catch {
    return '';
  }
}

/** "ANGLE (NVIDIA, NVIDIA GeForce RTX 4090 (0x00002684) Direct3D11 vs_5_0 ps_5_0, D3D11)" → "NVIDIA GeForce RTX 4090". */
export function cleanRenderer(raw: string): string {
  let s = raw.trim();
  const angle = /^ANGLE \((.*)\)$/.exec(s);
  if (angle) {
    const parts = angle[1].split(', ');
    s = parts.length > 1 ? parts[1] : parts[0];
  }
  s = s.replace(/^ANGLE Metal Renderer: /, '').replace(/ \(0x[0-9a-f]+\).*$/i, '').replace(/ (Direct3D|OpenGL|Vulkan).*$/, '').replace(/\/PCIe\/SSE2$/, '').replace(/, or similar$/, '');
  return /^(webkit webgl|generic renderer)$/i.test(s.trim()) ? '' : s.trim();
}

/** Whether the GPU shares the system memory, from the vendor and architecture names the browser gives. */
function sharesMemory(vendor: string, arch: string): boolean | null {
  const v = vendor.toLowerCase(), a = arch.toLowerCase();
  if (/apple|qualcomm|arm|imagination|img-tec|samsung|mediatek|broadcom/.test(v)) return true;
  if (v === 'intel') return !/hpg|alchemist|battlemage/.test(a);
  if (v === 'nvidia') return false;
  return null;
}

export async function hardwareReport(adapter: GPUAdapter, device: GPUDevice, isFallback: boolean): Promise<HardwareReport> {
  const info = (adapter.info ?? {}) as Partial<GPUAdapterInfo> & { memoryHeaps?: { size: number | bigint; properties: number }[] };
  const vendor = info.vendor ?? '';
  const architecture = info.architecture ?? '';
  const gpu = (info.description || '').trim() || webglRenderer() || [vendor, architecture].filter(Boolean).join(' ') || 'unknown GPU';
  let vram: number | null = null;
  if (Array.isArray(info.memoryHeaps)) {
    let sum = 0;
    for (const hp of info.memoryHeaps) if (hp.properties & 1) sum += Number(hp.size);
    if (sum > 0) vram = sum;
  }
  const features = ([['shader-f16', 'f16'], ['subgroups', 'subgroups'], ['timestamp-query', 'timestamps']] as const).filter(([f]) => adapter.features.has(f)).map(([, n]) => n);
  const dpr = window.devicePixelRatio || 1;
  const { browser, os } = await browserAndOs();
  return {
    browser, os, gpu, vendor, architecture, fallback: isFallback,
    unified: sharesMemory(vendor, architecture),
    vram,
    ram: Number((navigator as unknown as { deviceMemory?: number }).deviceMemory) || null,
    maxBuffer: device.limits.maxBufferSize,
    maxBinding: device.limits.maxStorageBufferBindingSize,
    features,
    threads: navigator.hardwareConcurrency || 0,
    screen: `${Math.round(screen.width * dpr)} × ${Math.round(screen.height * dpr)} px (${+dpr.toFixed(2)}×)`,
  };
}

const mp = (px: number) => `${px >= 1e7 ? (px / 1e6).toFixed(0) : (px / 1e6).toFixed(1)} MP`;
const gb = (n: number) => `${n >= 8 ? '≥ ' : '≈ '}${n} GB`;

/** The card on the start screen: the verdict first, then the rows that back it. */
export function renderSyscheck(host: HTMLElement, r: HardwareReport, prefs: Prefs, sample: { W: number; H: number } | null): void {
  const fastMax = maxFitPixels(PRESETS.fast.kPerPx, r.maxBinding, r.maxBuffer);
  const stdMax = maxFitPixels(PRESETS.standard.kPerPx, r.maxBinding, r.maxBuffer);
  const level = r.fallback ? 'bad' : stdMax < 2e6 ? 'warn' : 'ok';
  const [en, zh, sub] = r.fallback ? ['Slow', '很慢', 'software adapter, no GPU acceleration'] : level === 'warn' ? ['Limited', '受限', 'WebGPU works, small buffers'] : ['Ready', '可以運行', 'WebGPU works on this GPU'];
  const memory = r.vram ? `${fmtBytes(r.vram)} (driver heaps)` : r.unified ? 'shared with the system' : 'not reported by browsers';
  const memorySub = r.ram ? `RAM ${gb(r.ram)}` : '';
  const row = (k: string, kz: string, v: string, sub = '') => [h('dt', null, k, h('small', null, kz)), h('dd', null, h('span', null, v), sub ? h('small', null, sub) : null)];
  const notes: string[] = [];
  if (r.fallback) notes.push('WebGPU runs on the CPU here, so a fit takes minutes. Turn on hardware acceleration in the browser settings or try another browser.');
  else if (stdMax >= MAX_PIXELS) notes.push(`Every tier takes images up to the app's limit of ${mp(MAX_PIXELS)} (largest GPU buffer here: ${fmtBytes(Math.min(r.maxBinding, r.maxBuffer))}).`);
  else notes.push(`One fit takes images up to ${mp(fastMax)} at Fast and ${mp(stdMax)} at Standard (largest GPU buffer here: ${fmtBytes(Math.min(r.maxBinding, r.maxBuffer))}).`);
  const tier = prefs.preset === 'custom' ? null : prefs.preset;
  if (!r.fallback && sample && tier && prefs.throughput > 0) {
    const p = PRESETS[tier];
    const est = fitSecondsEstimate(prefs, gaussianCount(p.kPerPx, sample.W, sample.H), p.steps);
    if (est !== null) notes.push(`Measured here: ${(prefs.throughput / 1e6).toFixed(0)} M Gaussian-steps/s; the sample takes about ${fmtSeconds(est)} at ${p.label}.`);
  } else if (!r.fallback) notes.push(isPhone() ? 'Phones start on Fast; Turbo is about 4× quicker. The speed is measured on the first fit.' : 'Standard is the paper setting; Turbo is about 4× quicker than Fast. The speed is measured on the first fit.');
  host.replaceChildren(
    h('div', { class: 'console-head' }, h('span', null, 'System check'), h('span', { class: 'hatch' }), h('span', { class: 'zh' }, '硬體檢測')),
    h('div', { class: 'sc-body' },
      h('div', { class: `sc-verdict ${level}` }, h('i', { class: `sq ${level === 'ok' ? 'ok' : 'err'}` }), h('b', null, en), h('small', null, zh), h('span', null, sub)),
      h('p', { class: 'sc-note' }, notes.join(' ')),
      h('dl', { class: 'sc-rows' },
        ...row('Browser', '瀏覽器', r.browser, r.os),
        ...row('GPU', '顯示卡', r.gpu, [r.vendor, r.architecture].filter(Boolean).join(' · ') + (r.fallback ? ' · software' : '')),
        ...row('GPU memory', '顯存', memory, memorySub),
        ...row('Buffers', '緩衝區', `${fmtBytes(r.maxBuffer)} max`, `${fmtBytes(r.maxBinding)} per binding`),
        ...row('Features', '特性', r.features.length ? r.features.join(' · ') : 'core WebGPU'),
        ...row('Device', '裝置', `${r.threads || '?'} CPU threads`, r.screen),
      ),
    ),
  );
  host.dataset.level = level;
  host.hidden = false;
}

/** Why WebGPU did not start, for the full-screen notice: the browser, the page's origin and what the API answered. */
export async function noGpuDiagnosis(error: string): Promise<[string, string][]> {
  const { browser, os } = await browserAndOs();
  return [
    ['Browser', `${browser}${os ? ' · ' + os : ''}`],
    ['Secure page', window.isSecureContext ? 'yes (https or localhost)' : 'no — WebGPU needs https or localhost'],
    ['WebGPU API', 'gpu' in navigator ? 'present' : 'missing in this browser (or switched off)'],
    ['Adapter', error],
    ['GPU (WebGL)', webglRenderer() || 'unknown'],
  ];
}
