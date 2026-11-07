// Locate a Chromium-based browser with WebGPU for the browser-level tools (probe, UI smoke test).
import fs from 'node:fs';

const CANDIDATES = [
  process.env.OTSR_BROWSER,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
];

export function findBrowser() {
  const arg = process.argv.find(a => a.startsWith('--browser='));
  const list = arg ? [arg.slice('--browser='.length)] : CANDIDATES;
  for (const p of list) if (p && fs.existsSync(p)) return p;
  throw new Error('No Chrome/Edge found. Set OTSR_BROWSER or pass --browser=<path>.');
}

// Headless Chrome keeps WebGPU behind the blocklist and the "unsafe" switch; both are needed on Windows.
// Note: headless Chrome may pick the integrated GPU on dual-GPU machines (docs/troubleshooting.md).
export function webgpuArgs() {
  return ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-first-run', '--no-default-browser-check'];
}
