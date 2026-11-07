// Installs WebGPU in Node through Dawn (npm "webgpu") before each GPU test file.
// OTSR_DAWN_FLAGS="adapter=4090" picks another adapter (see docs/testing.md); the default is the system's
// high-performance adapter.

import { afterAll } from 'vitest';
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);
const flags = (process.env.OTSR_DAWN_FLAGS ?? '').split(/\s+/).filter(Boolean);
// Node >= 21 defines a read-only `navigator` getter; replace it for the duration of the file.
const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
Object.defineProperty(globalThis, 'navigator', { value: { gpu: create(flags) }, configurable: true, writable: true });

afterAll(() => {
  // Dropping the last reference lets the process exit (dawn.node keeps running while it is referenced).
  if (original) Object.defineProperty(globalThis, 'navigator', original);
  else delete (globalThis as any).navigator;
});
