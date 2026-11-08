import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Relative asset paths: the build runs from any folder or sub-path (GitHub Pages, a local static server).
  base: './',
  build: {
    target: 'es2022',
    outDir: 'dist',
    sourcemap: true,
    // two pages: the app and the technical report
    rollupOptions: { input: { main: fileURLToPath(new URL('index.html', import.meta.url)), report: fileURLToPath(new URL('report.html', import.meta.url)) } },
  },
  test: {
    projects: [
      {
        extends: true,
        test: { name: 'unit', include: ['tests/unit/**/*.test.ts'], environment: 'node' },
      },
      {
        // WGSL kernels on a real GPU through Dawn's Node bindings (npm "webgpu"); see docs/testing.md.
        extends: true,
        test: {
          name: 'gpu',
          include: ['tests/gpu/**/*.test.ts'],
          environment: 'node',
          setupFiles: ['tests/gpu/setup.ts'],
          pool: 'forks',
          fileParallelism: false,
          testTimeout: 300_000,
          hookTimeout: 120_000,
        },
      },
    ],
  },
});
