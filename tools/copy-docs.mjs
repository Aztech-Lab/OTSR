// Copies the documentation (and its figures, which the technical report shows) into dist/docs so the built app's
// "Documentation" link and report.html work offline.
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const dst = path.join(ROOT, 'dist', 'docs');
fs.mkdirSync(path.join(dst, 'figures'), { recursive: true });
for (const f of fs.readdirSync(path.join(ROOT, 'docs'))) {
  if (f.endsWith('.md')) fs.copyFileSync(path.join(ROOT, 'docs', f), path.join(dst, f));
}
for (const f of fs.readdirSync(path.join(ROOT, 'docs', 'figures'))) {
  if (/\.(jpe?g|png|webp|svg)$/i.test(f)) fs.copyFileSync(path.join(ROOT, 'docs', 'figures', f), path.join(dst, 'figures', f));
}
fs.copyFileSync(path.join(ROOT, 'README.md'), path.join(dst, 'README.md'));
console.log(`copied docs to ${dst}`);
