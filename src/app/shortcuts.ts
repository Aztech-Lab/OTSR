// Keyboard shortcuts (ignored while typing in a field).

import type { Actions } from './actions';
import type { Store } from './store';
import type { Viewer } from './viewer';

export const SHORTCUTS: [string, string][] = [
  ['O', 'Open image'],
  ['F / Enter', 'Fit'],
  ['E', 'Export PNG'],
  ['S', 'Save model'],
  ['Esc', 'Cancel the running job / leave the region tool'],
  ['C', 'Toggle the split comparison'],
  ['B / N / R (hold)', 'Peek at bicubic / nearest / raw Gaussians'],
  ['+ / −', 'Zoom in / out'],
  ['0', 'Fit the image to the view'],
  ['1 / 2 / 4 / 8', '100 % / 200 % / 400 % / 800 %'],
  ['M', 'Region tool'],
  ['V / T', 'View: Gaussian-id colours (vector view) / OT lattice overlay'],
  ['G / D', 'View: Gaussian outlines / error map'],
  ['Arrow keys', 'Nudge the region (Shift: 10 px)'],
  ['Space (hold)', 'Pan while in the region tool'],
  ['?', 'This list'],
];

export function installShortcuts(store: Store, actions: Actions, viewer: Viewer, openFile: () => void, toggleHelp: () => void): void {
  const typing = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement | null;
    return !!t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
  };
  const holdKeys: Record<string, 'bicubic' | 'nearest' | 'raw'> = { b: 'bicubic', n: 'nearest', r: 'raw' };
  let spaceTool: 'roi' | null = null;
  window.addEventListener('keydown', (e) => {
    if (typing(e) || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    const s = store.state;
    if (k in holdKeys && !e.repeat) {
      const m = holdKeys[k];
      if ((m === 'raw' && s.model) || (m !== 'raw' && s.sourceBuf)) viewer.hold(m);
      return;
    }
    if (k === ' ' && !e.repeat && viewer.currentTool === 'roi') { spaceTool = 'roi'; viewer.setTool('pan'); e.preventDefault(); return; }
    if (k === 'ArrowLeft' || k === 'ArrowRight' || k === 'ArrowUp' || k === 'ArrowDown') {
      const t = viewer.transform();
      if (viewer.roi.roi && t) {
        const d = e.shiftKey ? 10 : 1;
        viewer.roi.nudge(k === 'ArrowLeft' ? -d : k === 'ArrowRight' ? d : 0, k === 'ArrowUp' ? -d : k === 'ArrowDown' ? d : 0, t);
        e.preventDefault();
      }
      return;
    }
    switch (k) {
      case 'o': case 'O': openFile(); break;
      case 'f': case 'F': case 'Enter': void actions.fit(); break;
      case 'e': case 'E': void actions.exportPng(); break;
      case 's': case 'S': actions.saveModel(); break;
      case 'Escape':
        if (s.job) actions.jobs.cancel();
        else if (s.tool === 'roi') actions.setTool('pan');
        else toggleHelp();
        break;
      case 'c': case 'C': {
        const on = s.prefs.compare !== 'split';
        store.setPrefs({ compare: on ? 'split' : 'off' });
        viewer.compare = on;
        viewer.requestRender();
        break;
      }
      case '+': case '=': viewer.zoomBy(1.25); break;
      case '-': case '_': viewer.zoomBy(0.8); break;
      case '0': viewer.fitToView(); break;
      case '1': viewer.zoomAbout(1); break;
      case '2': viewer.zoomAbout(2); break;
      case '4': viewer.zoomAbout(4); break;
      case '8': viewer.zoomAbout(8); break;
      case 'm': case 'M': actions.setTool(s.tool === 'roi' ? 'pan' : 'roi'); break;
      case 'g': case 'G': actions.setLayer('gaussians', !s.layers.gaussians); break;
      case 'd': case 'D': actions.setLayer('error', !s.layers.error); break;
      case 'v': case 'V': actions.setLayer('vector', !s.layers.vector); break;
      case 't': case 'T': actions.setLayer('ot', !s.layers.ot); break;
      case '?': toggleHelp(); break;
      default: return;
    }
    e.preventDefault();
  });
  window.addEventListener('keyup', (e) => {
    if (e.key in holdKeys) viewer.hold(null);
    if (e.key === ' ' && spaceTool) { viewer.setTool(spaceTool); spaceTool = null; }
  });
  window.addEventListener('blur', () => viewer.hold(null));
}
