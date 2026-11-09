// The tier menu of the run button on the phone layout: the tiers from Turbo to Best with what each costs for the open
// image (Gaussians, steps, and the time once this device has been measured); in research mode also the custom budget,
// the steps and the pixel loss. On phones it takes the place of the console's Preset section.

import { bi, fmtCount, fmtSeconds, h, setText } from './dom';
import { PRESETS, TIERS, fitSecondsEstimate, fmtSteps, gaussianCount } from './presets';
import type { AppState, Loss, Mode, Preset, Store } from './store';

interface Item { key: Preset; button: HTMLButtonElement; detail: HTMLElement; est: HTMLElement }

export class TierMenu {
  private items: Item[] = [];
  private loss: [Loss, HTMLButtonElement][] = [];
  private fields: [HTMLInputElement, 'kPerPx' | 'steps'][] = [];
  private builtMode: Mode | null = null;

  constructor(private readonly button: HTMLButtonElement, private readonly menu: HTMLElement, private readonly store: Store) {
    button.addEventListener('click', () => (menu.hidden ? this.open() : this.close()));
    document.addEventListener('pointerdown', (e) => {
      if (!menu.hidden && !menu.contains(e.target as Node) && !button.contains(e.target as Node)) this.close();
    });
    window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !menu.hidden) { this.close(); e.stopImmediatePropagation(); } }, true);
    store.subscribe((s, changed) => {
      if (s.job && !menu.hidden) this.close();
      if (changed.has('prefs') || changed.has('image') || changed.has('model')) this.refresh(s);
    });
    this.refresh(store.state);
  }

  open(): void {
    this.refresh(this.store.state);
    this.menu.hidden = false;
    this.button.setAttribute('aria-expanded', 'true');
  }

  close(): void {
    this.menu.hidden = true;
    this.button.setAttribute('aria-expanded', 'false');
  }

  private build(mode: Mode): void {
    this.builtMode = mode;
    const s = this.store;
    const keys: Preset[] = mode === 'research' ? [...TIERS, 'custom'] : [...TIERS];
    this.items = keys.map((key) => {
      const [en, zh] = key === 'custom' ? ['Custom', '自訂'] : [PRESETS[key].label, PRESETS[key].zh];
      const detail = h('span', { class: 'd' });
      const est = h('span', { class: 'e' });
      const button = h('button', { class: 'tier', role: 'menuitemradio', title: key === 'custom' ? 'Your own budget and steps (research)' : PRESETS[key].hint, onClick: () => {
        s.setPrefs({ preset: key, presetChosen: true });
        if (key !== 'custom') this.close();
      } }, bi(en, zh), detail, est);
      return { key, button, detail, est };
    });
    const children: Node[] = [h('div', { class: 'micro head' }, 'Tier · 檔位', h('span', null, 'Turbo skips the fit animation'))];
    children.push(...this.items.map((i) => i.button));
    this.fields = [];
    this.loss = [];
    if (mode === 'research') {
      const field = (label: string, zh: string, key: 'kPerPx' | 'steps', min: number, max: number, step: number) => {
        const input = h('input', { type: 'number', min, max, step, value: String(s.state.prefs[key]) });
        input.addEventListener('change', () => {
          const v = Number(input.value);
          if (Number.isFinite(v) && v >= min && v <= max) s.setPrefs({ [key]: key === 'steps' ? Math.round(v) : v, preset: 'custom', presetChosen: true });
          else input.value = String(s.state.prefs[key]);
        });
        this.fields.push([input, key]);
        return h('label', { class: 'field' }, h('span', null, label, h('i', { class: 'zh' }, zh)), input);
      };
      const lossRow = h('div', { class: 'choices small', role: 'radiogroup' });
      for (const [k, en, zh] of [['l2', 'L2', '均方誤差'], ['l1', 'L1', '絕對誤差']] as const) {
        const b = h('button', { role: 'radio', onClick: () => s.setPrefs({ loss: k }) }, h('b', null, en), h('small', null, zh));
        lossRow.append(b);
        this.loss.push([k, b]);
      }
      children.push(h('div', { class: 'custom' },
        field('Gaussians / px', '每像素高斯數', 'kPerPx', 0.005, 1, 0.01),
        field('Steps', '步數', 'steps', 10, 20000, 100),
        h('div', { class: 'field' }, h('span', null, 'Pixel loss', h('i', { class: 'zh' }, '像素損失')), lossRow)));
    }
    this.menu.replaceChildren(...children);
  }

  private refresh(st: AppState): void {
    if (st.prefs.mode !== this.builtMode) this.build(st.prefs.mode);
    const img = st.image;
    for (const it of this.items) {
      const on = st.prefs.preset === it.key;
      it.button.classList.toggle('on', on);
      it.button.setAttribute('aria-checked', String(on));
      const { kPerPx, steps } = it.key === 'custom' ? { kPerPx: st.prefs.kPerPx, steps: st.prefs.steps } : PRESETS[it.key];
      const N = img ? gaussianCount(kPerPx, img.W, img.H) : 0;
      setText(it.detail, img ? `${fmtCount(N)} G · ${fmtSteps(steps)} steps` : `1/${+(1 / kPerPx).toFixed(1)} px · ${fmtSteps(steps)} steps`);
      const est = img ? fitSecondsEstimate(st.prefs, N, steps) : null;
      setText(it.est, est !== null ? `~${fmtSeconds(est)}` : '');
    }
    for (const [input, key] of this.fields) if (document.activeElement !== input) input.value = String(st.prefs[key]);
    for (const [k, b] of this.loss) b.classList.toggle('on', (st.prefs.loss ?? 'l2') === k);
  }
}
