// Transient notifications (top right). Errors stay until dismissed; everything else fades after a few seconds.

import { h, icon } from './dom';

let host: HTMLElement | null = null;

function container(): HTMLElement {
  if (!host) {
    host = h('div', { id: 'toasts', role: 'status', 'aria-live': 'polite' });
    document.body.appendChild(host);
  }
  return host;
}

export type ToastKind = 'info' | 'success' | 'error';

export function toast(message: string, kind: ToastKind = 'info', ms = kind === 'error' ? 0 : 5000): void {
  const el = h('div', { class: `toast ${kind}` }, h('span', { class: 'msg' }, message));
  const close = h('button', { class: 'close', title: 'Dismiss', onClick: () => el.remove() }, icon('x', 14));
  el.appendChild(close);
  container().appendChild(el);
  if (ms > 0) setTimeout(() => el.remove(), ms);
}
