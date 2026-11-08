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

/** Errors open the alert dialog (#alert in index.html); everything else is a toast that fades after `ms`. */
export function toast(message: string, kind: ToastKind = 'info', ms = 5000): void {
  if (kind === 'error') return showAlert(message);
  const el = h('div', { class: `toast ${kind}` }, h('span', { class: 'msg' }, message));
  const close = h('button', { class: 'close', title: 'Dismiss', onClick: () => el.remove() }, icon('x', 14));
  el.appendChild(close);
  container().appendChild(el);
  if (ms > 0) setTimeout(() => el.remove(), ms);
}

/** The error popup: one modal dialog, new messages are appended while it is open; OK / Esc close it. */
export function showAlert(message: string): void {
  const dlg = document.getElementById('alert') as HTMLDialogElement | null;
  const body = document.getElementById('alert-msg');
  if (!dlg || !body) { console.error(message); return; }
  if (dlg.open) body.append(h('div', null, message));
  else {
    body.replaceChildren(h('div', null, message));
    dlg.showModal();
  }
}
