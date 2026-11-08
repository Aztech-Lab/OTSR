// Notifications: errors open the alert dialog (#alert in index.html); everything else shows as a chip in the middle
// of the status strip (#notice) and fades after a few seconds.

import { h } from './dom';

export type ToastKind = 'info' | 'success' | 'error';

let timer = 0;

function hide(): void {
  const host = document.getElementById('notice');
  if (!host) return;
  host.hidden = true;
  host.parentElement?.classList.remove('has-notice');
}

export function toast(message: string, kind: ToastKind = 'info', ms = 5000): void {
  if (kind === 'error') return showAlert(message);
  const host = document.getElementById('notice');
  if (!host) { console.info(message); return; }
  host.hidden = true; // restart the entrance animation
  host.className = `notice ${kind}`;
  host.replaceChildren(h('span', { class: 'msg', title: message }, message), h('button', { class: 'close', title: 'Dismiss · 關閉', onClick: hide }, '×'));
  void host.offsetWidth;
  host.hidden = false;
  host.parentElement?.classList.add('has-notice');
  clearTimeout(timer);
  if (ms > 0) timer = window.setTimeout(hide, ms);
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
