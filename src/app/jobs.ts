// One GPU job at a time (fit, export, …) with progress, cancellation and error reporting in one place.

import type { Store } from './store';
import { toast } from './toast';

export interface JobContext {
  signal: AbortSignal;
  /** Reports progress; `inplace` lines replace the previous in-place line instead of adding to the log (step counters). */
  progress: (fraction: number, text: string, inplace?: boolean) => void;
}

const LOG_KEEP = 14;

export class JobRunner {
  constructor(private readonly store: Store) {
    window.addEventListener('beforeunload', (e) => {
      if (this.store.state.job) {
        e.preventDefault();
        e.returnValue = '';
      }
    });
  }

  /**
   * Runs `fn` as the single active job. Returns its result, or undefined if it failed (the error is shown) or
   * another job is running. Cancellation surfaces as `cancelled: true` in whatever `fn` returns.
   */
  async run<T>(name: string, fn: (ctx: JobContext) => Promise<T>): Promise<T | undefined> {
    if (this.store.state.job) {
      toast(`${this.store.state.job.name} is still running.`, 'info', 3000);
      return undefined;
    }
    const controller = new AbortController();
    let lastInplace = false;
    const progress = (fraction: number, text: string, inplace = false) => {
      const log = this.store.state.log;
      let next = log;
      if (text) {
        if (inplace && lastInplace && log.length) next = [...log.slice(0, -1), text];
        else if (text !== log[log.length - 1]) next = [...log.slice(-(LOG_KEEP - 1)), text];
      }
      lastInplace = inplace && !!text;
      this.store.set({ job: { name, fraction, text, cancel: () => controller.abort() }, log: next });
    };
    this.store.set({ log: [] });
    progress(-1, `${name} started`);
    try {
      return await fn({ signal: controller.signal, progress });
    } catch (e) {
      const msg = (e as Error)?.message ?? String(e);
      if (!controller.signal.aborted) {
        this.store.setStatus(`${name} failed: ${msg}`, true);
        toast(`${name} failed: ${msg}`, 'error');
      }
      return undefined;
    } finally {
      this.store.set({ job: null });
    }
  }

  cancel(): void {
    this.store.state.job?.cancel();
  }
}
