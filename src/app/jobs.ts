// One GPU job at a time (fit, export, …) with progress, cancellation and error reporting in one place.

import type { Store } from './store';
import { toast } from './toast';

export interface JobContext {
  signal: AbortSignal;
  progress: (fraction: number, text: string) => void;
}

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
    const progress = (fraction: number, text: string) => this.store.set({ job: { name, fraction, text, cancel: () => controller.abort() } });
    progress(-1, 'Starting …');
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
