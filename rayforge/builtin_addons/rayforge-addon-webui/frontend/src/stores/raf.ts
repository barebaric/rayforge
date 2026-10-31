/**
 * Minimal requestAnimationFrame driver. Calls `onFrame` with the
 * elapsed seconds since the previous frame (clamped to avoid huge
 * jumps after tab switches) until the returned stop function runs.
 */
export function startRaf(onFrame: (dtSeconds: number) => void): () => void {
  let running = true;
  let last = performance.now();
  const step = (now: number): void => {
    if (!running) return;
    const dt = Math.min(Math.max((now - last) / 1000, 0), 0.25);
    last = now;
    onFrame(dt);
    window.requestAnimationFrame(step);
  };
  window.requestAnimationFrame(step);
  return () => {
    running = false;
  };
}
