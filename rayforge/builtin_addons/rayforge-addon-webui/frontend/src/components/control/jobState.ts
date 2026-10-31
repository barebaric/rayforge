/**
 * Pure state machine that maps the backend job status onto the job bar
 * controls in the status bar (start / pause-resume / cancel) plus the
 * frame button availability in the control panel.
 */

export interface JobCommandInput {
  active: boolean;
  phase: string;
  connected: boolean;
  hasContent: boolean;
}

export interface JobCommandState {
  kind: "idle" | "running" | "paused";
  phaseLabel: string;
  startEnabled: boolean;
  holdEnabled: boolean;
  holdLabel: string;
  holdIcon: "pause" | "play";
  cancelEnabled: boolean;
  frameEnabled: boolean;
}

/** Derives the job bar control states from the current status. */
export function deriveJobCommands(input: JobCommandInput): JobCommandState {
  const { active, phase, connected, hasContent } = input;
  if (!active) {
    return {
      kind: "idle",
      phaseLabel: "Idle",
      startEnabled: connected && hasContent,
      holdEnabled: false,
      holdLabel: "Pause",
      holdIcon: "pause",
      cancelEnabled: false,
      frameEnabled: connected,
    };
  }
  const paused = phase === "paused";
  return {
    kind: paused ? "paused" : "running",
    phaseLabel: paused ? "Paused" : "Running",
    startEnabled: false,
    holdEnabled: true,
    holdLabel: paused ? "Resume" : "Pause",
    holdIcon: paused ? "play" : "pause",
    cancelEnabled: true,
    frameEnabled: false,
  };
}

/** Formats an ETA as mm:ss (h:mm:ss past an hour), or --:-- when unknown. */
export function fmtEta(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) {
    return "--:--";
  }
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
