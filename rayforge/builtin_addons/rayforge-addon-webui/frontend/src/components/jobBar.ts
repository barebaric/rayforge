import { api } from "../stores/api";
import { docStore } from "../stores/docStore";
import { jobStore } from "../stores/jobStore";
import { machineStore } from "../stores/machineStore";
import { opsStore } from "../stores/opsStore";
import { pushToast } from "../stores/uiStore";
import { button } from "./ui/Button";
import { el, fmtDuration } from "./ui/dom";
import { icon } from "./ui/icons";
import { deriveJobCommands, fmtEta } from "./control/jobState";

const CONFIRM_WINDOW_MS = 3000;

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Job bar overlay at the top of the canvas area: start / hold /
 * cancel, live progress and ETA. Cancel uses click-twice-to-confirm.
 */
export function buildJobBar(): HTMLElement {
  const root = el("div", "job-bar");

  const startBtn = button({
    label: "Start",
    iconName: "play",
    title: "Assemble and run the current document",
    onClick: () => {
      api
        .startJob()
        .catch((err: unknown) =>
          pushToast("error", `Start failed: ${describe(err)}`),
        );
    },
  });
  const holdBtn = button({
    label: "Pause",
    iconName: "pause",
    title: "Pause the running job",
    onClick: () => {
      const phase = jobStore.getState().phase;
      api
        .holdJob(phase !== "paused")
        .catch((err: unknown) =>
          pushToast("error", `Hold failed: ${describe(err)}`),
        );
    },
  });
  const cancelBtn = button({
    label: "Cancel",
    iconName: "stop",
    variant: "danger",
    title: "Cancel the running job (click again to confirm)",
    onClick: () => {
      if (!cancelBtn.classList.contains("armed")) {
        cancelBtn.classList.add("armed");
        window.setTimeout(
          () => cancelBtn.classList.remove("armed"),
          CONFIRM_WINDOW_MS,
        );
        return;
      }
      cancelBtn.classList.remove("armed");
      api
        .cancelJob()
        .catch((err: unknown) =>
          pushToast("error", `Cancel failed: ${describe(err)}`),
        );
    },
  });

  const progressTrack = el("div", "job-progress");
  const progressFill = el("div", "job-progress-fill");
  progressTrack.appendChild(progressFill);
  const phaseText = el("span", "job-phase");
  phaseText.textContent = "Idle";
  const etaText = el("span", "job-eta mono");
  etaText.textContent = "ETA --:--";

  root.append(
    startBtn,
    holdBtn,
    cancelBtn,
    progressTrack,
    phaseText,
    etaText,
  );

  const sync = (): void => {
    const { machineState } = machineStore.getState();
    const job = jobStore.getState();
    const hasContent = docStore
      .getState()
      .layers.some((layer) => layer.workpieces.length > 0);
    const state = deriveJobCommands({
      active: job.active,
      phase: job.phase,
      connected: machineState?.connected ?? false,
      hasContent,
    });

    startBtn.disabled = !state.startEnabled;
    holdBtn.disabled = !state.holdEnabled;
    holdBtn.replaceChildren(
      icon(state.holdIcon, 14),
      (() => {
        const label = el("span", "btn-label");
        label.textContent = state.holdLabel;
        return label;
      })(),
    );
    cancelBtn.disabled = !state.cancelEnabled;
    if (!state.cancelEnabled) cancelBtn.classList.remove("armed");

    progressFill.style.width = `${Math.round(
      Math.min(1, Math.max(0, job.progressFraction)) * 100,
    )}%`;
    phaseText.textContent = state.phaseLabel;
    etaText.textContent = job.active
      ? `ETA ${fmtEta(job.etaSeconds)}`
      : opsEstimateText();
  };

  machineStore.subscribe(sync);
  jobStore.subscribe(sync);
  docStore.subscribe(sync);
  opsStore.subscribe(sync);
  sync();

  return root;
}

/** Idle-time readout: the artifact estimate from /ops when available. */
function opsEstimateText(): string {
  const ops = opsStore.getState();
  return ops.estimatedSeconds !== null
    ? `est ${fmtDuration(ops.estimatedSeconds)}`
    : "ETA --:--";
}
