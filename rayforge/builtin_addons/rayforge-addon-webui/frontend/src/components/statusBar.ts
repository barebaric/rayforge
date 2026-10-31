import { machineStore } from "../stores/machineStore";
import { jobStore } from "../stores/jobStore";
import { setLeftTab } from "../stores/uiStore";
import {
  PLAYBACK_SPEEDS,
  opsStore,
  pause,
  play,
  setPlaybackPosition,
  setPlaybackSpeed,
  setScrubbing,
  stepBack,
  stepForward,
} from "../stores/opsStore";
import { button } from "./ui/Button";
import { el, fmtDuration, fmtMm } from "./ui/dom";

export interface StatusBarRefs {
  root: HTMLElement;
  statusChip: HTMLElement;
  statusText: HTMLElement;
  beamPos: HTMLElement;
  wcsText: HTMLElement;
}

export interface TransportRefs {
  root: HTMLElement;
  scrubber: HTMLInputElement;
}

/**
 * Bottom-left floating info strip: machine state / beam / WCS chip
 * (click opens the machine tab).
 */
export function buildStatusBar(): StatusBarRefs {
  const root = el("footer", "statusbar");

  const statusChip = el("button", "status-chip") as HTMLButtonElement;
  statusChip.type = "button";
  statusChip.title = "Open machine tab";
  const statusDot = el("span", "status-dot off");
  const statusText = el("span", "status-text");
  statusText.textContent = "Connecting…";
  const beamPos = el("span", "status-item mono");
  const wcsText = el("span", "status-item");
  statusChip.append(statusDot, statusText, beamPos, wcsText);
  statusChip.addEventListener("click", () => setLeftTab("machine"));

  root.append(statusChip);

  const sync = () => {
    const { machineState, viewport } = machineStore.getState();
    const connected = machineState?.connected ?? false;
    statusDot.className = `status-dot ${connected ? "on" : "off"}`;
    statusText.textContent = machineState
      ? `${connected ? "Connected" : "Offline"} · ${machineState.status}`
      : "Connecting…";
    const beam =
      machineState?.beam_position_panel ?? viewport?.beam_position_panel ?? null;
    beamPos.textContent = beam
      ? `beam ${fmtMm(beam.x)}, ${fmtMm(beam.y)} mm`
      : "";
    const wcs = viewport?.active_wcs;
    wcsText.textContent = viewport
      ? `${wcs?.name ?? "G54"} · ${wcs?.label ?? ""} · rotary ${viewport.rotary_active ? "on" : "off"}`
      : "";
  };
  machineStore.subscribe(sync);
  sync();

  return { root, statusChip, statusText, beamPos, wcsText };
}

/**
 * Bottom-center floating pill: playback transport with round
 * buttons, scrubber, time estimate and speed selector.
 */
export function buildTransport(): TransportRefs {
  const root = el("div", "transport-pill");

  const stepBackBtn = button({
    iconName: "stepBack",
    variant: "icon",
    title: "Step to previous segment",
    disabled: true,
    onClick: () => stepBack(),
  });
  const playBtn = button({
    iconName: "play",
    variant: "icon",
    title: "Play toolpath preview",
    disabled: true,
    onClick: () => play(),
  });
  const pauseBtn = button({
    iconName: "pause",
    variant: "icon",
    title: "Pause preview",
    disabled: true,
    onClick: () => pause(),
  });
  pauseBtn.hidden = true;
  const stepForwardBtn = button({
    iconName: "stepForward",
    variant: "icon",
    title: "Step to next segment",
    disabled: true,
    onClick: () => stepForward(),
  });

  const scrubber = el("input", "transport-scrubber") as HTMLInputElement;
  scrubber.type = "range";
  scrubber.min = "0";
  scrubber.max = "0";
  scrubber.step = "0.05";
  scrubber.value = "0";
  scrubber.disabled = true;
  scrubber.addEventListener("input", () => {
    setPlaybackPosition(Number.parseFloat(scrubber.value));
  });
  scrubber.addEventListener("pointerdown", () => setScrubbing(true));
  scrubber.addEventListener("pointerup", () => setScrubbing(false));
  scrubber.addEventListener("focus", () => setScrubbing(true));
  scrubber.addEventListener("blur", () => setScrubbing(false));

  const timeText = el("span", "transport-time mono");
  timeText.textContent = "– / –";

  const speedSelect = el("select", "transport-speed") as HTMLSelectElement;
  for (const speed of PLAYBACK_SPEEDS) {
    const option = el("option") as HTMLOptionElement;
    option.value = String(speed);
    option.textContent = `${speed}x`;
    speedSelect.appendChild(option);
  }
  speedSelect.disabled = true;
  speedSelect.addEventListener("change", () => {
    setPlaybackSpeed(Number.parseFloat(speedSelect.value));
  });

  root.append(
    stepBackBtn,
    playBtn,
    pauseBtn,
    stepForwardBtn,
    scrubber,
    timeText,
    speedSelect,
  );

  const syncTransport = () => {
    const ops = opsStore.getState();
    const job = jobStore.getState();
    const track = ops.track;
    const live = job.active;
    const usable = track !== null && !live;

    stepBackBtn.disabled = !usable;
    stepForwardBtn.disabled = !usable;
    playBtn.disabled = !usable;
    pauseBtn.disabled = !usable;
    scrubber.disabled = !usable;
    speedSelect.disabled = !usable;
    playBtn.hidden = ops.playing;
    pauseBtn.hidden = !ops.playing;

    if (track) {
      scrubber.max = String(track.totalDistance);
      const fraction = live
        ? job.progressFraction
        : track.totalDistance > 0
          ? ops.positionMm / track.totalDistance
          : 0;
      scrubber.value = String(
        live ? fraction * track.totalDistance : ops.positionMm,
      );
      const estimate = ops.estimatedSeconds;
      timeText.textContent = `${fmtDuration(estimate !== null ? fraction * estimate : null)} / ${fmtDuration(estimate)}`;
    } else {
      scrubber.max = "0";
      scrubber.value = "0";
      timeText.textContent = "– / –";
    }
  };
  opsStore.subscribe(syncTransport);
  jobStore.subscribe(syncTransport);
  syncTransport();

  return { root, scrubber };
}
