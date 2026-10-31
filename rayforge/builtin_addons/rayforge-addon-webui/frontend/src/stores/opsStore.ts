import { createStore } from "zustand/vanilla";
import type { ApiBackend } from "../api/client";
import { ApiError } from "../api/client";
import type { OpsSegment, OpsResponse } from "../api/types";
import { api } from "./api";
import { jobStore } from "./jobStore";
import {
  advancePosition,
  buildTrack,
  clampPosition,
  nextSegmentBoundary,
  playbackDeltaMm,
  prevSegmentBoundary,
  type Track,
} from "./playback";
import { startRaf } from "./raf";

/** Tolerance (mm) used when fetching the toolpath preview. */
export const OPS_TOLERANCE = 0.05;
export const PLAYBACK_SPEEDS = [1, 10, 60] as const;

export interface OpsState {
  /** True when the backend returned an artifact for /ops. */
  available: boolean;
  fetching: boolean;
  generationId: string | null;
  segments: OpsSegment[];
  estimatedSeconds: number | null;
  distanceMm: number | null;
  track: Track | null;
  playing: boolean;
  /** True while the user drags (or keyboard-focuses) the scrubber. */
  scrubbing: boolean;
  positionMm: number;
  speed: number;
}

export const opsStore = createStore<OpsState>()(() => ({
  available: false,
  fetching: false,
  generationId: null,
  segments: [],
  estimatedSeconds: null,
  distanceMm: null,
  track: null,
  playing: false,
  scrubbing: false,
  positionMm: 0,
  speed: 1,
}));

/**
 * Whether a job lifecycle transition warrants an /ops refetch: the
 * artifact is (re)generated when a job starts, and a fresh preview is
 * wanted once it finishes. WS job events carry no generation id.
 */
export function shouldRefetchOps(
  previousActive: boolean,
  nextActive: boolean,
): boolean {
  return previousActive !== nextActive;
}

/** Applies a fetched response, skipping work for a cached generation. */
export function applyOpsResponse(response: OpsResponse): void {
  const current = opsStore.getState();
  if (current.available && current.generationId === response.generation_id) {
    return;
  }
  opsStore.setState({
    available: true,
    generationId: response.generation_id,
    segments: response.segments,
    estimatedSeconds: response.estimated_seconds,
    distanceMm: response.distance_mm,
    track: buildTrack(response.segments),
    playing: false,
    scrubbing: false,
    positionMm: 0,
  });
}

export function clearOps(): void {
  opsStore.setState({
    available: false,
    generationId: null,
    segments: [],
    estimatedSeconds: null,
    distanceMm: null,
    track: null,
    playing: false,
    scrubbing: false,
    positionMm: 0,
  });
}

/**
 * Fetches /ops. A 404 means "no artifact" and clears the store; other
 * errors keep the previous preview so it survives transient failures.
 */
export async function refreshOps(
  backend: Pick<ApiBackend, "getOps"> = api,
): Promise<void> {
  opsStore.setState({ fetching: true });
  try {
    applyOpsResponse(await backend.getOps(OPS_TOLERANCE));
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) clearOps();
  } finally {
    opsStore.setState({ fetching: false });
  }
}

function canPlay(): boolean {
  return (
    opsStore.getState().track !== null && !jobStore.getState().active
  );
}

export function play(): void {
  if (!canPlay()) return;
  const state = opsStore.getState();
  const atEnd =
    state.track !== null && state.positionMm >= state.track.totalDistance;
  opsStore.setState({
    playing: true,
    positionMm: atEnd ? 0 : state.positionMm,
  });
}

export function pause(): void {
  opsStore.setState({ playing: false });
}

export function togglePlay(): void {
  if (opsStore.getState().playing) pause();
  else play();
}

export function setPlaybackPosition(positionMm: number): void {
  const track = opsStore.getState().track;
  if (!track) return;
  opsStore.setState({ positionMm: clampPosition(track, positionMm) });
}

export function setPlaybackSpeed(speed: number): void {
  if (!Number.isFinite(speed) || speed <= 0) return;
  opsStore.setState({ speed });
}

export function setScrubbing(scrubbing: boolean): void {
  if (opsStore.getState().scrubbing === scrubbing) return;
  opsStore.setState({ scrubbing });
}

export function stepForward(): void {
  const state = opsStore.getState();
  if (!canPlay()) return;
  opsStore.setState({
    positionMm: nextSegmentBoundary(state.track!, state.positionMm),
  });
}

export function stepBack(): void {
  const state = opsStore.getState();
  if (!canPlay()) return;
  opsStore.setState({
    positionMm: prevSegmentBoundary(state.track!, state.positionMm),
  });
}

/**
 * Advances local playback once per frame while playing. Live jobs own
 * the transport, so the ticker only runs when no job is active.
 */
export function startPlaybackTicker(): () => void {
  return startRaf((dt) => {
    const state = opsStore.getState();
    if (!state.playing || !state.track || jobStore.getState().active) return;
    const delta = playbackDeltaMm(
      state.track,
      state.speed,
      dt,
      state.estimatedSeconds,
    );
    const next = advancePosition(state.track, state.positionMm, delta);
    if (next >= state.track.totalDistance) {
      opsStore.setState({ positionMm: next, playing: false });
    } else {
      opsStore.setState({ positionMm: next });
    }
  });
}
