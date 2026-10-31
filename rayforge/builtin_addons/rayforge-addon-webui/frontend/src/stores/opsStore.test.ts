import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ApiBackend } from "../api/client";
import { ApiError } from "../api/client";
import type { OpsResponse } from "../api/types";
import { jobStore } from "./jobStore";
import {
  applyOpsResponse,
  clearOps,
  opsStore,
  pause,
  play,
  refreshOps,
  setPlaybackPosition,
  setPlaybackSpeed,
  setScrubbing,
  shouldRefetchOps,
  stepBack,
  stepForward,
} from "./opsStore";

function opsResponse(generationId: string): OpsResponse {
  return {
    segments: [
      { points: [0, 0, 10, 0], power: 1, rate: 1000, cutting: true },
      { points: [10, 0, 10, 10], power: 0, rate: 5000, cutting: false },
      { points: [10, 10, 0, 10, 0, 0], power: 0.4, rate: 800, cutting: true },
    ],
    estimated_seconds: 12.5,
    distance_mm: 40,
    generation_id: generationId,
  };
}

function backendQueue(
  ...responses: Array<{ value: OpsResponse } | { error: ApiError }>
): Pick<ApiBackend, "getOps"> {
  let index = 0;
  return {
    getOps: vi.fn(() => {
      const item = responses[index++];
      if (!item) throw new Error("no more queued ops responses");
      if ("error" in item) return Promise.reject(item.error);
      return Promise.resolve(item.value);
    }),
  };
}

beforeEach(() => {
  clearOps();
  opsStore.setState({ fetching: false, speed: 1 });
  jobStore.setState({
    active: false,
    phase: "idle",
    progressFraction: 0,
    etaSeconds: null,
    elapsedSeconds: null,
  });
});

describe("refreshOps", () => {
  it("populates the store from the backend", async () => {
    const response = opsResponse("gen-1");
    await refreshOps(backendQueue({ value: response }));
    const state = opsStore.getState();
    expect(state.available).toBe(true);
    expect(state.generationId).toBe("gen-1");
    expect(state.segments).toHaveLength(3);
    expect(state.estimatedSeconds).toBe(12.5);
    expect(state.distanceMm).toBe(40);
    expect(state.track).not.toBeNull();
    expect(state.track!.totalDistance).toBeCloseTo(40);
    expect(state.fetching).toBe(false);
  });

  it("clears the store on a 404", async () => {
    applyOpsResponse(opsResponse("gen-1"));
    await refreshOps(
      backendQueue({ error: new ApiError(404, "no artifact (404)") }),
    );
    expect(opsStore.getState().available).toBe(false);
    expect(opsStore.getState().track).toBeNull();
  });

  it("keeps the previous preview on other errors", async () => {
    applyOpsResponse(opsResponse("gen-1"));
    await refreshOps(
      backendQueue({ error: new ApiError(500, "boom (500)") }),
    );
    expect(opsStore.getState().available).toBe(true);
    expect(opsStore.getState().generationId).toBe("gen-1");
  });
});

describe("cache by generation", () => {
  it("keeps the existing track when the generation is unchanged", () => {
    const response = opsResponse("gen-1");
    applyOpsResponse(response);
    const track = opsStore.getState().track;
    applyOpsResponse(opsResponse("gen-1"));
    expect(opsStore.getState().track).toBe(track);
  });

  it("rebuilds and resets playback for a new generation", () => {
    applyOpsResponse(opsResponse("gen-1"));
    setPlaybackPosition(15);
    play();
    applyOpsResponse(opsResponse("gen-2"));
    const state = opsStore.getState();
    expect(state.generationId).toBe("gen-2");
    expect(state.positionMm).toBe(0);
    expect(state.playing).toBe(false);
    expect(state.track!.totalDistance).toBeCloseTo(40);
  });
});

describe("shouldRefetchOps", () => {
  it("refetches on job start and finish transitions", () => {
    expect(shouldRefetchOps(false, true)).toBe(true);
    expect(shouldRefetchOps(true, false)).toBe(true);
  });

  it("skips refetch while the state is unchanged", () => {
    expect(shouldRefetchOps(false, false)).toBe(false);
    expect(shouldRefetchOps(true, true)).toBe(false);
  });
});

describe("playback controls", () => {
  beforeEach(() => {
    applyOpsResponse(opsResponse("gen-1"));
  });

  it("refuses to play without a track", () => {
    clearOps();
    play();
    expect(opsStore.getState().playing).toBe(false);
  });

  it("refuses to play while a job is running", () => {
    jobStore.setState({ active: true, phase: "running" });
    play();
    expect(opsStore.getState().playing).toBe(false);
  });

  it("toggles play and pause", () => {
    play();
    expect(opsStore.getState().playing).toBe(true);
    pause();
    expect(opsStore.getState().playing).toBe(false);
  });

  it("clamps manual scrub positions", () => {
    setPlaybackPosition(-5);
    expect(opsStore.getState().positionMm).toBe(0);
    setPlaybackPosition(999);
    expect(opsStore.getState().positionMm).toBeCloseTo(40);
    setPlaybackPosition(7.5);
    expect(opsStore.getState().positionMm).toBeCloseTo(7.5);
  });

  it("steps to segment boundaries", () => {
    setPlaybackPosition(5);
    stepForward();
    expect(opsStore.getState().positionMm).toBeCloseTo(10);
    stepForward();
    expect(opsStore.getState().positionMm).toBeCloseTo(20);
    stepBack();
    expect(opsStore.getState().positionMm).toBeCloseTo(10);
    stepBack();
    expect(opsStore.getState().positionMm).toBe(0);
  });

  it("ignores invalid speeds", () => {
    setPlaybackSpeed(10);
    expect(opsStore.getState().speed).toBe(10);
    setPlaybackSpeed(0);
    setPlaybackSpeed(-5);
    expect(opsStore.getState().speed).toBe(10);
  });

  it("tracks the scrubbing flag", () => {
    setScrubbing(true);
    expect(opsStore.getState().scrubbing).toBe(true);
    setScrubbing(false);
    expect(opsStore.getState().scrubbing).toBe(false);
  });
});
