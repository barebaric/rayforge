import { describe, expect, it } from "vitest";
import {
  advancePosition,
  buildTrack,
  clampPosition,
  nextSegmentBoundary,
  playbackDeltaMm,
  prevSegmentBoundary,
  sampleAt,
  type TrackSourceSegment,
} from "./playback";

function squareTrack(): TrackSourceSegment[] {
  return [
    { points: [0, 0, 10, 0, 10, 10], power: 1, cutting: true },
    { points: [10, 10, 10, 10], power: 0, cutting: false },
    { points: [10, 10, 0, 10, 0, 0], power: 0.5, cutting: true },
  ];
}

describe("buildTrack", () => {
  it("computes cumulative segment boundaries", () => {
    const track = buildTrack(squareTrack());
    expect(track.totalDistance).toBeCloseTo(40);
    expect(track.segments).toHaveLength(3);
    expect(track.segments[0]!.startDistance).toBe(0);
    expect(track.segments[0]!.endDistance).toBeCloseTo(20);
    expect(track.segments[1]!.startDistance).toBeCloseTo(20);
    expect(track.segments[1]!.endDistance).toBeCloseTo(20);
    expect(track.segments[2]!.startDistance).toBeCloseTo(20);
    expect(track.segments[2]!.endDistance).toBeCloseTo(40);
  });

  it("keeps per-point cumulative distances", () => {
    const track = buildTrack(squareTrack());
    expect(track.segments[0]!.cumulative).toEqual([0, 10, 20]);
    expect(track.segments[2]!.cumulative).toEqual([0, 10, 20]);
  });

  it("handles an empty segment list", () => {
    const track = buildTrack([]);
    expect(track.segments).toHaveLength(0);
    expect(track.totalDistance).toBe(0);
  });
});

describe("sampleAt", () => {
  const track = buildTrack(squareTrack());

  it("returns the start point at position 0", () => {
    const sample = sampleAt(track, 0);
    expect(sample).not.toBeNull();
    expect(sample!.x).toBeCloseTo(0);
    expect(sample!.y).toBeCloseTo(0);
    expect(sample!.segmentIndex).toBe(0);
    expect(sample!.cutting).toBe(true);
  });

  it("interpolates inside the current segment", () => {
    const sample = sampleAt(track, 5);
    expect(sample!.x).toBeCloseTo(5);
    expect(sample!.y).toBeCloseTo(0);
    const corner = sampleAt(track, 10);
    expect(corner!.x).toBeCloseTo(10);
    expect(corner!.y).toBeCloseTo(0);
  });

  it("starts on a leading zero-length travel move", () => {
    const track = buildTrack([
      { points: [5, 5], power: 0, cutting: false },
      { points: [5, 5, 10, 5], power: 0.8, cutting: true },
    ]);
    const sample = sampleAt(track, 0);
    expect(sample!.x).toBeCloseTo(5);
    expect(sample!.y).toBeCloseTo(5);
    expect(sample!.segmentIndex).toBe(0);
    expect(sample!.cutting).toBe(false);
    const firstCut = sampleAt(track, 2.5);
    expect(firstCut!.x).toBeCloseTo(7.5);
    expect(firstCut!.cutting).toBe(true);
  });

  it("interpolates the second cut segment", () => {
    const sample = sampleAt(track, 30);
    expect(sample!.x).toBeCloseTo(0);
    expect(sample!.y).toBeCloseTo(10);
    expect(sample!.power).toBe(0.5);
  });

  it("clamps positions beyond the track ends", () => {
    const before = sampleAt(track, -5);
    expect(before!.x).toBeCloseTo(0);
    expect(before!.y).toBeCloseTo(0);
    const after = sampleAt(track, 100);
    expect(after!.x).toBeCloseTo(0);
    expect(after!.y).toBeCloseTo(0);
    expect(after!.segmentIndex).toBe(2);
  });

  it("returns null for an empty track", () => {
    expect(sampleAt(buildTrack([]), 0)).toBeNull();
  });
});

describe("boundaries", () => {
  const track = buildTrack(squareTrack());

  it("clamps positions into [0, totalDistance]", () => {
    expect(clampPosition(track, -3)).toBe(0);
    expect(clampPosition(track, 12)).toBeCloseTo(12);
    expect(clampPosition(track, 99)).toBeCloseTo(40);
  });

  it("steps forward to the next segment end, skipping zero-length", () => {
    expect(nextSegmentBoundary(track, 5)).toBeCloseTo(20);
    expect(nextSegmentBoundary(track, 20)).toBeCloseTo(40);
    expect(nextSegmentBoundary(track, 35)).toBeCloseTo(40);
    expect(nextSegmentBoundary(track, 40)).toBeCloseTo(40);
  });

  it("steps back to the current segment start", () => {
    expect(prevSegmentBoundary(track, 5)).toBe(0);
    expect(prevSegmentBoundary(track, 25)).toBeCloseTo(20);
    expect(prevSegmentBoundary(track, 20)).toBe(0);
    expect(prevSegmentBoundary(track, 0)).toBe(0);
  });
});

describe("advancePosition", () => {
  const track = buildTrack(squareTrack());

  it("advances by the delta and clamps", () => {
    expect(advancePosition(track, 0, 7)).toBeCloseTo(7);
    expect(advancePosition(track, 38, 10)).toBeCloseTo(40);
    expect(advancePosition(track, 5, -10)).toBe(0);
  });
});

describe("playbackDeltaMm", () => {
  const track = buildTrack(squareTrack());

  it("derives the rate from the time estimate", () => {
    expect(playbackDeltaMm(track, 1, 1, 40)).toBeCloseTo(1);
    expect(playbackDeltaMm(track, 10, 0.5, 40)).toBeCloseTo(5);
  });

  it("falls back to the default rate without an estimate", () => {
    expect(playbackDeltaMm(track, 1, 2, null)).toBeCloseTo(50);
  });
});
