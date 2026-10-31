import { describe, expect, it } from "vitest";
import {
  CHIP_LABEL_MAX,
  chipLabel,
  planRollToChip,
  planRollToPosition,
  snapToBoundary,
} from "./historyTimeline";

describe("planRollToPosition", () => {
  it("undoes everything to reach position 0", () => {
    expect(planRollToPosition(0, 4)).toEqual({ undos: 4, redos: 0 });
  });

  it("redoes everything past the undo chips", () => {
    expect(planRollToPosition(6, 4)).toEqual({ undos: 0, redos: 2 });
  });

  it("does nothing at the current position", () => {
    expect(planRollToPosition(4, 4)).toEqual({ undos: 0, redos: 0 });
  });

  it("rolls partway back to the middle chip", () => {
    expect(planRollToPosition(2, 5)).toEqual({ undos: 3, redos: 0 });
  });

  it("clamps negative positions to the start", () => {
    expect(planRollToPosition(-3, 2)).toEqual({ undos: 2, redos: 0 });
  });
});

describe("planRollToChip", () => {
  const undoLabels = ["Import", "Move", "Scale", "Move"];
  const redoLabels = ["Rotate", "Delete"];

  it("makes chip 3 the last command with no steps", () => {
    expect(planRollToChip(undoLabels, redoLabels, 3)).toEqual({
      undos: 0,
      redos: 0,
    });
  });

  it("rolls back to an earlier undo chip", () => {
    expect(planRollToChip(undoLabels, redoLabels, 1)).toEqual({
      undos: 2,
      redos: 0,
    });
  });

  it("replays a redo chip through redo steps", () => {
    expect(planRollToChip(undoLabels, redoLabels, 4)).toEqual({
      undos: 0,
      redos: 1,
    });
    expect(planRollToChip(undoLabels, redoLabels, 5)).toEqual({
      undos: 0,
      redos: 2,
    });
  });

  it("rejects out-of-range chips", () => {
    expect(planRollToChip(undoLabels, redoLabels, -1)).toBeNull();
    expect(planRollToChip(undoLabels, redoLabels, 6)).toBeNull();
  });
});

describe("chipLabel", () => {
  it("keeps short labels intact", () => {
    expect(chipLabel("Move")).toBe("Move");
  });

  it("truncates long labels with an ellipsis", () => {
    const long = "Change workpiece transform matrix";
    const shown = chipLabel(long);
    expect(shown.length).toBeLessThanOrEqual(CHIP_LABEL_MAX);
    expect(shown.endsWith("…")).toBe(true);
    expect(chipLabel(long)).not.toBe(long);
  });
});

describe("snapToBoundary", () => {
  const boundaries = [0, 40, 82, 120];

  it("snaps to the nearest chip boundary", () => {
    expect(snapToBoundary(2, boundaries)).toBe(0);
    expect(snapToBoundary(38, boundaries)).toBe(1);
    expect(snapToBoundary(100, boundaries)).toBe(2);
    expect(snapToBoundary(110, boundaries)).toBe(3);
  });

  it("prefers the lower boundary on exact ties", () => {
    expect(snapToBoundary(61, boundaries)).toBe(1);
  });
});
