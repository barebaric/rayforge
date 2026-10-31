import { describe, expect, it } from "vitest";
import type { Affine2D } from "../api/types";
import {
  invertAffine,
  panelDeltaToWorld,
  panelToScene,
  panelToWorld,
  pointInPolygon,
  sameAffine,
  sceneToPanel,
  worldDeltaToPanel,
  worldPolylineToPanel,
  worldToPanel,
} from "./coords";

const IDENTITY: Affine2D = [1, 0, 0, 0, 1, 0];
const ROTATE_90_CCW: Affine2D = [0, -1, 0, 1, 0, 0];
const FLIP_Y_SHIFT: Affine2D = [1, 0, 0, 0, -1, 100];

describe("worldToPanel", () => {
  it("applies identity unchanged", () => {
    expect(worldToPanel(IDENTITY, { x: 3, y: 4 })).toEqual({ x: 3, y: 4 });
  });

  it("rotates 90 degrees counter-clockwise", () => {
    expect(worldToPanel(ROTATE_90_CCW, { x: 1, y: 0 })).toEqual({
      x: 0,
      y: 1,
    });
    expect(worldToPanel(ROTATE_90_CCW, { x: 0, y: 1 })).toEqual({
      x: -1,
      y: 0,
    });
  });

  it("handles flipped Y with offset", () => {
    expect(worldToPanel(FLIP_Y_SHIFT, { x: 0, y: 0 })).toEqual({
      x: 0,
      y: 100,
    });
    expect(worldToPanel(FLIP_Y_SHIFT, { x: 5, y: 20 })).toEqual({
      x: 5,
      y: 80,
    });
  });
});

describe("panel/scene mapping", () => {
  it("maps panel Y to negative scene Z", () => {
    expect(panelToScene({ x: 10, y: 20 })).toEqual({
      x: 10,
      y: 0,
      z: -20,
    });
  });

  it("maps height to scene Y", () => {
    expect(panelToScene({ x: 10, y: 20 }, 1.5)).toEqual({
      x: 10,
      y: 1.5,
      z: -20,
    });
  });

  it("round-trips through sceneToPanel", () => {
    const samples = [
      { x: 0, y: 0 },
      { x: 12.5, y: -3 },
      { x: 400, y: 300 },
    ];
    for (const p of samples) {
      const s = panelToScene(p, 0.2);
      expect(sceneToPanel(s.x, s.y, s.z)).toEqual(p);
    }
  });

  it("maps rotated world geometry into the scene plane", () => {
    const panel = worldToPanel(ROTATE_90_CCW, { x: 1, y: 0 });
    const scene = panelToScene(panel, 0);
    expect(scene).toEqual({ x: 0, y: 0, z: -1 });
  });
});

describe("worldPolylineToPanel", () => {
  it("transforms every coordinate pair", () => {
    const out = worldPolylineToPanel(FLIP_Y_SHIFT, [0, 0, 5, 20]);
    expect(out).toEqual([0, 100, 5, 80]);
  });
});

describe("pointInPolygon", () => {
  it("detects points inside a rectangle", () => {
    const square = [0, 0, 10, 0, 10, 10, 0, 10, 0, 0];
    expect(pointInPolygon({ x: 5, y: 5 }, square)).toBe(true);
    expect(pointInPolygon({ x: 15, y: 5 }, square)).toBe(false);
    expect(pointInPolygon({ x: -1, y: -1 }, square)).toBe(false);
  });
});

describe("invertAffine", () => {
  it("inverts the identity into itself", () => {
    expect(invertAffine(IDENTITY)).toEqual([1, 0, 0, 0, 1, 0]);
  });

  it("round-trips a rotated matrix", () => {
    const inverse = invertAffine(ROTATE_90_CCW);
    const p = { x: 7, y: -3 };
    const back = worldToPanel(inverse, worldToPanel(ROTATE_90_CCW, p));
    expect(back.x).toBeCloseTo(p.x, 10);
    expect(back.y).toBeCloseTo(p.y, 10);
  });

  it("round-trips a mirrored, translated matrix", () => {
    const m: Affine2D = [0, 1, 5, 1, 0, -2];
    const inverse = invertAffine(m);
    const p = { x: 12.5, y: 4 };
    const back = worldToPanel(inverse, worldToPanel(m, p));
    expect(back.x).toBeCloseTo(p.x, 10);
    expect(back.y).toBeCloseTo(p.y, 10);
  });

  it("falls back to identity for a singular matrix", () => {
    expect(invertAffine([0, 0, 1, 0, 0, 2])).toEqual([1, 0, 0, 0, 1, 0]);
  });
});

describe("panelToWorld", () => {
  it("is the exact inverse of worldToPanel", () => {
    const samples: Array<[Affine2D, { x: number; y: number }]> = [
      [IDENTITY, { x: 3, y: 4 }],
      [ROTATE_90_CCW, { x: 9, y: 1 }],
      [FLIP_Y_SHIFT, { x: 5, y: 20 }],
    ];
    for (const [m, p] of samples) {
      const round = panelToWorld(m, worldToPanel(m, p));
      expect(round.x).toBeCloseTo(p.x, 10);
      expect(round.y).toBeCloseTo(p.y, 10);
    }
  });
});

describe("delta conversions", () => {
  it("maps world deltas through the linear part only", () => {
    const shifted: Affine2D = [1, 0, 50, 0, 1, 50];
    expect(worldDeltaToPanel(shifted, { x: 1, y: 2 })).toEqual({
      x: 1,
      y: 2,
    });
  });

  it("inverts deltas through the mirrored matrix", () => {
    const d = panelDeltaToWorld(FLIP_Y_SHIFT, { x: 3, y: -4 });
    expect(d).toEqual({ x: 3, y: 4 });
    const rotated = panelDeltaToWorld(ROTATE_90_CCW, { x: 0, y: 1 });
    expect(rotated.x).toBeCloseTo(1, 10);
    expect(rotated.y).toBeCloseTo(0, 10);
  });

  it("round-trips deltas for every sample matrix", () => {
    for (const m of [IDENTITY, ROTATE_90_CCW, FLIP_Y_SHIFT]) {
      const d = { x: 2, y: -7 };
      const round = worldDeltaToPanel(m, panelDeltaToWorld(m, d));
      expect(round.x).toBeCloseTo(d.x, 10);
      expect(round.y).toBeCloseTo(d.y, 10);
    }
  });
});

describe("sameAffine", () => {
  it("matches equal matrices and rejects changed or missing ones", () => {
    expect(sameAffine(IDENTITY, [1, 0, 0, 0, 1, 0])).toBe(true);
    expect(sameAffine(IDENTITY, IDENTITY)).toBe(true);
    expect(sameAffine(IDENTITY, FLIP_Y_SHIFT)).toBe(false);
    expect(sameAffine(IDENTITY, null)).toBe(false);
  });
});
