import { describe, expect, it } from "vitest";
import { worldToPanel } from "../coords";
import {
  centerOf,
  cornerWorld,
  previewAffine,
  snapDegrees,
  toItemTransform,
  TransformController,
  wrapAngle,
  xformOf,
  type Xform,
} from "./TransformController";

const RECT: Xform = {
  pos: { x: 10, y: 20 },
  size: { w: 30, h: 40 },
  angleDeg: 0,
};

const ROTATED: Xform = {
  pos: { x: 0, y: 0 },
  size: { w: 30, h: 40 },
  angleDeg: 90,
};

describe("geometry helpers", () => {
  it("computes rect corners", () => {
    expect(cornerWorld(RECT, 0)).toEqual({ x: 10, y: 20 });
    expect(cornerWorld(RECT, 2)).toEqual({ x: 40, y: 60 });
  });

  it("rotates corners with the item", () => {
    const c = cornerWorld(ROTATED, 2);
    expect(c.x).toBeCloseTo(-40, 10);
    expect(c.y).toBeCloseTo(30, 10);
  });

  it("keeps the center invariant under rotation", () => {
    const center = centerOf(RECT);
    expect(center.x).toBeCloseTo(25, 10);
    expect(center.y).toBeCloseTo(40, 10);
    const rotatedCenter = centerOf(ROTATED);
    expect(rotatedCenter.x).toBeCloseTo(-20, 10);
    expect(rotatedCenter.y).toBeCloseTo(15, 10);
  });

  it("wraps angles into (-PI, PI]", () => {
    expect(wrapAngle(0)).toBeCloseTo(0, 10);
    expect(wrapAngle(Math.PI)).toBeCloseTo(Math.PI, 10);
    expect(wrapAngle(-Math.PI)).toBeCloseTo(Math.PI, 10);
    expect(wrapAngle(3 * Math.PI)).toBeCloseTo(Math.PI, 10);
  });

  it("snaps to a step", () => {
    expect(snapDegrees(43, 15)).toBe(45);
    expect(snapDegrees(-7, 15)).toBe(0);
  });

  it("converts to and from the tree contract", () => {
    const t = toItemTransform(RECT);
    expect(t).toEqual({
      pos_mm: { x: 10, y: 20 },
      size_mm: { w: 30, h: 40 },
      angle_deg: 0,
    });
    expect(xformOf(t)).toEqual(RECT);
  });
});

describe("TransformController move", () => {
  it("translates the start transform by the pointer delta", () => {
    const controller = new TransformController();
    expect(controller.active).toBe(false);
    controller.begin({ kind: "move", transform: RECT, pointer: { x: 0, y: 0 } });
    expect(controller.active).toBe(true);
    expect(controller.kind).toBe("move");
    const next = controller.update({ pointer: { x: 5, y: -3 } });
    expect(next).toEqual({
      pos: { x: 15, y: 17 },
      size: { w: 30, h: 40 },
      angleDeg: 0,
    });
  });

  it("reports the start transform for cancel/restore", () => {
    const controller = new TransformController();
    controller.begin({ kind: "move", transform: RECT, pointer: { x: 1, y: 1 } });
    controller.update({ pointer: { x: 9, y: 9 } });
    expect(controller.start?.transform).toEqual(RECT);
    controller.cancel();
    expect(controller.active).toBe(false);
    expect(controller.current).toBeNull();
  });
});

describe("TransformController scale", () => {
  it("keeps the opposite corner anchored", () => {
    const controller = new TransformController();
    controller.begin({
      kind: "scale",
      transform: RECT,
      pointer: cornerWorld(RECT, 2),
      corner: 2,
    });
    const next = controller.update({
      pointer: { x: 50, y: 65 },
    });
    expect(next?.pos.x).toBeCloseTo(10, 10);
    expect(next?.pos.y).toBeCloseTo(20, 10);
    expect(next?.size).toEqual({ w: 40, h: 45 });
    expect(next?.angleDeg).toBe(0);
  });

  it("scales toward the anchor when dragging the top-left corner", () => {
    const controller = new TransformController();
    controller.begin({
      kind: "scale",
      transform: RECT,
      pointer: cornerWorld(RECT, 0),
      corner: 0,
    });
    const next = controller.update({ pointer: { x: 5, y: 12 } });
    expect(next?.size).toEqual({ w: 35, h: 48 });
    expect(next?.pos.x).toBeCloseTo(5, 10);
    expect(next?.pos.y).toBeCloseTo(12, 10);
    const anchor = cornerWorld(RECT, 2);
    expect(cornerWorld(next!, 2).x).toBeCloseTo(anchor.x, 6);
    expect(cornerWorld(next!, 2).y).toBeCloseTo(anchor.y, 6);
  });

  it("clamps the size to the minimum", () => {
    const controller = new TransformController();
    controller.begin({
      kind: "scale",
      transform: RECT,
      pointer: cornerWorld(RECT, 2),
      corner: 2,
    });
    const next = controller.update({
      pointer: { x: 10.2, y: 20.1 },
    });
    expect(next?.size.w).toBe(1);
    expect(next?.size.h).toBe(1);
  });

  it("works on rotated items", () => {
    const startCorner = cornerWorld(ROTATED, 2);
    const controller = new TransformController();
    controller.begin({
      kind: "scale",
      transform: ROTATED,
      pointer: startCorner,
      corner: 2,
    });
    const dragged = { x: startCorner.x, y: startCorner.y + 20 };
    const next = controller.update({ pointer: dragged });
    expect(next?.size.w).toBeCloseTo(50, 6);
    expect(next?.size.h).toBeCloseTo(40, 6);
    const anchor = cornerWorld(ROTATED, 0);
    expect(cornerWorld(next!, 0).x).toBeCloseTo(anchor.x, 6);
    expect(cornerWorld(next!, 0).y).toBeCloseTo(anchor.y, 6);
  });
});

describe("TransformController rotate", () => {
  it("rotates around the fixed center", () => {
    const controller = new TransformController();
    const start = {
      pos: { x: 10, y: 40 },
      size: { w: 30, h: 20 },
      angleDeg: 0,
    };
    const center = centerOf(start);
    controller.begin({
      kind: "rotate",
      transform: start,
      pointer: { x: center.x + 10, y: center.y },
    });
    const next = controller.update({
      pointer: { x: center.x, y: center.y + 10 },
    });
    expect(next?.angleDeg).toBeCloseTo(90, 6);
    const nextCenter = centerOf(next!);
    expect(nextCenter.x).toBeCloseTo(center.x, 6);
    expect(nextCenter.y).toBeCloseTo(center.y, 6);
    expect(next?.size).toEqual({ w: 30, h: 20 });
  });

  it("snaps to 15 degrees on request", () => {
    const controller = new TransformController();
    const center = centerOf(RECT);
    controller.begin({
      kind: "rotate",
      transform: RECT,
      pointer: { x: center.x + 10, y: center.y },
    });
    const angle = (41 * Math.PI) / 180;
    const next = controller.update({
      pointer: {
        x: center.x + 10 * Math.cos(angle),
        y: center.y + 10 * Math.sin(angle),
      },
      snap: true,
    });
    expect(next?.angleDeg).toBeCloseTo(45, 6);
  });
});

describe("previewAffine", () => {
  it("is the identity for an unchanged transform", () => {
    const { linear, translation } = previewAffine(RECT, RECT, [
      1, 0, 0, 0, 1, 0,
    ]);
    expect(linear.a).toBeCloseTo(1, 10);
    expect(linear.b).toBeCloseTo(0, 10);
    expect(linear.d).toBeCloseTo(0, 10);
    expect(linear.e).toBeCloseTo(1, 10);
    expect(translation.x).toBeCloseTo(0, 10);
    expect(translation.y).toBeCloseTo(0, 10);
  });

  it("translates by the center delta", () => {
    const moved: Xform = {
      ...RECT,
      pos: { x: 15, y: 17 },
    };
    const { translation } = previewAffine(RECT, moved, [1, 0, 0, 0, 1, 0]);
    expect(translation.x).toBeCloseTo(5, 10);
    expect(translation.y).toBeCloseTo(-3, 10);
  });

  it("maps baked points onto the rotated geometry", () => {
    const rotated: Xform = { ...RECT, angleDeg: 90 };
    const m = [0, 1, 5, -1, 0, 10] as const;
    const { linear, translation } = previewAffine(RECT, rotated, [...m]);
    const apply = (p: { x: number; y: number }) => ({
      x: linear.a * p.x + linear.b * p.y + translation.x,
      y: linear.d * p.x + linear.e * p.y + translation.y,
    });
    for (const corner of [0, 1, 2, 3] as const) {
      const baked = worldToPanel([...m], cornerWorld(RECT, corner));
      const target = worldToPanel([...m], cornerWorld(rotated, corner));
      const got = apply(baked);
      expect(got.x).toBeCloseTo(target.x, 6);
      expect(got.y).toBeCloseTo(target.y, 6);
    }
  });
});
