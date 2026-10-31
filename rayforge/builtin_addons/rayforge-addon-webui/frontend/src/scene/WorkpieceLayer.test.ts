import { describe, expect, it } from "vitest";
import type { ApiBackend } from "../api/client";
import type { DocTree } from "../api/types";
import { WorkpieceLayer } from "./WorkpieceLayer";

const IDENTITY = [1, 0, 0, 0, 1, 0] as [number, number, number, number, number, number];
const ROTATED = [0, -1, 200, 1, 0, 0] as [
  number,
  number,
  number,
  number,
  number,
  number,
];

const SQUARE = [0, 0, 20, 0, 20, 20, 0, 20, 0, 0];

function makeApi(): ApiBackend {
  return {
    getItemGeometry: async () => ({
      rev: 1,
      polylines: [SQUARE],
      outline: { x_min: 0, y_min: 0, x_max: 20, y_max: 20 },
    }),
    getItemTexture: async () => null,
  } as unknown as ApiBackend;
}

function makeTree(): DocTree {
  return {
    rev: 1,
    assets_count: 1,
    stock_items: [],
    layers: [
      {
        uid: "l1",
        name: "Layer",
        visible: true,
        color: "#4d9fff",
        is_active: true,
        wcs: null,
        steps: [],
        workpieces: [
          {
            uid: "wp1",
            name: "Square",
            pos_mm: { x: 0, y: 0 },
            size_mm: { w: 20, h: 20 },
            angle_deg: 0,
            geometry_revision: 1,
            transform_revision: 1,
          },
        ],
      },
    ],
  };
}

describe("WorkpieceLayer reproject", () => {
  it("keeps geometry when the panel mapping is unchanged", async () => {
    let matrix = IDENTITY;
    const layer = new WorkpieceLayer(makeApi(), () => matrix);
    await layer.sync(makeTree());
    expect(layer.pick({ x: 10, y: 10 })).toBe("wp1");
    layer.reproject();
    expect(layer.pick({ x: 10, y: 10 })).toBe("wp1");
  });

  it("re-bakes geometry against a changed world_to_panel matrix", async () => {
    let matrix = IDENTITY;
    const layer = new WorkpieceLayer(makeApi(), () => matrix);
    await layer.sync(makeTree());
    expect(layer.pick({ x: 10, y: 10 })).toBe("wp1");

    matrix = ROTATED;
    await layer.reproject();
    expect(layer.pick({ x: 10, y: 10 })).toBeNull();
    expect(layer.pick({ x: 190, y: 10 })).toBe("wp1");
  });

  it("is a no-op before anything was loaded", () => {
    let matrix = IDENTITY;
    const layer = new WorkpieceLayer(makeApi(), () => matrix);
    matrix = ROTATED;
    expect(() => layer.reproject()).not.toThrow();
  });
});
