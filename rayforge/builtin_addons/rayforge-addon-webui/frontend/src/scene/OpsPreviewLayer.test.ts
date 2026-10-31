import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { OpsPreviewLayer, type OpsPreviewData } from "./OpsPreviewLayer";

const IDENTITY = [1, 0, 0, 0, 1, 0] as [
  number,
  number,
  number,
  number,
  number,
  number,
];

function makeData(): OpsPreviewData {
  return {
    generationId: "gen-1",
    segments: [
      { points: [0, 0, 10, 10], power: 0.8, rate: 1000, cutting: true },
      { points: [10, 10, 40, 40], power: 0, rate: null, cutting: false },
    ],
  };
}

describe("OpsPreviewLayer travel visibility", () => {
  it("shows both line sets by default", () => {
    const layer = new OpsPreviewLayer(() => IDENTITY);
    layer.setData(makeData());
    expect(layer.group.visible).toBe(true);
  });

  it("hides travel lines but keeps the group when travels are off", () => {
    const layer = new OpsPreviewLayer(() => IDENTITY);
    layer.setData(makeData());
    layer.setTravelVisible(false);
    const [cutLines, travelLines] = layer.group.children as THREE.LineSegments[];
    expect(cutLines).toBeDefined();
    expect(travelLines).toBeDefined();
    expect(travelLines.visible).toBe(false);
    expect(cutLines!.visible).toBe(true);
    expect(layer.group.visible).toBe(true);
  });

  it("applies the travel flag to later builds too", () => {
    const layer = new OpsPreviewLayer(() => IDENTITY);
    layer.setTravelVisible(false);
    layer.setData(makeData());
    const [, travelLines] = layer.group.children as THREE.LineSegments[];
    expect(travelLines.visible).toBe(false);
    layer.setTravelVisible(true);
    expect(travelLines.visible).toBe(true);
  });
});
