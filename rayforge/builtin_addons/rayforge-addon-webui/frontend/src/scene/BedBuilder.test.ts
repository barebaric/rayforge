import { describe, expect, it } from "vitest";
import { axisArrowDirections, zoneOutlinePoints } from "./BedBuilder";

describe("axisArrowDirections", () => {
  it("points both arrows into the bed for a bottom-left origin", () => {
    const dirs = axisArrowDirections({
      x_axis_right: false,
      y_axis_down: false,
    });
    expect(dirs.x).toEqual({ x: 1, y: 0, z: 0 });
    expect(dirs.y).toEqual({ x: 0, y: 0, z: -1 });
  });

  it("points X left for a top-right origin (axis grows leftward)", () => {
    const dirs = axisArrowDirections({
      x_axis_right: true,
      y_axis_down: true,
    });
    expect(dirs.x.x).toBe(-1);
    expect(dirs.y.z).toBe(1);
  });

  it("splits the corners independently", () => {
    const rightOnly = axisArrowDirections({
      x_axis_right: true,
      y_axis_down: false,
    });
    expect(rightOnly.x.x).toBe(-1);
    expect(rightOnly.y.z).toBe(-1);
    const topOnly = axisArrowDirections({
      x_axis_right: false,
      y_axis_down: true,
    });
    expect(topOnly.x.x).toBe(1);
    expect(topOnly.y.z).toBe(1);
  });
});

describe("zoneOutlinePoints", () => {
  it("projects a flat panel polygon onto the scene ground plane", () => {
    const pts = zoneOutlinePoints([10, 20, 30, 20, 30, 40, 10, 40]);
    expect(pts).toHaveLength(4);
    expect(pts[0]).toEqual({ x: 10, y: expect.any(Number), z: -20 });
    expect(pts[1]).toEqual({ x: 30, y: expect.any(Number), z: -20 });
  });

  it("places the outline slightly above the bed", () => {
    const pts = zoneOutlinePoints([0, 0, 10, 0, 10, 10]);
    expect(pts[0]!.y).toBeGreaterThan(0);
  });

  it("returns an empty outline for degenerate polygons", () => {
    expect(zoneOutlinePoints([5, 5])).toHaveLength(1);
  });
});

