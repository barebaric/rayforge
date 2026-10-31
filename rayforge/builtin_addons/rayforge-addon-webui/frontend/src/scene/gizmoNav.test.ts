import { describe, expect, it } from "vitest";
import {
  cameraBasisFromQuaternion,
  pickGizmo,
  projectTips,
  tipDirection,
  TIPS,
  viewForTip,
  type GizmoHit,
  type ProjectedTip,
} from "./gizmoNav";

const CENTER = { centerX: 100, centerY: 80, radius: 38 };
const PICK = {
  ...CENTER,
  tipRadius: 14,
  centerRadius: 7,
};

/** Quaternion for a rotation of `deg` degrees about the X axis. */
function quatX(deg: number): { x: number; y: number; z: number; w: number } {
  const half = (deg * Math.PI) / 360;
  return { x: Math.sin(half), y: 0, z: 0, w: Math.cos(half) };
}

function tipById(
  tips: readonly ProjectedTip[],
  tip: string,
): ProjectedTip {
  const found = tips.find((t) => t.tip === tip);
  expect(found, `projected ${tip}`).toBeDefined();
  return found!;
}

describe("tipDirection", () => {
  it("maps X to scene +X", () => {
    expect(tipDirection("X")).toEqual({ x: 1, y: 0, z: 0 });
  });

  it("maps Y to scene -Z (bed depth)", () => {
    expect(tipDirection("Y")).toEqual({ x: 0, y: 0, z: -1 });
  });

  it("maps Z to scene +Y (up)", () => {
    expect(tipDirection("Z")).toEqual({ x: 0, y: 1, z: 0 });
  });

  it("maps every negative tip to the opposite direction", () => {
    for (const axis of ["X", "Y", "Z"] as const) {
      const pos = tipDirection(axis);
      const neg = tipDirection(`-${axis}`);
      expect(neg.x).toBeCloseTo(-pos.x, 12);
      expect(neg.y).toBeCloseTo(-pos.y, 12);
      expect(neg.z).toBeCloseTo(-pos.z, 12);
    }
  });
});

describe("viewForTip", () => {
  it("snaps the up axis to top and bottom", () => {
    expect(viewForTip("Z")).toBe("top");
    expect(viewForTip("-Z")).toBe("bottom");
  });

  it("snaps the X axis to right and left", () => {
    expect(viewForTip("X")).toBe("right");
    expect(viewForTip("-X")).toBe("left");
  });

  it("snaps the bed axis to back and front", () => {
    expect(viewForTip("Y")).toBe("back");
    expect(viewForTip("-Y")).toBe("front");
  });

  it("covers all six tips", () => {
    expect(new Set(TIPS.map(viewForTip)).size).toBe(6);
  });
});

describe("cameraBasisFromQuaternion", () => {
  it("yields the world axes for the default stance", () => {
    const basis = cameraBasisFromQuaternion({ x: 0, y: 0, z: 0, w: 1 });
    expect(basis.right).toEqual({ x: 1, y: 0, z: 0 });
    expect(basis.up).toEqual({ x: 0, y: 1, z: 0 });
    expect(basis.forward).toEqual({ x: 0, y: 0, z: -1 });
  });

  it("follows a camera pitched down 90 degrees", () => {
    const basis = cameraBasisFromQuaternion(quatX(-90));
    expect(basis.right.x).toBeCloseTo(1, 12);
    expect(basis.forward.y).toBeCloseTo(-1, 12);
    expect(basis.up.z).toBeCloseTo(-1, 12);
  });

  it("keeps the basis orthonormal for a compound rotation", () => {
    const { right, up, forward } = cameraBasisFromQuaternion({
      x: 0.5,
      y: 0.5,
      z: 0.5,
      w: 0.5,
    });
    const len = (v: { x: number; y: number; z: number }) =>
      Math.hypot(v.x, v.y, v.z);
    for (const v of [right, up, forward]) {
      expect(len(v)).toBeCloseTo(1, 10);
    }
    const cross = {
      x: forward.y * up.z - forward.z * up.y,
      y: forward.z * up.x - forward.x * up.z,
      z: forward.x * up.y - forward.y * up.x,
    };
    expect(cross.x).toBeCloseTo(right.x, 10);
    expect(cross.y).toBeCloseTo(right.y, 10);
    expect(cross.z).toBeCloseTo(right.z, 10);
  });
});

describe("projectTips", () => {
  it("projects the default stance onto the screen cross", () => {
    const basis = cameraBasisFromQuaternion({ x: 0, y: 0, z: 0, w: 1 });
    const tips = projectTips(basis, CENTER);

    expect(tipById(tips, "X")).toMatchObject({ x: 138, y: 80, depth: 0 });
    expect(tipById(tips, "-X")).toMatchObject({ x: 62, y: 80, depth: 0 });
    expect(tipById(tips, "Z")).toMatchObject({ x: 100, y: 42, depth: 0 });
    expect(tipById(tips, "-Z")).toMatchObject({ x: 100, y: 118, depth: 0 });
  });

  it("foreshortens the axis pointing at the camera", () => {
    const basis = cameraBasisFromQuaternion({ x: 0, y: 0, z: 0, w: 1 });
    const tips = projectTips(basis, CENTER);
    const toward = tipById(tips, "-Y");
    const away = tipById(tips, "Y");
    expect(toward.x).toBeCloseTo(CENTER.centerX, 10);
    expect(toward.y).toBeCloseTo(CENTER.centerY, 10);
    expect(toward.depth).toBeCloseTo(-1, 10);
    expect(away.depth).toBeCloseTo(1, 10);
  });

  it("rotates tip positions with the camera", () => {
    const basis = cameraBasisFromQuaternion(quatX(-90));
    const tips = projectTips(basis, CENTER);
    expect(tipById(tips, "Z").depth).toBeCloseTo(-1, 10);
    expect(tipById(tips, "-Z").depth).toBeCloseTo(1, 10);
    expect(tipById(tips, "X")).toMatchObject({ x: 138, y: 80 });
  });
});

/** Oblique stance where no axis is foreshortened. */
const OBLIQUE = {
  right: { x: 1, y: 0, z: 0 },
  up: { x: 0, y: 0.6, z: 0.8 },
  forward: { x: 0, y: 0.8, z: -0.6 },
};

describe("pickGizmo", () => {
  const basis = cameraBasisFromQuaternion({ x: 0, y: 0, z: 0, w: 1 });
  const tips = projectTips(basis, CENTER);
  const obliqueTips = projectTips(OBLIQUE, CENTER);

  function pick(
    px: number,
    py: number,
    projected: readonly ProjectedTip[] = obliqueTips,
  ): GizmoHit | null {
    return pickGizmo(px, py, projected, PICK);
  }

  it("hits each tip at its projected position", () => {
    for (const t of obliqueTips) {
      expect(pick(t.x, t.y)).toBe(t.tip);
    }
  });

  it("hits the center dot in an oblique stance", () => {
    expect(pick(CENTER.centerX, CENTER.centerY)).toBe("center");
    expect(
      pick(CENTER.centerX + PICK.centerRadius - 1, CENTER.centerY),
    ).toBe("center");
  });

  it("resolves a foreshortened axis to its front tip", () => {
    expect(pick(CENTER.centerX, CENTER.centerY, tips)).toBe("-Y");
  });

  it("returns null between tips inside the circle", () => {
    expect(pick(CENTER.centerX + 20, CENTER.centerY + 20)).toBeNull();
  });

  it("returns null outside the circle so clicks pass through", () => {
    expect(
      pick(CENTER.centerX + 30, CENTER.centerY + CENTER.radius + 4),
    ).toBeNull();
    expect(pick(2, 2)).toBeNull();
  });

  it("returns null just past a tip's hit radius", () => {
    const x = obliqueTips.find((t) => t.tip === "X")!;
    expect(pick(x.x + PICK.tipRadius + 1, x.y)).toBeNull();
  });

  it("prefers the nearest tip when two overlap", () => {
    const shifted: ProjectedTip[] = [
      { tip: "X", x: 100, y: 100, depth: 0 },
      { tip: "Y", x: 103, y: 101, depth: 0 },
    ];
    expect(pick(101, 100, shifted)).toBe("X");
    expect(pick(104, 101, shifted)).toBe("Y");
  });

  it("prefers the front tip when distances tie", () => {
    const stacked: ProjectedTip[] = [
      { tip: "Y", x: 100, y: 100, depth: 1 },
      { tip: "-Y", x: 100, y: 100, depth: -1 },
    ];
    expect(pick(100, 100, stacked)).toBe("-Y");
  });
});
