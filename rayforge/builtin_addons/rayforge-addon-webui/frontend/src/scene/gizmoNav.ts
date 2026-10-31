/**
 * Pure math for the canvas navigation cross: axis directions, the
 * axis-to-view mapping, screen projection and hit testing. WebGL-free
 * so it can be unit tested directly.
 *
 * Labels follow machine convention while living in scene space:
 *   X -> scene +X,  Y -> scene -Z (bed depth),  Z -> scene +Y (up)
 */

export type TipId = "X" | "-X" | "Y" | "-Y" | "Z" | "-Z";

/** Signals that the center of the cross or the home chip was hit. */
export type GizmoHit = TipId | "center" | "home";

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface CameraBasis {
  /** Camera right in world space (screen +x). */
  right: Vec3;
  /** Camera up in world space (screen -y). */
  up: Vec3;
  /** Camera viewing direction in world space. */
  forward: Vec3;
}

export interface ProjectedTip {
  tip: TipId;
  /** CSS px, canvas top-left origin. */
  x: number;
  y: number;
  /** Dot with the camera forward; positive points away from viewer. */
  depth: number;
}

export interface GizmoGeometry {
  /** Center of the cross, CSS px, canvas top-left origin. */
  centerX: number;
  centerY: number;
  /** Distance from center to each tip, CSS px. */
  radius: number;
}

export interface PickOptions extends GizmoGeometry {
  /** Hit radius around a tip, CSS px. */
  tipRadius: number;
  /** Hit radius around the center dot, CSS px. */
  centerRadius: number;
}

/** Scene-space unit direction each tip points to. */
export function tipDirection(tip: TipId): Vec3 {
  switch (tip) {
    case "X":
      return { x: 1, y: 0, z: 0 };
    case "-X":
      return { x: -1, y: 0, z: 0 };
    case "Y":
      return { x: 0, y: 0, z: -1 };
    case "-Y":
      return { x: 0, y: 0, z: 1 };
    case "Z":
      return { x: 0, y: 1, z: 0 };
    case "-Z":
      return { x: 0, y: -1, z: 0 };
  }
}

export const TIPS: readonly TipId[] = ["X", "-X", "Y", "-Y", "Z", "-Z"];

/** Stance name the camera snaps to when a tip is clicked. */
export type AxisView = "top" | "bottom" | "left" | "right" | "front" | "back";

export function viewForTip(tip: TipId): AxisView {
  switch (tip) {
    case "Z":
      return "top";
    case "-Z":
      return "bottom";
    case "X":
      return "right";
    case "-X":
      return "left";
    case "Y":
      return "back";
    case "-Y":
      return "front";
  }
}

export function isTipId(hit: GizmoHit): hit is TipId {
  return hit !== "center";
}

/** Camera basis vectors from a quaternion as {x, y, z, w}. */
export function cameraBasisFromQuaternion(q: {
  x: number;
  y: number;
  z: number;
  w: number;
}): CameraBasis {
  const { x, y, z, w } = q;
  const clean = (v: number) => (v === 0 ? 0 : v);
  const right = {
    x: clean(1 - 2 * (y * y + z * z)),
    y: clean(2 * (x * y + z * w)),
    z: clean(2 * (x * z - y * w)),
  };
  const up = {
    x: clean(2 * (x * y - z * w)),
    y: clean(1 - 2 * (x * x + z * z)),
    z: clean(2 * (y * z + x * w)),
  };
  const forward = {
    x: clean(-2 * (x * z + y * w)),
    y: clean(-2 * (y * z - x * w)),
    z: clean(-(1 - 2 * (x * x + y * y))),
  };
  return { right, up, forward };
}

/** Screen position of every tip for the given camera orientation. */
export function projectTips(
  basis: CameraBasis,
  geom: GizmoGeometry,
): ProjectedTip[] {
  const dot = (a: Vec3, b: Vec3) => a.x * b.x + a.y * b.y + a.z * b.z;
  return TIPS.map((tip) => {
    const d = tipDirection(tip);
    return {
      tip,
      x: geom.centerX + geom.radius * dot(d, basis.right),
      y: geom.centerY - geom.radius * dot(d, basis.up),
      depth: dot(d, basis.forward),
    };
  });
}

/**
 * Resolves a pointer position against the projected cross. Returns
 * null for anything outside the tip/center hit radii so the click
 * passes through to orbit and selection handling. When candidates
 * sit within a pixel of each other (a foreshortened axis stacks both
 * tips on the center) the front-facing tip wins.
 */
export function pickGizmo(
  px: number,
  py: number,
  tips: readonly ProjectedTip[],
  opts: PickOptions,
): GizmoHit | null {
  const candidates = tips
    .map((t) => ({ t, dist: Math.hypot(px - t.x, py - t.y) }))
    .filter((c) => c.dist <= opts.tipRadius)
    .sort((a, b) => a.dist - b.dist);
  if (candidates.length > 0) {
    const nearest = candidates[0]!.dist;
    const front = candidates
      .filter((c) => c.dist <= nearest + 1)
      .sort((a, b) => a.t.depth - b.t.depth);
    return front[0]!.t.tip;
  }

  const centerDist = Math.hypot(
    px - opts.centerX,
    py - opts.centerY,
  );
  if (centerDist <= opts.centerRadius) return "center";
  return null;
}
