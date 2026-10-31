import type { Affine2D, Vec2, WorkareaRect } from "../api/types";

/**
 * Panel space is the flat 2D bed coordinate system (millimetres) that
 * the web API exposes; its origin and axis directions are defined by
 * the viewport contract. Scene space is the three.js world:
 *
 *   panel X -> scene X
 *   panel Y -> scene -Z  (+Y on the bed points away from the camera)
 *   panel Z -> scene Y   (height above the bed)
 *
 * All modules mapping between the two go through this file.
 */
export interface ScenePoint {
  x: number;
  y: number;
  z: number;
}

export function panelToScene(p: Vec2, height = 0): ScenePoint {
  return { x: p.x, y: height, z: -p.y };
}

export function sceneToPanel(x: number, _y: number, z: number): Vec2 {
  return { x, y: -z };
}

/** Exact equality of two affine matrices (they are small integers or
 * exact translations in practice). */
export function sameAffine(a: Affine2D, b: Affine2D | null): boolean {
  if (!b) return false;
  return a.every((v, i) => v === b[i]);
}

/** Applies the contract's world_to_panel matrix to a world point. */
export function worldToPanel(m: Affine2D, p: Vec2): Vec2 {
  return {
    x: m[0] * p.x + m[1] * p.y + m[2],
    y: m[3] * p.x + m[4] * p.y + m[5],
  };
}

/**
 * Exact inverse of the contract's world_to_panel matrix. The matrix
 * is rigid (rotation + translation), so the determinant is ±1 and
 * the inversion is exact.
 */
export function invertAffine(m: Affine2D): Affine2D {
  const [a, b, tx, d, e, ty] = m;
  const det = a * e - b * d;
  if (det === 0) return [1, 0, 0, 0, 1, 0];
  const raw: Affine2D = [
    e / det,
    -b / det,
    -((e / det) * tx + (-b / det) * ty),
    -d / det,
    a / det,
    -((-d / det) * tx + (a / det) * ty),
  ];
  return raw.map((v) => (v === 0 ? 0 : v)) as Affine2D;
}

/** Maps a panel point back to world space (inverse of worldToPanel). */
export function panelToWorld(m: Affine2D, p: Vec2): Vec2 {
  return worldToPanel(invertAffine(m), p);
}

/** Maps a world displacement into panel space (linear part only). */
export function worldDeltaToPanel(m: Affine2D, d: Vec2): Vec2 {
  return { x: m[0] * d.x + m[1] * d.y, y: m[3] * d.x + m[4] * d.y };
}

/** Maps a panel displacement back into world space. */
export function panelDeltaToWorld(m: Affine2D, d: Vec2): Vec2 {
  const inv = invertAffine(m);
  return { x: inv[0] * d.x + inv[1] * d.y, y: inv[3] * d.x + inv[4] * d.y };
}

export function worldPolylineToPanel(
  m: Affine2D,
  polyline: number[],
): number[] {
  const out: number[] = new Array(polyline.length);
  for (let i = 0; i + 1 < polyline.length; i += 2) {
    const p = worldToPanel(m, { x: polyline[i]!, y: polyline[i + 1]! });
    out[i] = p.x;
    out[i + 1] = p.y;
  }
  return out;
}

export function transformWorkarea(
  m: Affine2D,
  rect: WorkareaRect,
): WorkareaRect {
  const p1 = worldToPanel(m, { x: rect.x_min, y: rect.y_min });
  const p2 = worldToPanel(m, { x: rect.x_max, y: rect.y_max });
  return {
    x_min: Math.min(p1.x, p2.x),
    y_min: Math.min(p1.y, p2.y),
    x_max: Math.max(p1.x, p2.x),
    y_max: Math.max(p1.y, p2.y),
  };
}

export function pointInPolygon(pt: Vec2, poly: number[]): boolean {
  let inside = false;
  const n = Math.floor(poly.length / 2);
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = poly[i * 2]!;
    const yi = poly[i * 2 + 1]!;
    const xj = poly[j * 2]!;
    const yj = poly[j * 2 + 1]!;
    const intersects =
      yi > pt.y !== yj > pt.y &&
      pt.x < ((xj - xi) * (pt.y - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}
