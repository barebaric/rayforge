import type { Affine2D, ItemTransform, Size2, Vec2 } from "../../api/types";
import { worldToPanel } from "../coords";

/**
 * Drag-gizmo math. All state is WORLD space (the document contract);
 * the caller converts panel pointer positions via the inverse of the
 * viewport's world_to_panel matrix. The backend decomposes an item
 * into pos (world position of the item's local origin), size (world
 * scale) and angle (rotation about the item center), so that is the
 * decomposition used here.
 */

export const MIN_SIZE_MM = 1;
export const SNAP_DEGREES = 15;

export type CornerIndex = 0 | 1 | 2 | 3;

/** Local corner fractions: bottom-left, bottom-right, top-right, top-left. */
const CORNER_FRACTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
];

/** World-space transform decomposition of a workpiece. */
export interface Xform {
  pos: Vec2;
  size: Size2;
  angleDeg: number;
}

export function xformOf(info: {
  pos_mm: Vec2;
  size_mm: Size2;
  angle_deg: number;
}): Xform {
  return {
    pos: { x: info.pos_mm.x, y: info.pos_mm.y },
    size: { w: info.size_mm.w, h: info.size_mm.h },
    angleDeg: info.angle_deg,
  };
}

export function toItemTransform(x: Xform): ItemTransform {
  return {
    pos_mm: { x: x.pos.x, y: x.pos.y },
    size_mm: { w: x.size.w, h: x.size.h },
    angle_deg: x.angleDeg,
  };
}

/** Rotates v by deg degrees (CCW, world is Y-up). */
export function rotateVec(deg: number, v: Vec2): Vec2 {
  const rad = (deg * Math.PI) / 180;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return { x: c * v.x - s * v.y, y: s * v.x + c * v.y };
}

/** Rotates v by -deg degrees. */
export function unrotateVec(deg: number, v: Vec2): Vec2 {
  return rotateVec(-deg, v);
}

export function addVec(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}

export function subVec(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

/** World position of one rect corner of the transformed rect. */
export function cornerWorld(t: Xform, corner: CornerIndex): Vec2 {
  const [fx, fy] = CORNER_FRACTIONS[corner];
  return addVec(t.pos, rotateVec(t.angleDeg, { x: fx * t.size.w, y: fy * t.size.h }));
}

/** World position of the rect center (local 0.5, 0.5 anchor). */
export function centerOf(t: Xform): Vec2 {
  return addVec(t.pos, rotateVec(t.angleDeg, { x: t.size.w / 2, y: t.size.h / 2 }));
}

/** Wraps an angle in radians into (-PI, PI]. */
export function wrapAngle(rad: number): number {
  const twoPi = Math.PI * 2;
  let wrapped = rad % twoPi;
  if (wrapped <= -Math.PI) wrapped += twoPi;
  if (wrapped > Math.PI) wrapped -= twoPi;
  return wrapped;
}

export function snapDegrees(deg: number, step: number): number {
  return Math.round(deg / step) * step || 0;
}

export type DragKind = "move" | "scale" | "rotate";

export interface DragStart {
  kind: DragKind;
  transform: Xform;
  pointer: Vec2;
  corner?: CornerIndex;
}

export interface DragUpdate {
  pointer: Vec2;
  snap?: boolean;
}

function scaleTransform(u: DragUpdate, start: DragStart): Xform {
  const corner = start.corner ?? 2;
  const [fx, fy] = CORNER_FRACTIONS[corner];
  const anchor = cornerWorld(start.transform, ((corner + 2) % 4) as CornerIndex);
  const local = unrotateVec(
    start.transform.angleDeg,
    subVec(u.pointer, anchor),
  );
  const w = Math.max(MIN_SIZE_MM, Math.abs(local.x));
  const h = Math.max(MIN_SIZE_MM, Math.abs(local.y));
  const pos = subVec(
    anchor,
    rotateVec(start.transform.angleDeg, { x: (1 - fx) * w, y: (1 - fy) * h }),
  );
  return { pos, size: { w, h }, angleDeg: start.transform.angleDeg };
}

function rotateTransform(u: DragUpdate, start: DragStart): Xform {
  const center = centerOf(start.transform);
  const a0 = Math.atan2(
    start.pointer.y - center.y,
    start.pointer.x - center.x,
  );
  const a1 = Math.atan2(u.pointer.y - center.y, u.pointer.x - center.x);
  let deg = start.transform.angleDeg + (wrapAngle(a1 - a0) * 180) / Math.PI;
  if (u.snap) deg = snapDegrees(deg, SNAP_DEGREES);
  const offset = rotateVec(deg, { x: start.transform.size.w / 2, y: start.transform.size.h / 2 });
  return {
    pos: subVec(center, offset),
    size: { ...start.transform.size },
    angleDeg: deg,
  };
}

function moveTransform(u: DragUpdate, start: DragStart): Xform {
  const delta = subVec(u.pointer, start.pointer);
  return {
    pos: addVec(start.transform.pos, delta),
    size: { ...start.transform.size },
    angleDeg: start.transform.angleDeg,
  };
}

/**
 * Pointer-state machine for one gizmo drag: idle or dragging with a
 * kind. update() derives the preview transform from the start state,
 * so cancel() is implicit (drop the controller state).
 */
export class TransformController {
  private startState: DragStart | null = null;
  private currentTransform: Xform | null = null;

  get active(): boolean {
    return this.startState !== null;
  }

  get kind(): DragKind | null {
    return this.startState?.kind ?? null;
  }

  get start(): DragStart | null {
    return this.startState;
  }

  /** Latest preview produced by update(); null before/after a drag. */
  get current(): Xform | null {
    return this.currentTransform;
  }

  begin(start: DragStart): void {
    this.startState = start;
    this.currentTransform = { ...start.transform };
  }

  update(u: DragUpdate): Xform | null {
    const start = this.startState;
    if (!start) return null;
    const kind = start.kind;
    const next =
      kind === "move"
        ? moveTransform(u, start)
        : kind === "scale"
          ? scaleTransform(u, start)
          : rotateTransform(u, start);
    this.currentTransform = next;
    return next;
  }

  cancel(): void {
    this.startState = null;
    this.currentTransform = null;
  }
}

/**
 * Linear 2x2 matrix [a b; d e].
 */
export interface Mat2 {
  a: number;
  b: number;
  d: number;
  e: number;
}

function mat2Mul(m: Mat2, n: Mat2): Mat2 {
  return {
    a: m.a * n.a + m.b * n.d,
    b: m.a * n.b + m.b * n.e,
    d: m.d * n.a + m.e * n.d,
    e: m.d * n.b + m.e * n.e,
  };
}

function mat2Apply(m: Mat2, v: Vec2): Vec2 {
  return { x: m.a * v.x + m.b * v.y, y: m.d * v.x + m.e * v.y };
}

function mat2Invert(m: Mat2): Mat2 {
  const det = m.a * m.e - m.b * m.d;
  if (det === 0) return { a: 1, b: 0, d: 0, e: 1 };
  return {
    a: m.e / det,
    b: -m.b / det,
    d: -m.d / det,
    e: m.a / det,
  };
}

function affineLinear(m: Affine2D): Mat2 {
  return { a: m[0], b: m[1], d: m[3], e: m[4] };
}

/**
 * Panel-space affine (linear part + translation) that previews
 * `current` on top of geometry baked at `start`. Because the panel
 * mapping is rigid, the same conjugated-scale linear map works for
 * move, resize and rotate; callers embed it into the scene plane.
 */
export function previewAffine(
  start: Xform,
  current: Xform,
  worldToPanelMatrix: Affine2D,
): { linear: Mat2; translation: Vec2 } {
  const sw = Math.abs(start.size.w) < 1e-9 ? MIN_SIZE_MM : start.size.w;
  const sh = Math.abs(start.size.h) < 1e-9 ? MIN_SIZE_MM : start.size.h;
  const rotN = rotateVec(current.angleDeg, { x: 1, y: 0 });
  const rotO = rotateVec(start.angleDeg, { x: 1, y: 0 });
  const scaleN: Mat2 = { a: (current.size.w / sw) * rotN.x, b: -(current.size.h / sh) * rotN.y, d: (current.size.w / sw) * rotN.y, e: (current.size.h / sh) * rotN.x };
  const rotOT: Mat2 = { a: rotO.x, b: rotO.y, d: -rotO.y, e: rotO.x };
  const worldLinear = mat2Mul(scaleN, rotOT);

  const a = affineLinear(worldToPanelMatrix);
  const aInv = mat2Invert(a);
  const panelLinear = mat2Mul(mat2Mul(a, worldLinear), aInv);

  const oldCenter = worldToPanel(worldToPanelMatrix, centerOf(start));
  const newCenter = worldToPanel(worldToPanelMatrix, centerOf(current));
  const translation = subVec(newCenter, mat2Apply(panelLinear, oldCenter));
  return { linear: panelLinear, translation };
}
