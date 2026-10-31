import * as THREE from "three";
import type { Affine2D } from "../../api/types";
import { panelToScene, worldToPanel } from "../coords";
import {
  centerOf,
  cornerWorld,
  rotateVec,
  type CornerIndex,
  type Xform,
} from "./TransformController";

const HANDLE_Y = 0.6;
const RING_PADDING_MM = 8;
const HANDLE_HIT_PX = 14;
const HANDLE_PX = 9;
const RING_SEGMENTS = 72;

export const SCALE_HANDLE_COLOR = 0x4d9fff;
export const ROTATE_HANDLE_COLOR = 0x3ddc84;
const RING_COLOR = 0x4d9fff;

export type HandleHit =
  | { kind: "scale"; corner: CornerIndex }
  | { kind: "rotate" };

interface HandleTarget {
  object: THREE.Object3D;
  hit: HandleHit;
}

/**
 * Selection gizmo visuals: four corner resize handles, a rotation
 * handle on a ring, and a rect outline of the selection. All handles
 * are kept at a roughly constant pixel size by the render loop.
 */
export class GizmoOverlay {
  readonly group = new THREE.Group();
  private readonly cornerHandles: THREE.Mesh[] = [];
  private readonly rotateHandle: THREE.Mesh;
  private readonly ring: THREE.Line;
  private readonly outline: THREE.Line;
  private visible = false;

  constructor() {
    const cornerGeo = new THREE.SphereGeometry(1, 12, 8);
    for (let i = 0; i < 4; i++) {
      const mesh = new THREE.Mesh(
        cornerGeo,
        new THREE.MeshBasicMaterial({ color: SCALE_HANDLE_COLOR }),
      );
      mesh.renderOrder = 10;
      this.cornerHandles.push(mesh);
      this.group.add(mesh);
    }
    this.rotateHandle = new THREE.Mesh(
      new THREE.SphereGeometry(1, 12, 8),
      new THREE.MeshBasicMaterial({ color: ROTATE_HANDLE_COLOR }),
    );
    this.rotateHandle.renderOrder = 10;
    this.group.add(this.rotateHandle);

    this.ring = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({
        color: RING_COLOR,
        transparent: true,
        opacity: 0.45,
      }),
    );
    this.ring.renderOrder = 9;
    this.group.add(this.ring);

    this.outline = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({
        color: SCALE_HANDLE_COLOR,
        transparent: true,
        opacity: 0.8,
      }),
    );
    this.outline.renderOrder = 9;
    this.group.add(this.outline);

    this.group.visible = false;
  }

  /** Positions the gizmo for a selection transform. */
  show(t: Xform, worldToPanelMatrix: Affine2D): void {
    const corners = ([0, 1, 2, 3] as CornerIndex[]).map((corner) =>
      this.scenePoint(cornerWorld(t, corner), worldToPanelMatrix),
    );
    this.cornerHandles.forEach((handle, i) => {
      const c = corners[i]!;
      handle.position.set(c.x, HANDLE_Y, c.z);
    });

    const center = centerOf(t);
    const centerScene = this.scenePoint(center, worldToPanelMatrix);
    const radius = this.ringRadius(t, worldToPanelMatrix);
    this.ring.geometry.dispose();
    this.ring.geometry = circleGeometry(centerScene, radius);

    const topLocal = rotateVec(t.angleDeg, {
      x: t.size.w / 2,
      y: t.size.h / 2 + RING_PADDING_MM,
    });
    const top = this.scenePoint(
      { x: center.x + topLocal.x, y: center.y + topLocal.y },
      worldToPanelMatrix,
    );
    this.rotateHandle.position.set(top.x, HANDLE_Y, top.z);

    const outlinePts: THREE.Vector3[] = [];
    for (let i = 0; i <= 4; i++) {
      const c = corners[i % 4]!;
      outlinePts.push(new THREE.Vector3(c.x, HANDLE_Y, c.z));
    }
    this.outline.geometry.dispose();
    this.outline.geometry = new THREE.BufferGeometry().setFromPoints(
      outlinePts,
    );

    this.group.visible = true;
    this.visible = true;
  }

  hide(): void {
    this.group.visible = false;
    this.visible = false;
  }

  get shown(): boolean {
    return this.visible;
  }

  /**
   * Hit test in pixel space: returns the handle under the pointer,
   * or null when the pointer is not on a handle.
   */
  pick(
    clientX: number,
    clientY: number,
    camera: THREE.PerspectiveCamera,
    canvas: { left: number; top: number; width: number; height: number },
  ): HandleHit | null {
    if (!this.visible) return null;
    const targets: HandleTarget[] = this.cornerHandles.map(
      (object, i): HandleTarget => ({
        object,
        hit: { kind: "scale", corner: i as CornerIndex },
      }),
    );
    targets.push({ object: this.rotateHandle, hit: { kind: "rotate" } });

    const projected = new THREE.Vector3();
    let bestDist = Infinity;
    let best: HandleHit | null = null;
    for (const target of targets) {
      projected.copy(target.object.position).project(camera);
      const px = canvas.left + ((projected.x + 1) / 2) * canvas.width;
      const py = canvas.top + ((1 - projected.y) / 2) * canvas.height;
      const dist = Math.hypot(px - clientX, py - clientY);
      if (dist <= HANDLE_HIT_PX && dist < bestDist) {
        bestDist = dist;
        best = target.hit;
      }
    }
    return best;
  }

  /** Keeps the handles at a constant on-screen size. */
  updatePixelScale(
    camera: THREE.PerspectiveCamera,
    canvasHeight: number,
  ): void {
    if (!this.visible) return;
    const dist = camera.position.distanceTo(this.group.position);
    const worldPerPixel =
      (2 * Math.max(dist, 1) * Math.tan(((camera.fov / 2) * Math.PI) / 180)) /
      Math.max(canvasHeight, 1);
    const s = Math.max(worldPerPixel * HANDLE_PX, 0.5);
    for (const handle of [...this.cornerHandles, this.rotateHandle]) {
      handle.scale.setScalar(s);
    }
  }

  private ringRadius(t: Xform, worldToPanelMatrix: Affine2D): number {
    const center = worldToPanel(worldToPanelMatrix, centerOf(t));
    const corner = worldToPanel(worldToPanelMatrix, cornerWorld(t, 2));
    return (
      Math.hypot(corner.x - center.x, corner.y - center.y) + RING_PADDING_MM
    );
  }

  private scenePoint(
    world: { x: number; y: number },
    worldToPanelMatrix: Affine2D,
  ): { x: number; z: number } {
    const p = panelToScene(worldToPanel(worldToPanelMatrix, world), 0);
    return { x: p.x, z: p.z };
  }
}

function circleGeometry(
  center: { x: number; z: number },
  radius: number,
): THREE.BufferGeometry {
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i <= RING_SEGMENTS; i++) {
    const a = (i / RING_SEGMENTS) * Math.PI * 2;
    pts.push(
      new THREE.Vector3(
        center.x + radius * Math.cos(a),
        HANDLE_Y,
        center.z + radius * Math.sin(a),
      ),
    );
  }
  return new THREE.BufferGeometry().setFromPoints(pts);
}
