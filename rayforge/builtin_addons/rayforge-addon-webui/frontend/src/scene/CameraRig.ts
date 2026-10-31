import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { WorkareaRect } from "../api/types";
import { panelToScene } from "./coords";

export type ViewPreset = "fit" | "top" | "front" | "iso";

type Rect = WorkareaRect;

const FOV = 45;
const TWEEN_MS = 600;
const DEFAULT_MAX_POLAR = Math.PI / 2 + 0.02;
const BOTTOM_MAX_POLAR = Math.PI - 0.02;

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

function rectCenterPanel(rect: Rect): { x: number; y: number } {
  return { x: (rect.x_min + rect.x_max) / 2, y: (rect.y_min + rect.y_max) / 2 };
}

function rectSpan(rect: Rect): number {
  return Math.max(
    rect.x_max - rect.x_min,
    rect.y_max - rect.y_min,
    10,
  );
}

/**
 * Orbit camera with damping, animated framing and view presets.
 * Default stance is a 3/4 view from the front-left of the bed.
 */
export class CameraRig {
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private homeTarget = new THREE.Vector3(150, 0, -100);
  private tween: {
    fromPos: THREE.Vector3;
    toPos: THREE.Vector3;
    fromTarget: THREE.Vector3;
    toTarget: THREE.Vector3;
    start: number;
  } | null = null;

  constructor(domElement: HTMLElement) {
    this.camera = new THREE.PerspectiveCamera(
      FOV,
      domElement.clientWidth / Math.max(domElement.clientHeight, 1),
      0.5,
      8000,
    );
    // OrbitControls capture the pointer on their domElement during
    // drags; binding the whole container would swallow every click on
    // overlays inside it (job bar, view pill), so callers pass the
    // renderer canvas itself.
    this.controls = new OrbitControls(this.camera, domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.maxPolarAngle = DEFAULT_MAX_POLAR;
    this.controls.minDistance = 20;
    this.controls.maxDistance = 4000;
    this.camera.position.set(-160, 320, 420);
    this.controls.target.copy(this.homeTarget);
  }

  setHome(workarea: WorkareaRect): void {
    const center = rectCenterPanel(workarea);
    const c = panelToScene(center, 0);
    this.homeTarget.set(c.x, c.y, c.z);
    if (!this.tween) {
      this.controls.target.copy(this.homeTarget);
    }
  }

  distanceToTarget(): number {
    return this.camera.position.distanceTo(this.controls.target);
  }

  preset(preset: ViewPreset, workarea: WorkareaRect): void {
    const center = panelToScene(rectCenterPanel(workarea), 0);
    const d = rectSpan(workarea) * 1.55;
    const target = new THREE.Vector3(center.x, 0, center.z);
    let pos: THREE.Vector3;
    switch (preset) {
      case "top":
        pos = new THREE.Vector3(center.x, d, center.z + d * 0.001);
        break;
      case "front":
        pos = new THREE.Vector3(center.x, d * 0.3, center.z + d);
        break;
      case "iso":
        pos = new THREE.Vector3(
          center.x - d * 0.65,
          d * 0.8,
          center.z + d * 0.9,
        );
        break;
      case "fit":
      default:
        pos = this.camera.position
          .clone()
          .sub(this.controls.target)
          .normalize()
          .multiplyScalar(d)
          .add(target);
        break;
    }
    this.animateTo(pos, target);
  }

  private animateTo(pos: THREE.Vector3, target: THREE.Vector3): void {
    this.tween = {
      fromPos: this.camera.position.clone(),
      toPos: pos.clone(),
      fromTarget: this.controls.target.clone(),
      toTarget: target.clone(),
      start: performance.now(),
    };
  }

  /**
   * Tweens the camera onto the given scene-space direction while
   * keeping the current orbit distance, looking at the target.
   * Standard stances stay inside the orbit polar clamp; the bottom
   * view widens it until another view is requested (OrbitControls
   * re-derives the position from clamped angles every frame, so the
   * clamp has to allow the stance while it is held).
   *
   * Near-vertical stances nudge the position toward scene +Z relative
   * to the target (like the top preset) so the top/bottom snap keeps
   * the same azimuth as the presets instead of a degenerate roll.
   */
  viewAxis(dir: { x: number; y: number; z: number }): void {
    const dirVec = new THREE.Vector3(dir.x, dir.y, dir.z);
    if (dirVec.lengthSq() < 1e-6) return;
    dirVec.normalize();
    this.controls.maxPolarAngle =
      dirVec.y < -0.999 ? BOTTOM_MAX_POLAR : DEFAULT_MAX_POLAR;
    const target = this.controls.target.clone();
    const distance = this.distanceToTarget();
    const pos = target.clone().addScaledVector(dirVec, distance);
    if (Math.abs(dirVec.y) > 0.999) {
      pos.z = target.z + distance * 0.001;
    }
    this.animateTo(pos, target);
  }

  update(): void {
    if (this.tween) {
      const t = Math.min(
        1,
        (performance.now() - this.tween.start) / TWEEN_MS,
      );
      const k = easeInOutCubic(t);
      this.camera.position.lerpVectors(this.tween.fromPos, this.tween.toPos, k);
      this.controls.target.lerpVectors(
        this.tween.fromTarget,
        this.tween.toTarget,
        k,
      );
      if (t >= 1) this.tween = null;
    }
    this.controls.update();
  }

  resize(width: number, height: number): void {
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
  }

  dispose(): void {
    this.controls.dispose();
  }
}
