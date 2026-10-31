import * as THREE from "three";
import type { Vec2 } from "../api/types";
import { BEAM_COLOR } from "./colors";
import { panelToScene, type ScenePoint } from "./coords";

const PLAYHEAD_Y = 0.5;
const TRAIL_POINTS = 24;
const MOVE_EPSILON_SQ = 1e-4;

/**
 * Playback playhead: a bright glowing dot with a short fading trail
 * that follows the interpolated toolpath position. Styled after the
 * live BeamIndicator so both read as "where the laser is".
 */
export class Playhead {
  readonly group = new THREE.Group();
  private readonly core: THREE.Mesh;
  private readonly halo: THREE.Mesh;
  private readonly haloMaterial: THREE.MeshBasicMaterial;
  private readonly trail: THREE.Line;
  private readonly trailPositions: Float32Array;
  private readonly trailGeometry: THREE.BufferGeometry;
  private trailCount = 0;
  private last: ScenePoint | null = null;

  constructor() {
    this.core = new THREE.Mesh(
      new THREE.CircleGeometry(1.5, 24),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
      }),
    );
    this.core.rotation.x = -Math.PI / 2;
    this.core.renderOrder = 7;
    this.group.add(this.core);

    this.haloMaterial = new THREE.MeshBasicMaterial({
      color: BEAM_COLOR,
      transparent: true,
      opacity: 0.4,
      depthWrite: false,
    });
    this.halo = new THREE.Mesh(
      new THREE.CircleGeometry(3.6, 24),
      this.haloMaterial,
    );
    this.halo.rotation.x = -Math.PI / 2;
    this.halo.renderOrder = 6;
    this.group.add(this.halo);

    this.trailPositions = new Float32Array(TRAIL_POINTS * 3);
    this.trailGeometry = new THREE.BufferGeometry();
    this.trailGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(this.trailPositions, 3),
    );
    this.trailGeometry.setDrawRange(0, 0);
    this.trail = new THREE.Line(
      this.trailGeometry,
      new THREE.LineBasicMaterial({
        color: BEAM_COLOR,
        transparent: true,
        opacity: 0.6,
        depthWrite: false,
      }),
    );
    this.trail.renderOrder = 5;
    this.group.add(this.trail);

    this.group.visible = false;
  }

  /** Moves the playhead; null hides it and clears the trail. */
  setPosition(panel: Vec2 | null): void {
    if (!panel) {
      this.group.visible = false;
      this.reset();
      return;
    }
    const p = panelToScene(panel, PLAYHEAD_Y);
    this.group.visible = true;
    this.core.position.set(p.x, p.y, p.z);
    this.halo.position.set(p.x, p.y, p.z);
    const moved =
      !this.last ||
      (this.last.x - p.x) ** 2 + (this.last.y - p.y) ** 2 > MOVE_EPSILON_SQ;
    if (moved) {
      this.pushTrail(p);
      this.last = { ...p };
    }
  }

  /** Pulses the halo; call once per frame with elapsed seconds. */
  tick(time: number): void {
    if (!this.group.visible) return;
    this.haloMaterial.opacity = 0.3 + 0.18 * Math.sin(time * 8);
  }

  private pushTrail(p: ScenePoint): void {
    if (this.trailCount < TRAIL_POINTS) {
      this.trailCount++;
    } else {
      this.trailPositions.copyWithin(0, 3);
    }
    const offset = (this.trailCount - 1) * 3;
    this.trailPositions[offset] = p.x;
    this.trailPositions[offset + 1] = p.y;
    this.trailPositions[offset + 2] = p.z;
    this.trailGeometry.attributes.position.needsUpdate = true;
    this.trailGeometry.setDrawRange(0, this.trailCount);
  }

  private reset(): void {
    this.trailCount = 0;
    this.trailGeometry.setDrawRange(0, 0);
    this.last = null;
  }

  dispose(): void {
    this.core.geometry.dispose();
    (this.core.material as THREE.Material).dispose();
    this.halo.geometry.dispose();
    this.haloMaterial.dispose();
    this.trailGeometry.dispose();
    (this.trail.material as THREE.Material).dispose();
  }
}
