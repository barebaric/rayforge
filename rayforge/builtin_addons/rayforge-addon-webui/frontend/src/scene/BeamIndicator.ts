import * as THREE from "three";
import type { Vec2 } from "../api/types";
import { BEAM_COLOR } from "./colors";
import { panelToScene } from "./coords";

/**
 * Laser head indicator: emissive cone, glow dot and a point light at
 * the beam position reported by machine state.
 */
export class BeamIndicator {
  readonly group = new THREE.Group();
  private readonly cone: THREE.Mesh;
  private readonly dot: THREE.Mesh;
  private readonly light: THREE.PointLight;
  private readonly material: THREE.MeshBasicMaterial;

  constructor() {
    this.material = new THREE.MeshBasicMaterial({
      color: BEAM_COLOR,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
    });

    const coneGeo = new THREE.ConeGeometry(1.1, 9, 16, 1, true);
    this.cone = new THREE.Mesh(coneGeo, this.material);
    this.cone.rotation.x = Math.PI;
    this.group.add(this.cone);

    this.dot = new THREE.Mesh(
      new THREE.CircleGeometry(1.7, 24),
      new THREE.MeshBasicMaterial({
        color: BEAM_COLOR,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
      }),
    );
    this.dot.rotation.x = -Math.PI / 2;
    this.group.add(this.dot);

    this.light = new THREE.PointLight(BEAM_COLOR, 40, 60, 2);
    this.group.add(this.light);

    this.group.visible = false;
    this.group.renderOrder = 5;
  }

  setPosition(panel: Vec2 | null): void {
    if (!panel) {
      this.group.visible = false;
      return;
    }
    const p = panelToScene(panel, 0.25);
    this.group.visible = true;
    this.dot.position.set(p.x, p.y, p.z);
    this.cone.position.set(p.x, p.y + 4.5, p.z);
    this.light.position.set(p.x, p.y + 2, p.z);
  }

  /** Pulses the beam glow; call once per frame with elapsed seconds. */
  tick(time: number): void {
    if (!this.group.visible) return;
    const pulse = 0.45 + 0.2 * Math.sin(time * 6);
    this.material.opacity = pulse;
    (this.dot.material as THREE.MeshBasicMaterial).opacity = 0.7 + 0.3 * pulse;
  }
}
