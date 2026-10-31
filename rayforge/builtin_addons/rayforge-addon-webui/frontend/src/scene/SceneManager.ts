import * as THREE from "three";
import type { ApiBackend } from "../api/client";
import type {
  DocTree,
  ItemTransform,
  Vec2,
  ViewportContract,
} from "../api/types";
import { SCENE_BG } from "./colors";
import { sceneToPanel } from "./coords";
import { BedBuilder } from "./BedBuilder";
import { BeamIndicator } from "./BeamIndicator";
import { CameraRig, type ViewPreset } from "./CameraRig";
import { GizmoHelper } from "./GizmoHelper";
import { tipDirection, type GizmoHit, type TipId } from "./gizmoNav";
import { OpsPreviewLayer, type OpsPreviewData } from "./OpsPreviewLayer";
import { Playhead } from "./Playhead";
import { WorkpieceLayer } from "./WorkpieceLayer";
import { GizmoOverlay } from "./interaction/GizmoOverlay";
import {
  InteractionManager,
  type CommitItem,
} from "./interaction/InteractionManager";
import { type Xform } from "./interaction/TransformController";

export type { CommitItem };

const GROUND_PLANE = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const GRID_FINE_DISTANCE = 700;
const GIZMO_CLICK_SLOP_PX = 5;
/** Clears the 38px topbar so the badge sits snug under it. */
const GIZMO_OFFSET_Y = 46;

export interface SceneManagerCallbacks {
  onClick?(hitUid: string | null, ev: PointerEvent): void;
  onDragSelect?(uid: string): void;
  onPreview?(uid: string | null, transform: ItemTransform | null): void;
  onCommit?(items: CommitItem[]): void;
  getTransform?(uid: string): Xform | null;
}

/**
 * Owns the renderer, the render loop and every scene sub-module.
 * The app talks to the scene exclusively through this class.
 */
export class SceneManager {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly rig: CameraRig;
  private readonly bed = new BedBuilder();
  private readonly workpieces: WorkpieceLayer;
  private readonly beam = new BeamIndicator();
  private readonly opsPreview: OpsPreviewLayer;
  private readonly playhead = new Playhead();
  private readonly gizmo = new GizmoHelper();
  private readonly raycaster = new THREE.Raycaster();
  private readonly resizeObserver: ResizeObserver;
  private readonly container: HTMLElement;
  private readonly selectionGizmo = new GizmoOverlay();
  private interaction: InteractionManager | null = null;
  private selection: string[] = [];
  private viewport: ViewportContract | null = null;
  private clock = new THREE.Clock();
  private frameCount = 0;
  private gizmoPress: {
    x: number;
    y: number;
    hit: GizmoHit;
  } | null = null;
  private gizmoCursor = false;

  constructor(
    container: HTMLElement,
    api: ApiBackend,
    private readonly callbacks: SceneManagerCallbacks = {},
  ) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(SCENE_BG);
    this.renderer.autoClear = true;
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(SCENE_BG);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 0.7);
    key.position.set(-300, 500, 400);
    this.scene.add(key);

    this.workpieces = new WorkpieceLayer(api, () =>
      this.viewport ? this.viewport.world_to_panel : [1, 0, 0, 0, 1, 0],
    );
    this.opsPreview = new OpsPreviewLayer(() =>
      this.viewport ? this.viewport.world_to_panel : [1, 0, 0, 0, 1, 0],
    );
    this.scene.add(this.bed.group);
    this.scene.add(this.workpieces.group);
    this.scene.add(this.opsPreview.group);
    this.scene.add(this.beam.group);
    this.scene.add(this.playhead.group);
    this.scene.add(this.selectionGizmo.group);

    // Registered before the orbit controls so a press on the nav
    // cross can pre-empt orbiting and selection picking.
    this.renderer.domElement.addEventListener(
      "pointerdown",
      this.onGizmoPointerDown,
    );
    window.addEventListener("pointermove", this.onGizmoPointerMove);
    window.addEventListener("pointerup", this.onGizmoPointerUp);

    this.rig = new CameraRig(this.renderer.domElement);
    this.gizmo.setScreenOffset(this.gizmoOffsetX(), GIZMO_OFFSET_Y);

    this.interaction = new InteractionManager({
      dom: this.renderer.domElement,
      camera: () => this.rig.camera,
      canvasRect: () => this.renderer.domElement.getBoundingClientRect(),
      worldToPanel: () =>
        this.viewport ? this.viewport.world_to_panel : [1, 0, 0, 0, 1, 0],
      panelAt: (clientX, clientY) =>
        this.panelPointAt(clientX, clientY),
      workpieces: this.workpieces,
      gizmo: this.selectionGizmo,
      controls: this.rig.controls,
      selection: () => this.selection,
      transformOf: (uid) => this.callbacks.getTransform?.(uid) ?? null,
      callbacks: {
        onClick: (hitUid, ev) => this.callbacks.onClick?.(hitUid, ev),
        onDragSelect: (uid) => this.callbacks.onDragSelect?.(uid),
        onPreview: (uid, transform) =>
          this.callbacks.onPreview?.(uid, transform),
        onCommit: (items) => this.callbacks.onCommit?.(items),
      },
    });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();

    this.renderer.setAnimationLoop(() => this.tick());
  }

  setViewport(vp: ViewportContract): void {
    if (this.viewport === vp) return;
    if (
      !Number.isFinite(vp.extents.width_mm) ||
      !Number.isFinite(vp.extents.height_mm) ||
      vp.extents.width_mm <= 0 ||
      vp.extents.height_mm <= 0
    ) {
      return;
    }
    const first = this.viewport === null;
    this.viewport = vp;
    this.bed.build(vp);
    void this.workpieces.reproject();
    this.opsPreview.viewportChanged();
    this.rig.setHome(vp.workarea);
    if (first) this.rig.preset("iso", vp.workarea);
  }

  async syncTree(tree: DocTree): Promise<void> {
    await this.workpieces.sync(tree);
  }

  setSelection(uids: string[]): void {
    this.selection = [...uids];
    this.workpieces.setSelection(uids);
    this.interaction?.setSelection(uids);
  }

  /** Drops any in-flight drag preview without committing. */
  discardPreviews(): void {
    this.workpieces.clearAllPreviews();
  }

  /** True while a gizmo drag is in progress. */
  get dragging(): boolean {
    return this.interaction !== null && this.interaction.dragging;
  }

  setBeam(panel: Vec2 | null): void {
    this.beam.setPosition(panel);
  }

  setOpsPreview(data: OpsPreviewData | null): void {
    this.opsPreview.setData(data);
  }

  setOpsVisible(visible: boolean): void {
    this.opsPreview.setVisible(visible);
  }

  setTravelVisible(visible: boolean): void {
    this.opsPreview.setTravelVisible(visible);
  }

  setImageVisible(visible: boolean): void {
    this.workpieces.setImageVisible(visible);
  }

  setStockVisible(visible: boolean): void {
    this.workpieces.setStockVisible(visible);
  }

  setNogoZonesVisible(visible: boolean): void {
    this.bed.setNogoZonesVisible(visible);
  }

  setPlayhead(panel: Vec2 | null): void {
    this.playhead.setPosition(panel);
  }

  viewPreset(preset: ViewPreset): void {
    if (!this.viewport) return;
    this.rig.preset(preset, this.viewport.workarea);
  }

  /**
   * Moves the navigation cross on screen; (0, 0) is the canvas
   * top-right corner. The shell can call this when the right panel
   * expands or collapses.
   */
  setNavOffset(x: number, y: number): void {
    this.gizmo.setScreenOffset(x, y);
  }

  /**
   * Default horizontal offset: the canvas top-right corner, above
   * the floating right panel, with a small margin.
   */
  private gizmoOffsetX(): number {
    return 22;
  }

  private gizmoHitAt(ev: PointerEvent): GizmoHit | null {
    if (ev.target !== this.renderer.domElement) return null;
    return this.gizmo.hitTest(
      ev.clientX,
      ev.clientY,
      this.rig.camera,
      this.renderer.domElement.getBoundingClientRect(),
    );
  }

  private onGizmoPointerDown = (ev: PointerEvent): void => {
    if (ev.button !== 0) return;
    const hit = this.gizmoHitAt(ev);
    if (!hit) return;
    ev.stopImmediatePropagation();
    this.gizmoPress = { x: ev.clientX, y: ev.clientY, hit };
    this.rig.controls.enabled = false;
  };

  private onGizmoPointerMove = (ev: PointerEvent): void => {
    if (this.gizmoPress) {
      if (
        Math.hypot(
          ev.clientX - this.gizmoPress.x,
          ev.clientY - this.gizmoPress.y,
        ) > GIZMO_CLICK_SLOP_PX
      ) {
        this.clearGizmoPress();
      }
      return;
    }
    const hit = this.gizmoHitAt(ev);
    const hover: TipId | "home" | null =
      hit !== null && hit !== "center" ? hit : null;
    this.gizmo.setHover(hover);
    this.setGizmoCursor(hit !== null);
  };

  private onGizmoPointerUp = (ev: PointerEvent): void => {
    const press = this.gizmoPress;
    if (!press) return;
    this.clearGizmoPress();
    if (
      Math.hypot(ev.clientX - press.x, ev.clientY - press.y) >
      GIZMO_CLICK_SLOP_PX
    ) {
      return;
    }
    this.activateGizmoHit(press.hit);
  };

  private clearGizmoPress(): void {
    this.gizmoPress = null;
    this.rig.controls.enabled = true;
    this.setGizmoCursor(false);
    this.gizmo.setHover(null);
  }

  private setGizmoCursor(active: boolean): void {
    if (active === this.gizmoCursor) return;
    this.gizmoCursor = active;
    this.renderer.domElement.style.cursor = active ? "pointer" : "";
  }

  private activateGizmoHit(hit: GizmoHit): void {
    if (!this.viewport) return;
    if (hit === "center" || hit === "home") {
      this.rig.preset("iso", this.viewport.workarea);
      return;
    }
    this.rig.viewAxis(tipDirection(hit));
  }

  /** Panel-space (mm) point under the given client coordinates. */
  panelPointAt(clientX: number, clientY: number): Vec2 | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.rig.camera);
    const hit = new THREE.Vector3();
    const intersection = this.raycaster.ray.intersectPlane(GROUND_PLANE, hit);
    if (!intersection) return null;
    return sceneToPanel(hit.x, hit.y, hit.z);
  }

  private resize(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h);
    this.rig.resize(w, h);
  }

  private tick(): void {
    const time = this.clock.getElapsedTime();
    this.rig.update();
    this.beam.tick(time);
    this.playhead.tick(time);
    this.interaction?.frame();

    this.frameCount++;
    if (this.frameCount % 15 === 0) {
      this.bed.setDensity(
        this.rig.distanceToTarget() < GRID_FINE_DISTANCE ? "fine" : "coarse",
      );
    }

    this.renderer.render(this.scene, this.rig.camera);
    this.gizmo.render(this.renderer, this.rig.camera);
  }

  dispose(): void {
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    this.renderer.domElement.removeEventListener(
      "pointerdown",
      this.onGizmoPointerDown,
    );
    window.removeEventListener("pointermove", this.onGizmoPointerMove);
    window.removeEventListener("pointerup", this.onGizmoPointerUp);
    this.gizmo.dispose();
    this.interaction?.dispose();
    this.opsPreview.dispose();
    this.playhead.dispose();
    this.rig.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
