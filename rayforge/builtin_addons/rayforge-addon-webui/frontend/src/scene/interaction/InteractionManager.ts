import type { PerspectiveCamera } from "three";
import type { Affine2D, ItemTransform } from "../../api/types";
import { panelToWorld } from "../coords";
import { GizmoOverlay } from "./GizmoOverlay";
import {
  toItemTransform,
  TransformController,
  type CornerIndex,
  type Xform,
} from "./TransformController";

const CLICK_SLOP_PX = 5;
const DRAG_SLOP_PX = 3;

export interface CommitItem {
  uid: string;
  transform: ItemTransform;
}

export interface InteractionCallbacks {
  /** Click (pointerup within slop) on a workpiece or empty canvas. */
  onClick(hitUid: string | null, ev: PointerEvent): void;
  /** A move drag started on an unselected workpiece. */
  onDragSelect(uid: string): void;
  /** Live preview values for the right panel; null when a drag ends. */
  onPreview(uid: string | null, transform: ItemTransform | null): void;
  /** Committed transforms after a finished drag. */
  onCommit(items: CommitItem[]): void;
}

export interface WorkpieceSceneAccess {
  pick(panel: { x: number; y: number }): string | null;
  setPreview(
    uid: string,
    start: Xform,
    current: Xform,
    worldToPanelMatrix: Affine2D,
  ): void;
  clearPreview(uid: string): void;
  clearAllPreviews(): void;
  setHover(uid: string | null): void;
}

export interface InteractionDeps {
  dom: HTMLElement;
  camera: () => PerspectiveCamera;
  canvasRect: () => {
    left: number;
    top: number;
    width: number;
    height: number;
  };
  worldToPanel: () => Affine2D;
  panelAt: (
    clientX: number,
    clientY: number,
  ) => { x: number; y: number } | null;
  workpieces: WorkpieceSceneAccess;
  gizmo: GizmoOverlay;
  controls: { enabled: boolean };
  selection: () => string[];
  transformOf: (uid: string) => Xform | null;
  callbacks: InteractionCallbacks;
}

type Mode =
  | { tag: "idle" }
  | {
      tag: "press";
      downAt: { x: number; y: number };
      hitUid: string | null;
    }
  | {
      tag: "drag";
      uid: string;
      others: string[];
      controller: TransformController;
      startTransforms: Map<string, Xform>;
      downAt: { x: number; y: number };
      moved: boolean;
      hitUid: string | null;
    };

/**
 * Canvas interaction: picking, hover, and the move/resize/rotate
 * gizmo drags. Owns the pointer state machine and suppresses the
 * orbit controls while a gizmo drag is in progress.
 */
export class InteractionManager {
  private readonly deps: InteractionDeps;
  private mode: Mode = { tag: "idle" };
  private hoverUid: string | null = null;

  constructor(deps: InteractionDeps) {
    this.deps = deps;
    deps.dom.addEventListener("pointerdown", this.onPointerDown);
    window.addEventListener("pointermove", this.onPointerMove);
    window.addEventListener("pointerup", this.onPointerUp);
    window.addEventListener("keydown", this.onKeyDown);
  }

  setSelection(uids: string[]): void {
    if (this.mode.tag === "drag") return;
    if (uids.length === 1) {
      const t = this.deps.transformOf(uids[0]!);
      if (t) {
        this.deps.gizmo.show(t, this.deps.worldToPanel());
        return;
      }
    }
    this.deps.gizmo.hide();
  }

  /** True while a gizmo drag is in progress. */
  get dragging(): boolean {
    return this.mode.tag === "drag";
  }

  /** Per-frame upkeep (constant-size handles). */
  frame(): void {
    if (!this.deps.gizmo.shown) return;
    this.deps.gizmo.updatePixelScale(
      this.deps.camera(),
      this.deps.canvasRect().height,
    );
  }

  /** Reverts an in-progress drag and restores the start state. */
  cancelDrag(): void {
    if (this.mode.tag !== "drag") return;
    const controller = this.mode.controller;
    this.mode = { tag: "idle" };
    this.deps.workpieces.clearAllPreviews();
    const start = controller.start;
    if (start) {
      this.deps.gizmo.show(start.transform, this.deps.worldToPanel());
    }
    controller.cancel();
    this.deps.controls.enabled = true;
    this.deps.callbacks.onPreview(null, null);
  }

  dispose(): void {
    this.deps.dom.removeEventListener("pointerdown", this.onPointerDown);
    window.removeEventListener("pointermove", this.onPointerMove);
    window.removeEventListener("pointerup", this.onPointerUp);
    window.removeEventListener("keydown", this.onKeyDown);
  }

  private beginDrag(
    uid: string,
    kind: "move" | "scale" | "rotate",
    ev: PointerEvent,
    hitUid: string | null,
    corner?: CornerIndex,
  ): boolean {
    const transform = this.deps.transformOf(uid);
    if (!transform) return false;
    const panel = this.deps.panelAt(ev.clientX, ev.clientY);
    if (!panel) return false;
    const world = panelToWorld(this.deps.worldToPanel(), panel);

    const selection = this.deps.selection();
    const others = kind === "move"
      ? selection.filter((id) => id !== uid)
      : [];

    const controller = new TransformController();
    controller.begin({ kind, transform, pointer: world, corner });

    const startTransforms = new Map<string, Xform>([[uid, transform]]);
    for (const id of others) {
      const t = this.deps.transformOf(id);
      if (t) startTransforms.set(id, t);
    }

    this.mode = {
      tag: "drag",
      uid,
      others,
      controller,
      startTransforms,
      downAt: { x: ev.clientX, y: ev.clientY },
      moved: false,
      hitUid,
    };
    this.deps.controls.enabled = false;
    this.setHover(null);
    return true;
  }

  private onPointerDown = (ev: PointerEvent): void => {
    if (ev.button !== 0 || this.mode.tag === "drag") return;
    const handle = this.deps.gizmo.pick(
      ev.clientX,
      ev.clientY,
      this.deps.camera(),
      this.deps.canvasRect(),
    );
    if (handle) {
      const uid = this.deps.selection()[0];
      if (uid) {
        const started =
          handle.kind === "rotate"
            ? this.beginDrag(uid, "rotate", ev, null)
            : this.beginDrag(uid, "scale", ev, null, handle.corner);
        if (started) return;
      }
    }

    const panel = this.deps.panelAt(ev.clientX, ev.clientY);
    const hitUid = panel ? this.deps.workpieces.pick(panel) : null;
    if (hitUid && !ev.shiftKey) {
      if (!this.deps.selection().includes(hitUid)) {
        this.deps.callbacks.onDragSelect(hitUid);
      }
      if (this.beginDrag(hitUid, "move", ev, hitUid)) return;
    }
    this.mode = {
      tag: "press",
      downAt: { x: ev.clientX, y: ev.clientY },
      hitUid,
    };
  };

  private onPointerMove = (ev: PointerEvent): void => {
    if (this.mode.tag === "drag") {
      if (
        Math.hypot(
          ev.clientX - this.mode.downAt.x,
          ev.clientY - this.mode.downAt.y,
        ) > DRAG_SLOP_PX
      ) {
        this.mode.moved = true;
      }
      const { uid, others, controller, startTransforms } = this.mode;
      const panel = this.deps.panelAt(ev.clientX, ev.clientY);
      if (!panel) return;
      const m = this.deps.worldToPanel();
      const world = panelToWorld(m, panel);
      const current = controller.update({
        pointer: world,
        snap: ev.shiftKey,
      });
      const start = controller.start;
      if (!current || !start) return;
      this.deps.workpieces.setPreview(uid, start.transform, current, m);
      if (controller.kind === "move") {
        const dx = current.pos.x - start.transform.pos.x;
        const dy = current.pos.y - start.transform.pos.y;
        for (const id of others) {
          const s = startTransforms.get(id);
          if (!s) continue;
          this.deps.workpieces.setPreview(
            id,
            s,
            {
              pos: { x: s.pos.x + dx, y: s.pos.y + dy },
              size: { ...s.size },
              angleDeg: s.angleDeg,
            },
            m,
          );
        }
      }
      this.deps.gizmo.show(current, m);
      this.deps.callbacks.onPreview(uid, toItemTransform(current));
      return;
    }

    if (ev.buttons !== 0) {
      this.setHover(null);
      return;
    }
    const panel = this.deps.panelAt(ev.clientX, ev.clientY);
    const uid = panel ? this.deps.workpieces.pick(panel) : null;
    this.setHover(uid);
  };

  private onPointerUp = (ev: PointerEvent): void => {
    if (ev.button !== 0) return;
    if (this.mode.tag === "drag") {
      const { uid, others, controller, startTransforms, moved, hitUid } =
        this.mode;
      const current = controller.current;
      const startTransform = controller.start?.transform ?? null;
      this.mode = { tag: "idle" };
      this.deps.controls.enabled = true;
      this.deps.callbacks.onPreview(null, null);

      const committed =
        moved &&
        current !== null &&
        startTransform !== null &&
        this.transformChanged(startTransform, current);
      if (!committed) {
        this.deps.workpieces.clearAllPreviews();
        if (startTransform) {
          this.deps.gizmo.show(
            startTransform,
            this.deps.worldToPanel(),
          );
        }
        this.deps.callbacks.onClick(hitUid, ev);
        return;
      }

      const commits: CommitItem[] = [
        { uid, transform: toItemTransform(current) },
      ];
      if (controller.kind === "move" && startTransform) {
        const dx = current.pos.x - startTransform.pos.x;
        const dy = current.pos.y - startTransform.pos.y;
        for (const id of others) {
          const s = startTransforms.get(id);
          if (!s) continue;
          commits.push({
            uid: id,
            transform: toItemTransform({
              pos: { x: s.pos.x + dx, y: s.pos.y + dy },
              size: { ...s.size },
              angleDeg: s.angleDeg,
            }),
          });
        }
      }
      this.deps.callbacks.onCommit(commits);
      return;
    }

    if (this.mode.tag === "press") {
      const { downAt, hitUid } = this.mode;
      this.mode = { tag: "idle" };
      const dist = Math.hypot(
        ev.clientX - downAt.x,
        ev.clientY - downAt.y,
      );
      if (dist <= CLICK_SLOP_PX) {
        this.deps.callbacks.onClick(hitUid, ev);
      }
    }
  };

  private onKeyDown = (ev: KeyboardEvent): void => {
    if (ev.key === "Escape" && this.mode.tag === "drag") {
      ev.preventDefault();
      this.cancelDrag();
    }
  };

  private transformChanged(a: Xform, b: Xform): boolean {
    return (
      Math.abs(a.pos.x - b.pos.x) > 1e-6 ||
      Math.abs(a.pos.y - b.pos.y) > 1e-6 ||
      Math.abs(a.size.w - b.size.w) > 1e-6 ||
      Math.abs(a.size.h - b.size.h) > 1e-6 ||
      Math.abs(a.angleDeg - b.angleDeg) > 1e-6
    );
  }

  private setHover(uid: string | null): void {
    if (uid === this.hoverUid) return;
    this.hoverUid = uid;
    this.deps.workpieces.setHover(uid);
  }
}
