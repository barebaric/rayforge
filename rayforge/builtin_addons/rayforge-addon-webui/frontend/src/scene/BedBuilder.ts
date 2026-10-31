import * as THREE from "three";
import type { Vec2, ViewportContract } from "../api/types";
import {
  AXIS_X,
  AXIS_Y,
  BED_EDGE,
  BED_FILL,
  GRID_MAJOR,
  GRID_MINOR,
  NOGO_ZONE,
  SOFT_LIMIT_COLOR,
  WORKAREA_EDGE,
  WORKAREA_FILL,
  WORK_ORIGIN,
} from "./colors";
import { panelToScene } from "./coords";
import { makeTextSprite } from "./textSprite";

export type GridDensity = "fine" | "coarse";

const MINOR_STEP_MM = 10;
const MAJOR_STEP_MM = 50;

/**
 * Scene-space direction of the displayed +X/+Y axis arrows at the
 * origin corner. The contract's x_axis_right/y_axis_down flags say
 * which way the native axis grows on screen: when the displayed origin
 * sits on the right, native X grows toward the left (panel -x); when
 * it sits at the top, native Y grows downward (panel -y), i.e. toward
 * the viewer (+scene Z).
 */
export function axisArrowDirections(flags: {
  x_axis_right: boolean;
  y_axis_down: boolean;
}): { x: THREE.Vector3; y: THREE.Vector3 } {
  return {
    x: new THREE.Vector3(flags.x_axis_right ? -1 : 1, 0, 0),
    y: new THREE.Vector3(0, 0, flags.y_axis_down ? 1 : -1),
  };
}

function lineMaterial(color: number, opacity = 1): THREE.LineBasicMaterial {
  return new THREE.LineBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity,
  });
}

type LineMat = THREE.LineBasicMaterial | THREE.LineDashedMaterial;

function rectLine(
  xMin: number,
  yMin: number,
  xMax: number,
  yMax: number,
  y: number,
  material: LineMat,
): THREE.LineLoop {
  const pts = [
    panelToScene({ x: xMin, y: yMin }, y),
    panelToScene({ x: xMax, y: yMin }, y),
    panelToScene({ x: xMax, y: yMax }, y),
    panelToScene({ x: xMin, y: yMax }, y),
  ].map((p) => new THREE.Vector3(p.x, p.y, p.z));
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.LineLoop(geo, material);
}

function rectFill(
  xMin: number,
  yMin: number,
  xMax: number,
  yMax: number,
  y: number,
  color: number,
  opacity: number,
): THREE.Mesh {
  const w = xMax - xMin;
  const h = yMax - yMin;
  const geo = new THREE.PlaneGeometry(w, h);
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshBasicMaterial({
      color,
      transparent: opacity < 1,
      opacity,
      depthWrite: opacity >= 1,
    }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set((xMin + xMax) / 2, y, -(yMin + yMax) / 2);
  return mesh;
}

function gridSegments(
  width: number,
  height: number,
  step: number,
  y: number,
): THREE.Vector3[] {
  const pts: THREE.Vector3[] = [];
  for (let x = 0; x <= width + 1e-6; x += step) {
    pts.push(
      new THREE.Vector3(x, y, 0),
      new THREE.Vector3(x, y, -height),
    );
  }
  for (let yP = 0; yP <= height + 1e-6; yP += step) {
    pts.push(
      new THREE.Vector3(0, y, -yP),
      new THREE.Vector3(width, y, -yP),
    );
  }
  return pts;
}

const NOGO_ZONE_Y = 0.1;

/** Scene-space outline points of one closed panel-space polygon. */
export function zoneOutlinePoints(
  polygon: number[],
  y = NOGO_ZONE_Y,
): THREE.Vector3[] {
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i + 1 < polygon.length; i += 2) {
    const p = panelToScene({ x: polygon[i]!, y: polygon[i + 1]! }, y);
    pts.push(new THREE.Vector3(p.x, p.y, p.z));
  }
  return pts;
}

function zoneFill(
  polygon: number[],
  y = NOGO_ZONE_Y,
): THREE.Mesh | null {
  if (polygon.length < 6) return null;
  const shape = new THREE.Shape();
  shape.moveTo(polygon[0]!, polygon[1]!);
  for (let i = 2; i + 1 < polygon.length; i += 2) {
    shape.lineTo(polygon[i]!, polygon[i + 1]!);
  }
  const mesh = new THREE.Mesh(
    new THREE.ShapeGeometry(shape),
    new THREE.MeshBasicMaterial({
      color: NOGO_ZONE,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = y - 0.01;
  return mesh;
}

/**
 * Builds the static bed furniture from the viewport contract: bed
 * outline, grid, workarea, axis arrows and the work-origin marker.
 */
export class BedBuilder {
  readonly group = new THREE.Group();
  private gridGroup = new THREE.Group();
  private nogoGroup: THREE.Group | null = null;
  private nogoVisible = true;
  private density: GridDensity | null = null;
  private vp: ViewportContract | null = null;

  constructor() {
    this.group.add(this.gridGroup);
  }

  build(vp: ViewportContract): void {
    this.vp = vp;
    this.nogoGroup = null;
    this.disposeChildren(this.group);
    this.gridGroup = new THREE.Group();
    this.group.add(this.gridGroup);

    const width = vp.extents.width_mm;
    const height = vp.extents.height_mm;

    this.group.add(rectFill(0, 0, width, height, -0.02, BED_FILL, 1));
    this.group.add(
      rectLine(0, 0, width, height, 0.02, lineMaterial(BED_EDGE)),
    );

    const wa = vp.workarea;
    this.group.add(
      rectFill(wa.x_min, wa.y_min, wa.x_max, wa.y_max, 0.06, WORKAREA_FILL, 0.6),
    );
    this.group.add(
      rectLine(wa.x_min, wa.y_min, wa.x_max, wa.y_max, 0.08, lineMaterial(WORKAREA_EDGE)),
    );

    if (vp.soft_limits_panel) {
      const sl = vp.soft_limits_panel;
      const mat = new THREE.LineDashedMaterial({
        color: SOFT_LIMIT_COLOR,
        dashSize: 4,
        gapSize: 4,
        transparent: true,
        opacity: 0.8,
      });
      const loop = rectLine(
        sl.x,
        sl.y,
        sl.x + sl.w,
        sl.y + sl.h,
        0.09,
        mat,
      );
      loop.computeLineDistances();
      this.group.add(loop);
    }

    this.buildAxisArrows(vp);
    this.buildWorkOriginMarker(vp.work_origin_panel);
    this.buildNogoZones(vp.nogo_zones);
    this.setDensity(this.density ?? "fine");
  }

  /** Shows or hides the no-go zone overlays built by build(). */
  setNogoZonesVisible(visible: boolean): void {
    this.nogoVisible = visible;
    if (this.nogoGroup) this.nogoGroup.visible = visible;
  }

  private buildNogoZones(zones: number[][]): void {
    if (!zones || zones.length === 0) return;
    const group = new THREE.Group();
    for (const polygon of zones) {
      if (polygon.length < 6) continue;
      const fill = zoneFill(polygon);
      if (fill) group.add(fill);
      group.add(
        new THREE.LineLoop(
          new THREE.BufferGeometry().setFromPoints(
            zoneOutlinePoints(polygon),
          ),
          lineMaterial(NOGO_ZONE, 0.85),
        ),
      );
    }
    group.visible = this.nogoVisible;
    this.nogoGroup = group;
    this.group.add(group);
  }

  /** Rebuilds the grid when the desired density changed. */
  setDensity(density: GridDensity): void {
    if (density === this.density && this.gridGroup.children.length > 0) return;
    this.density = density;
    this.disposeChildren(this.gridGroup);
    const vp = this.vp;
    if (!vp) return;
    const width = vp.extents.width_mm;
    const height = vp.extents.height_mm;

    const majorPts = gridSegments(width, height, MAJOR_STEP_MM, 0.04);
    const majorGeo = new THREE.BufferGeometry().setFromPoints(majorPts);
    this.gridGroup.add(new THREE.LineSegments(majorGeo, lineMaterial(GRID_MAJOR)));

    if (density === "fine") {
      const minorPts = gridSegments(width, height, MINOR_STEP_MM, 0.03);
      const minorGeo = new THREE.BufferGeometry().setFromPoints(minorPts);
      this.gridGroup.add(
        new THREE.LineSegments(minorGeo, lineMaterial(GRID_MINOR)),
      );
    }
  }

  private buildAxisArrows(vp: ViewportContract): void {
    const origin: Vec2 = vp.workarea_origin_panel;
    const spanX = Math.abs(vp.workarea.x_max - vp.workarea.x_min);
    const spanY = Math.abs(vp.workarea.y_max - vp.workarea.y_min);
    const len = Math.max(15, Math.min(spanX, spanY) * 0.25);
    const headLen = len * 0.22;
    const headW = headLen * 0.5;
    const dirs = axisArrowDirections(vp);

    const xArrow = new THREE.ArrowHelper(
      dirs.x,
      new THREE.Vector3(origin.x, 0.12, -origin.y),
      len,
      AXIS_X,
      headLen,
      headW,
    );
    this.group.add(xArrow);
    const xLabel = makeTextSprite("X", "#e25555", len * 0.28);
    const xp = panelToScene(
      {
        x: origin.x + dirs.x.x * len * 1.25,
        y: origin.y,
      },
      0.12,
    );
    xLabel.position.set(xp.x, xp.y + len * 0.1, xp.z);
    this.group.add(xLabel);

    const yArrow = new THREE.ArrowHelper(
      dirs.y,
      new THREE.Vector3(origin.x, 0.12, -origin.y),
      len,
      AXIS_Y,
      headLen,
      headW,
    );
    this.group.add(yArrow);
    const yLabel = makeTextSprite("Y", "#3ddc84", len * 0.28);
    const yp = panelToScene(
      {
        x: origin.x,
        y: origin.y - dirs.y.z * len * 1.25,
      },
      0.12,
    );
    yLabel.position.set(yp.x, yp.y + len * 0.1, yp.z);
    this.group.add(yLabel);
  }

  private buildWorkOriginMarker(origin: Vec2): void {
    const g = new THREE.Group();
    const size = 5;
    const mat = lineMaterial(WORK_ORIGIN);
    const mkLine = (a: Vec2, b: Vec2) => {
      const pts = [
        panelToScene(a, 0.14),
        panelToScene(b, 0.14),
      ].map((p) => new THREE.Vector3(p.x, p.y, p.z));
      return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
    };
    g.add(
      mkLine({ x: origin.x - size, y: origin.y }, { x: origin.x + size, y: origin.y }),
    );
    g.add(
      mkLine({ x: origin.x, y: origin.y - size }, { x: origin.x, y: origin.y + size }),
    );
    const ringPts: THREE.Vector3[] = [];
    for (let i = 0; i <= 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      ringPts.push(
        new THREE.Vector3(
          origin.x + Math.cos(a) * size,
          0.14,
          -(origin.y + Math.sin(a) * size),
        ),
      );
    }
    g.add(
      new THREE.Line(new THREE.BufferGeometry().setFromPoints(ringPts), mat),
    );
    const label = makeTextSprite("W", "#e6e8ec", 5);
    const lp = panelToScene({ x: origin.x + size + 4, y: origin.y + size + 2 }, 0.14);
    label.position.set(lp.x, lp.y + 3, lp.z);
    g.add(label);
    this.group.add(g);
  }

  private disposeChildren(parent: THREE.Object3D): void {
    const children = [...parent.children];
    for (const child of children) {
      parent.remove(child);
      child.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = (mesh as unknown as { material?: THREE.Material }).material;
        if (mat) mat.dispose();
      });
    }
  }
}
