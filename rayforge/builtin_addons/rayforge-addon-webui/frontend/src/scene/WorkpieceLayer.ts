import * as THREE from "three";
import type { ApiBackend } from "../api/client";
import type { DocTree, StockInfo, WorkpieceInfo } from "../api/types";
import { geometryStale } from "../stores/docStore";
import { HOVER_LIFT, SELECT_COLOR } from "./colors";
import {
  pointInPolygon,
  sameAffine,
  worldPolylineToPanel,
} from "./coords";
import { previewAffine, type Xform } from "./interaction/TransformController";

const FILL_Y = 0.2;
const TEXTURE_Y = 0.3;
const HIDDEN_COLOR = "#3a3e46";
const FILL_COLOR = 0x272e3a;
const SELECT_FILL_COLOR = 0x1d3a5e;
const STOCK_FILL_COLOR = 0x1d2026;
const STOCK_OUTLINE_COLOR = 0x4a505c;
const STOCK_FILL_Y = 0.1;
const STOCK_OUTLINE_Y = 0.13;

type Affine = [number, number, number, number, number, number];

interface CacheEntry {
  geometryRevision: number;
  transformRevision: number;
  visible: boolean;
  baseColor: THREE.Color;
  panelPolylines: number[][];
  holder: THREE.Group;
  fillMesh: THREE.Mesh | null;
  outline: THREE.Line;
  textureMesh: THREE.Mesh | null;
}

interface StockEntry {
  geometryRevision: number;
  transformRevision: number;
  panelPolylines: number[][];
  holder: THREE.Group;
}

function outlineObject(panelPolylines: number[][]): THREE.Line {
  const pts: THREE.Vector3[] = [];
  for (const poly of panelPolylines) {
    for (let i = 0; i + 1 < poly.length; i += 2) {
      pts.push(new THREE.Vector3(poly[i]!, FILL_Y + 0.02, -poly[i + 1]!));
    }
  }
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xffffff }));
}

function stockOutlineObject(panelPolylines: number[][]): THREE.Line {
  const pts: THREE.Vector3[] = [];
  for (const poly of panelPolylines) {
    for (let i = 0; i + 1 < poly.length; i += 2) {
      pts.push(new THREE.Vector3(poly[i]!, STOCK_OUTLINE_Y, -poly[i + 1]!));
    }
  }
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.Line(
    geo,
    new THREE.LineBasicMaterial({ color: STOCK_OUTLINE_COLOR }),
  );
}

function stockFillObject(poly: number[]): THREE.Mesh | null {
  if (poly.length < 6) return null;
  const shape = new THREE.Shape();
  shape.moveTo(poly[0]!, poly[1]!);
  for (let i = 2; i + 1 < poly.length; i += 2) {
    shape.lineTo(poly[i]!, poly[i + 1]!);
  }
  const mesh = new THREE.Mesh(
    new THREE.ShapeGeometry(shape),
    new THREE.MeshBasicMaterial({
      color: STOCK_FILL_COLOR,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
    }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = STOCK_FILL_Y;
  return mesh;
}

function fillObject(poly: number[]): THREE.Mesh | null {
  if (poly.length < 6) return null;
  const shape = new THREE.Shape();
  shape.moveTo(poly[0]!, poly[1]!);
  for (let i = 2; i + 1 < poly.length; i += 2) {
    shape.lineTo(poly[i]!, poly[i + 1]!);
  }
  const mesh = new THREE.Mesh(
    new THREE.ShapeGeometry(shape),
    new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
    }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = FILL_Y;
  return mesh;
}

function boundsOf(polylines: number[][]): {
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
} {
  let xMin = Infinity;
  let yMin = Infinity;
  let xMax = -Infinity;
  let yMax = -Infinity;
  for (const poly of polylines) {
    for (let i = 0; i + 1 < poly.length; i += 2) {
      xMin = Math.min(xMin, poly[i]!);
      xMax = Math.max(xMax, poly[i]!);
      yMin = Math.min(yMin, poly[i + 1]!);
      yMax = Math.max(yMax, poly[i + 1]!);
    }
  }
  return { xMin, yMin, xMax, yMax };
}

/**
 * Owns the workpiece objects. Geometry is fetched per workpiece and
 * cached by geometry_revision; world polylines are mapped through the
 * contract's world_to_panel matrix onto the bed plane. Each workpiece
 * lives in a holder group so drag previews can transform it in place.
 */
export class WorkpieceLayer {
  readonly group = new THREE.Group();
  private readonly stockGroup = new THREE.Group();
  private readonly cache = new Map<string, CacheEntry>();
  private readonly stockCache = new Map<string, StockEntry>();
  private readonly selected = new Set<string>();
  private readonly hovered: { uid: string | null } = { uid: null };
  private readonly pending = new Set<string>();
  private lastTree: DocTree | null = null;
  private bakedMatrix: Affine | null = null;
  private imagesVisible = true;
  private stockVisible = true;

  constructor(
    private readonly api: ApiBackend,
    private readonly getWorldToPanel: () => Affine,
  ) {
    this.group.add(this.stockGroup);
  }

  /** Brings the layer in sync with the doc tree. */
  async sync(tree: DocTree): Promise<void> {
    this.lastTree = tree;
    const m = this.getWorldToPanel();
    this.bakedMatrix = m;
    const wanted = new Map<
      string,
      { wp: WorkpieceInfo; color: THREE.Color; visible: boolean }
    >();
    for (const layer of tree.layers) {
      const color = new THREE.Color(layer.visible ? layer.color : HIDDEN_COLOR);
      for (const wp of layer.workpieces) {
        wanted.set(wp.uid, { wp, color, visible: layer.visible });
      }
    }

    for (const uid of [...this.cache.keys()]) {
      if (!wanted.has(uid)) this.remove(uid);
    }

    const loads: Promise<void>[] = [];
    for (const [uid, { wp, color, visible }] of wanted) {
      const entry = this.cache.get(uid);
      if (entry && !this.entryStale(entry, wp)) {
        entry.baseColor.copy(color);
        entry.visible = visible;
        this.paint(uid);
        continue;
      }
      if (this.pending.has(uid)) continue;
      this.pending.add(uid);
      loads.push(
        this.loadWorkpiece(uid, wp, color, visible, m).finally(() =>
          this.pending.delete(uid),
        ),
      );
    }
    await Promise.all(loads);
    this.syncStock(tree.stock_items ?? [], m);
  }

  /**
   * Drops baked geometry when the contract's world_to_panel mapping
   * changed (e.g. the machine or its panel orientation changed while
   * the page is open) and reloads the last tree against the new one.
   */
  reproject(): Promise<void> {
    const m = this.getWorldToPanel();
    if (sameAffine(m, this.bakedMatrix)) return Promise.resolve();
    for (const uid of [...this.cache.keys()]) this.remove(uid);
    for (const uid of [...this.stockCache.keys()]) this.removeStock(uid);
    this.bakedMatrix = m;
    const tree = this.lastTree;
    if (!tree) return Promise.resolve();
    return this.sync(tree);
  }

  setSelection(uids: string[]): void {
    this.selected.clear();
    for (const uid of uids) this.selected.add(uid);
    for (const uid of this.cache.keys()) this.paint(uid);
  }

  setHover(uid: string | null): void {
    if (this.hovered.uid === uid) return;
    const previous = this.hovered.uid;
    this.hovered.uid = uid;
    if (previous) this.paint(previous);
    if (uid) this.paint(uid);
  }

  /** Shows or hides the raster texture planes (outlines stay visible). */
  setImageVisible(visible: boolean): void {
    this.imagesVisible = visible;
    for (const entry of this.cache.values()) {
      if (entry.textureMesh) entry.textureMesh.visible = visible;
    }
  }

  /** Current texture visibility, applied to newly loaded textures too. */
  get imageVisible(): boolean {
    return this.imagesVisible;
  }

  /** Shows or hides the stock material shapes. */
  setStockVisible(visible: boolean): void {
    this.stockVisible = visible;
    this.stockGroup.visible = visible;
  }

  /** True while stock shapes should be drawn. */
  get stockShown(): boolean {
    return this.stockVisible;
  }

  /** Hit test in panel space; returns the last-added matching uid. */
  pick(panel: { x: number; y: number }): string | null {
    let hit: string | null = null;
    for (const [uid, entry] of this.cache) {
      if (!entry.visible) continue;
      if (entry.panelPolylines.some((poly) => pointInPolygon(panel, poly))) {
        hit = uid;
      }
    }
    return hit;
  }

  /**
   * Live drag preview: applies the affine that maps geometry baked at
   * `start` onto `current`, without touching cached data.
   */
  setPreview(
    uid: string,
    start: Xform,
    current: Xform,
    worldToPanelMatrix: Affine,
  ): void {
    const entry = this.cache.get(uid);
    if (!entry) return;
    const { linear, translation } = previewAffine(
      start,
      current,
      worldToPanelMatrix,
    );
    entry.holder.matrix.set(
      linear.a,
      0,
      -linear.b,
      translation.x,
      0,
      1,
      0,
      0,
      -linear.d,
      0,
      linear.e,
      -translation.y,
      0,
      0,
      0,
      1,
    );
    entry.holder.matrixWorldNeedsUpdate = true;
  }

  clearPreview(uid: string): void {
    const entry = this.cache.get(uid);
    if (!entry) return;
    entry.holder.matrix.identity();
    entry.holder.matrixWorldNeedsUpdate = true;
  }

  clearAllPreviews(): void {
    for (const uid of this.cache.keys()) this.clearPreview(uid);
  }

  private entryStale(entry: CacheEntry, wp: WorkpieceInfo): boolean {
    return (
      geometryStale(entry.geometryRevision, wp) ||
      entry.transformRevision !== wp.transform_revision
    );
  }

  private async loadWorkpiece(
    uid: string,
    wp: WorkpieceInfo,
    color: THREE.Color,
    visible: boolean,
    m: Affine,
  ): Promise<void> {
    try {
      const geo = await this.api.getItemGeometry(uid);
      this.remove(uid);
      const panelPolylines = geo.polylines.map((poly) =>
        worldPolylineToPanel(m, poly),
      );
      const holder = new THREE.Group();
      holder.matrixAutoUpdate = false;
      const entry: CacheEntry = {
        geometryRevision: wp.geometry_revision,
        transformRevision: wp.transform_revision,
        visible,
        baseColor: color.clone(),
        panelPolylines,
        holder,
        fillMesh: fillObject(panelPolylines[0] ?? []),
        outline: outlineObject(panelPolylines),
        textureMesh: null,
      };
      this.cache.set(uid, entry);
      this.paint(uid);
      this.group.add(holder);
      if (entry.fillMesh) holder.add(entry.fillMesh);
      holder.add(entry.outline);
      const texUrl = await this.api.getItemTexture(uid, wp.geometry_revision);
      if (texUrl && this.cache.get(uid) === entry) {
        this.addTexture(entry, texUrl);
      }
    } catch {
      return;
    }
  }

  private addTexture(entry: CacheEntry, url: string): void {
    new THREE.TextureLoader().load(
      url,
      (texture) => {
        texture.colorSpace = THREE.SRGBColorSpace;
        const b = boundsOf(entry.panelPolylines);
        const mesh = new THREE.Mesh(
          new THREE.PlaneGeometry(b.xMax - b.xMin, b.yMax - b.yMin),
          new THREE.MeshBasicMaterial({ map: texture, transparent: true }),
        );
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.set(
          (b.xMin + b.xMax) / 2,
          TEXTURE_Y,
          -(b.yMin + b.yMax) / 2,
        );
        mesh.renderOrder = 1;
        mesh.visible = this.imagesVisible;
        entry.textureMesh = mesh;
        entry.holder.add(mesh);
      },
      undefined,
      () => undefined,
    );
  }

  private paint(uid: string): void {
    const entry = this.cache.get(uid);
    if (!entry) return;
    const isSelected = this.selected.has(uid);
    const isHovered = this.hovered.uid === uid && !isSelected;
    const color = isSelected
      ? new THREE.Color(SELECT_COLOR)
      : isHovered
        ? entry.baseColor.clone().lerp(new THREE.Color(0xffffff), HOVER_LIFT)
        : entry.baseColor;
    (entry.outline.material as THREE.LineBasicMaterial).color.copy(color);
    if (entry.fillMesh) {
      (entry.fillMesh.material as THREE.MeshBasicMaterial).color.setHex(
        isSelected ? SELECT_FILL_COLOR : FILL_COLOR,
      );
    }
  }

  private remove(uid: string): void {
    const entry = this.cache.get(uid);
    if (!entry) return;
    for (const obj of [entry.fillMesh, entry.outline, entry.textureMesh]) {
      if (!obj) continue;
      entry.holder.remove(obj);
      const mesh = obj as THREE.Mesh;
      mesh.geometry?.dispose();
      const mat = mesh.material as THREE.Material | undefined;
      if (Array.isArray(mat)) {
        mat.forEach((m) => m.dispose());
      } else {
        mat?.dispose();
      }
    }
    this.group.remove(entry.holder);
    this.cache.delete(uid);
  }

  private syncStock(stocks: StockInfo[], m: Affine): void {
    const wanted = new Map(stocks.map((stock) => [stock.uid, stock]));
    for (const uid of [...this.stockCache.keys()]) {
      if (!wanted.has(uid)) this.removeStock(uid);
    }
    for (const [uid, stock] of wanted) {
      const entry = this.stockCache.get(uid);
      if (
        entry &&
        entry.geometryRevision === stock.geometry_revision &&
        entry.transformRevision === stock.transform_revision
      ) {
        continue;
      }
      void this.loadStock(uid, stock, m);
    }
  }

  private async loadStock(
    uid: string,
    stock: StockInfo,
    m: Affine,
  ): Promise<void> {
    try {
      const geo = await this.api.getItemGeometry(uid);
      this.removeStock(uid);
      const panelPolylines = geo.polylines.map((poly) =>
        worldPolylineToPanel(m, poly),
      );
      const holder = new THREE.Group();
      holder.matrixAutoUpdate = false;
      const fill = stockFillObject(panelPolylines[0] ?? []);
      if (fill) holder.add(fill);
      holder.add(stockOutlineObject(panelPolylines));
      const entry: StockEntry = {
        geometryRevision: stock.geometry_revision,
        transformRevision: stock.transform_revision,
        panelPolylines,
        holder,
      };
      this.stockCache.set(uid, entry);
      this.stockGroup.add(holder);
    } catch {
      return;
    }
  }

  private removeStock(uid: string): void {
    const entry = this.stockCache.get(uid);
    if (!entry) return;
    for (const obj of [...entry.holder.children]) {
      entry.holder.remove(obj);
      const mesh = obj as THREE.Mesh;
      mesh.geometry?.dispose();
      const mat = mesh.material as THREE.Material | undefined;
      mat?.dispose();
    }
    this.stockGroup.remove(entry.holder);
    this.stockCache.delete(uid);
  }
}
