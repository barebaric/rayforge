import * as THREE from "three";
import type { Affine2D, OpsSegment } from "../api/types";
import { panelToScene, worldPolylineToPanel } from "./coords";
import { powerToColor } from "./powerGradient";

/** Height above the bed so paths stay visible over workpiece fills. */
const OPS_Y = 0.4;
const TRAVEL_COLOR = 0x9aa0aa;
const TRAVEL_OPACITY = 0.32;
const CUT_OPACITY = 0.95;

export interface OpsPreviewData {
  generationId: string;
  segments: OpsSegment[];
}

interface BuildResult {
  cutLines: THREE.LineSegments;
  travelLines: THREE.LineSegments;
}

/**
 * Renders the ops toolpath preview: cutting segments as one
 * vertex-colored line strip set (color = power gradient) and travel
 * moves as a faint dashed overlay. All segments merge into two
 * geometries so ~10k points stay at two draw calls.
 */
export class OpsPreviewLayer {
  readonly group = new THREE.Group();
  private data: OpsPreviewData | null = null;
  private built: BuildResult | null = null;
  private travelVisible = true;
  private readonly getWorldToPanel: () => Affine2D;

  constructor(getWorldToPanel: () => Affine2D) {
    this.getWorldToPanel = getWorldToPanel;
    this.group.visible = false;
  }

  /** Replaces the preview data; rebuilds unless the generation matches. */
  setData(data: OpsPreviewData | null): void {
    if (this.data && data && this.data.generationId === data.generationId) {
      return;
    }
    this.data = data;
    this.rebuild();
  }

  /** Re-projects the cached world segments after a viewport change. */
  viewportChanged(): void {
    if (this.data) this.rebuild();
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible && this.data !== null;
  }

  /** Shows or hides the dashed travel moves (cut lines stay). */
  setTravelVisible(visible: boolean): void {
    this.travelVisible = visible;
    if (this.built) this.built.travelLines.visible = visible;
  }

  private rebuild(): void {
    this.disposeBuild();
    this.group.visible = false;
    if (!this.data || this.data.segments.length === 0) return;
    this.built = this.build(this.data.segments);
    this.built.travelLines.visible = this.travelVisible;
    this.group.add(this.built.cutLines, this.built.travelLines);
    this.group.visible = true;
  }

  private build(segments: OpsSegment[]): BuildResult {
    const matrix = this.getWorldToPanel();
    const cutPositions: number[] = [];
    const cutColors: number[] = [];
    const travelPositions: number[] = [];

    for (const segment of segments) {
      const panel = worldPolylineToPanel(matrix, segment.points);
      if (segment.cutting) {
        const color = powerToColor(segment.power);
        for (let i = 2; i + 1 < panel.length; i += 2) {
          const a = panelToScene({ x: panel[i - 2]!, y: panel[i - 1]! }, OPS_Y);
          const b = panelToScene({ x: panel[i]!, y: panel[i + 1]! }, OPS_Y);
          cutPositions.push(a.x, a.y, a.z, b.x, b.y, b.z);
          cutColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
        }
      } else {
        for (let i = 2; i + 1 < panel.length; i += 2) {
          const a = panelToScene({ x: panel[i - 2]!, y: panel[i - 1]! }, OPS_Y);
          const b = panelToScene({ x: panel[i]!, y: panel[i + 1]! }, OPS_Y);
          travelPositions.push(a.x, a.y, a.z, b.x, b.y, b.z);
        }
      }
    }

    const cutGeometry = new THREE.BufferGeometry();
    cutGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(cutPositions, 3),
    );
    cutGeometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(cutColors, 3),
    );
    const cutLines = new THREE.LineSegments(
      cutGeometry,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: CUT_OPACITY,
        depthWrite: false,
      }),
    );
    cutLines.renderOrder = 3;

    const travelGeometry = new THREE.BufferGeometry();
    travelGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(travelPositions, 3),
    );
    const travelLines = new THREE.LineSegments(
      travelGeometry,
      new THREE.LineDashedMaterial({
        color: TRAVEL_COLOR,
        transparent: true,
        opacity: TRAVEL_OPACITY,
        dashSize: 1.6,
        gapSize: 1.6,
        depthWrite: false,
      }),
    );
    travelLines.computeLineDistances();
    travelLines.renderOrder = 2;

    return { cutLines, travelLines };
  }

  private disposeBuild(): void {
    if (!this.built) return;
    for (const lines of [this.built.cutLines, this.built.travelLines]) {
      this.group.remove(lines);
      lines.geometry.dispose();
      (lines.material as THREE.Material).dispose();
    }
    this.built = null;
  }

  dispose(): void {
    this.data = null;
    this.disposeBuild();
  }
}
