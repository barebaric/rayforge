import * as THREE from "three";
import {
  cameraBasisFromQuaternion,
  pickGizmo,
  projectTips,
  TIPS,
  tipDirection,
  type GizmoHit,
  type PickOptions,
  type TipId,
} from "./gizmoNav";

const VIEWPORT_PX = 64;
const HALF_EXTENT = 2.2;
const WORLD_PER_PX = (HALF_EXTENT * 2) / VIEWPORT_PX;
const BADGE_SIZE_PX = 46;
const CROSS_RADIUS_PX = 13;
const BALL_SIZE_PX = 14;
const NEGATIVE_BALL_PX = 9;
const CENTER_DOT_PX = 3;
const HOME_CHIP_PX = 16;
const CENTER_RADIUS_PX = 5;
const TIP_RADIUS_PX = 10;
const HOME_RADIUS_PX = 10;
const DEFAULT_OFFSET_X = 22;
const DEFAULT_OFFSET_Y = 46;

const AXIS_COLORS: Record<"X" | "Y" | "Z", number> = {
  X: 0xe5484d,
  Y: 0x4d9fff,
  Z: 0x46a758,
};

function toVector3(d: { x: number; y: number; z: number }): THREE.Vector3 {
  return new THREE.Vector3(d.x, d.y, d.z);
}

function makeBadgeTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const mid = size / 2;
  ctx.beginPath();
  ctx.arc(mid, mid, mid - 8, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(30, 32, 38, 0.92)";
  ctx.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function makeBallTexture(
  axis: string,
  colorHex: string,
  dimmed: boolean,
): THREE.CanvasTexture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const mid = size / 2;
  ctx.beginPath();
  ctx.arc(mid, mid, 27, 0, Math.PI * 2);
  ctx.fillStyle = dimmed
    ? `rgba(74, 78, 86, 0.9)`
    : colorHex;
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0, 0, 0, 0.35)";
  ctx.stroke();
  if (!dimmed) {
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 36px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(axis, mid, mid + 2);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function makeCenterDotTexture(): THREE.CanvasTexture {
  const size = 32;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const mid = size / 2;
  ctx.beginPath();
  ctx.arc(mid, mid, 9, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(154, 160, 170, 0.95)";
  ctx.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function makeHouseTexture(): THREE.CanvasTexture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const s = size * 0.62;
  const o = (size - s) / 2;
  ctx.strokeStyle = "#9aa0aa";
  ctx.fillStyle = "#9aa0aa";
  ctx.lineWidth = 4;
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(o, s * 0.5 + o * 0.4);
  ctx.lineTo(size / 2, o * 0.7);
  ctx.lineTo(size - o, s * 0.5 + o * 0.4);
  ctx.stroke();
  ctx.beginPath();
  ctx.rect(size / 2 - s * 0.22, s * 0.52 + o * 0.3, s * 0.44, s * 0.4);
  ctx.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function makeSprite(
  texture: THREE.CanvasTexture,
  sizePx: number,
): THREE.Sprite {
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
      depthTest: false,
    }),
  );
  sprite.scale.setScalar(sizePx * WORLD_PER_PX);
  return sprite;
}

interface TipVisuals {
  tip: TipId;
  ball: THREE.Sprite;
  baseColor: THREE.Color;
}

/**
 * Navigation cross in the top-right of the viewport: a compact
 * rounded badge holding three orthogonal lines with labelled balls
 * that rotate with the main camera. Labels follow machine convention:
 * X along scene X, Y along scene -Z (bed depth), Z up. Clicking a
 * ball snaps the camera to that axis view (handled by SceneManager
 * via hitTest), the house chip and the center reset to iso.
 */
export class GizmoHelper {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.OrthographicCamera(
    -HALF_EXTENT,
    HALF_EXTENT,
    HALF_EXTENT,
    -HALF_EXTENT,
    0.1,
    20,
  );
  private readonly tipVisuals: TipVisuals[] = [];
  private readonly house: THREE.Sprite;
  private offsetX = DEFAULT_OFFSET_X;
  private offsetY = DEFAULT_OFFSET_Y;
  private hover: TipId | "home" | null = null;

  constructor() {
    this.camera.position.set(0, 0, 5);

    const badge = makeSprite(makeBadgeTexture(), BADGE_SIZE_PX + 4);
    badge.renderOrder = -2;
    this.scene.add(badge);

    for (const tip of TIPS) {
      this.scene.add(...this.buildTip(tip));
    }

    const dot = makeSprite(makeCenterDotTexture(), CENTER_DOT_PX);
    dot.renderOrder = 5;
    this.scene.add(dot);

    this.house = makeSprite(makeHouseTexture(), HOME_CHIP_PX);
    this.house.renderOrder = 6;
    this.scene.add(this.house);
  }

  private placeBadgeCorner(sprite: THREE.Sprite): void {
    const badgeHalf = (BADGE_SIZE_PX / 2) * WORLD_PER_PX;
    const inset = 1 * WORLD_PER_PX;
    sprite.position.set(-badgeHalf + inset, badgeHalf - inset, 0);
  }

  private buildTip(tip: TipId): THREE.Object3D[] {
    const axis = tip.replace("-", "") as "X" | "Y" | "Z";
    const negative = tip.startsWith("-");
    const baseColor = new THREE.Color(AXIS_COLORS[axis]);
    if (negative) baseColor.multiplyScalar(0.55);
    const dir = toVector3(tipDirection(tip));

    const material = new THREE.MeshBasicMaterial({
      color: baseColor,
      transparent: true,
      opacity: negative ? 0.35 : 0.9,
    });
    const shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.05, CROSS_RADIUS_PX * WORLD_PER_PX, 8),
      material,
    );
    shaft.position.copy(dir).multiplyScalar(
      (CROSS_RADIUS_PX * WORLD_PER_PX) / 2,
    );
    shaft.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      dir.clone(),
    );
    this.scene.add(shaft);

    const ball = makeSprite(
      makeBallTexture(axis, `#${baseColor.getHexString()}`, negative),
      negative ? NEGATIVE_BALL_PX : BALL_SIZE_PX,
    );
    ball.userData.baseScale = ball.scale.x;
    ball.position.copy(dir).multiplyScalar(CROSS_RADIUS_PX * WORLD_PER_PX);
    ball.renderOrder = negative ? 2 : 4;
    this.scene.add(ball);
    this.tipVisuals.push({ tip, ball, baseColor });

    return [shaft, ball];
  }

  /** Shifts the badge on screen; (0, 0) is the canvas top-right. */
  setScreenOffset(x: number, y: number): void {
    this.offsetX = x;
    this.offsetY = y;
  }

  /** Highlights one tip (ball brightens and grows) or the house. */
  setHover(hit: TipId | "home" | null): void {
    if (hit === this.hover) return;
    this.hover = hit;
    for (const visuals of this.tipVisuals) {
      const hovered = visuals.tip === hit;
      visuals.ball.scale.setScalar(
        (visuals.ball.userData.baseScale as number) * (hovered ? 1.25 : 1),
      );
    }
    this.house.scale.setScalar(
      HOME_CHIP_PX * WORLD_PER_PX * (hit === "home" ? 1.3 : 1),
    );
  }

  private geometry(width: number) {
    const half = VIEWPORT_PX / 2;
    return {
      centerX: width - this.offsetX - half,
      centerY: this.offsetY + half,
      radius: CROSS_RADIUS_PX,
    };
  }

  /**
   * Screen-space hit test in client coordinates; returns null when
   * the pointer is outside the badge so the event can pass through
   * to orbit and selection handling.
   */
  hitTest(
    clientX: number,
    clientY: number,
    mainCamera: THREE.Camera,
    rect: { left: number; top: number; width: number; height: number },
  ): GizmoHit | null {
    const localX = clientX - rect.left;
    const localY = clientY - rect.top;
    const geom = this.geometry(rect.width);
    const badgeHalf = BADGE_SIZE_PX / 2;
    const homeX = geom.centerX - badgeHalf + 5;
    const homeY = geom.centerY - badgeHalf + 5;
    if (Math.hypot(localX - homeX, localY - homeY) <= HOME_RADIUS_PX) {
      return "home";
    }
    const options: PickOptions = {
      ...geom,
      tipRadius: TIP_RADIUS_PX,
      centerRadius: CENTER_RADIUS_PX,
    };
    return pickGizmo(
      localX,
      localY,
      projectTips(cameraBasisFromQuaternion(mainCamera.quaternion), geom),
      options,
    );
  }

  render(renderer: THREE.WebGLRenderer, mainCamera: THREE.Camera): void {
    const size = renderer.getSize(new THREE.Vector2());
    const half = VIEWPORT_PX / 2;
    const geom = this.geometry(size.x);
    if (
      geom.centerX - half < 0 ||
      geom.centerX + half > size.x ||
      geom.centerY - half < 0 ||
      geom.centerY + half > size.y
    ) {
      return;
    }

    this.placeBadgeCorner(this.house);
    this.camera.quaternion.copy(mainCamera.quaternion);
    this.camera.position.set(0, 0, 5).applyQuaternion(this.camera.quaternion);

    const previousAutoClear = renderer.autoClear;
    renderer.autoClear = false;
    renderer.setScissorTest(true);
    renderer.setScissor(
      geom.centerX - half,
      size.y - geom.centerY - half,
      VIEWPORT_PX,
      VIEWPORT_PX,
    );
    renderer.setViewport(
      geom.centerX - half,
      size.y - geom.centerY - half,
      VIEWPORT_PX,
      VIEWPORT_PX,
    );
    renderer.clearDepth();
    renderer.render(this.scene, this.camera);
    renderer.autoClear = previousAutoClear;
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, size.x, size.y);
  }

  dispose(): void {
    this.scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose();
      }
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Sprite) {
        const material = obj.material as THREE.Material & {
          map?: THREE.Texture;
        };
        material.map?.dispose();
        material.dispose();
      }
    });
  }
}
