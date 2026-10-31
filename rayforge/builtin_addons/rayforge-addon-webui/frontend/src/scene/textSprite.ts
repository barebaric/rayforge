import * as THREE from "three";

/** Canvas-rendered text sprite used for axis and origin labels. */
export function makeTextSprite(
  text: string,
  color: string,
  heightMm = 6,
): THREE.Sprite {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  const fontSize = 48;
  const font = `600 ${fontSize}px system-ui, sans-serif`;
  ctx.font = font;
  const metrics = ctx.measureText(text);
  const pad = 10;
  canvas.width = Math.ceil(metrics.width + pad * 2);
  canvas.height = Math.ceil(fontSize * 1.4);

  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });
  const aspect = canvas.width / canvas.height;
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(heightMm * aspect, heightMm, 1);
  return sprite;
}
