/**
 * Power-to-color gradient for toolpath rendering.
 *
 * Documented stops (linear RGB, 0..1):
 *   power 0.0 -> blue   #2e8cff
 *   power 0.5 -> lime   #b8eb33
 *   power 1.0 -> red    #ff2619
 *
 * Powers outside [0, 1] are clamped.
 */
export interface Rgb {
  r: number;
  g: number;
  b: number;
}

export const POWER_LOW: Rgb = { r: 0.18, g: 0.55, b: 1.0 };
export const POWER_MID: Rgb = { r: 0.72, g: 0.92, b: 0.2 };
export const POWER_HIGH: Rgb = { r: 1.0, g: 0.15, b: 0.1 };

function clamp01(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

function lerpRgb(a: Rgb, b: Rgb, t: number): Rgb {
  if (t <= 0) return { ...a };
  if (t >= 1) return { ...b };
  return {
    r: a.r + (b.r - a.r) * t,
    g: a.g + (b.g - a.g) * t,
    b: a.b + (b.b - a.b) * t,
  };
}

export function powerToColor(power: number): Rgb {
  const p = clamp01(power);
  return p <= 0.5
    ? lerpRgb(POWER_LOW, POWER_MID, p * 2)
    : lerpRgb(POWER_MID, POWER_HIGH, (p - 0.5) * 2);
}

/** Same gradient as a 0xRRGGBB number for three.js Color inputs. */
export function powerToColorHex(power: number): number {
  const c = powerToColor(power);
  const r = Math.round(clamp01(c.r) * 255);
  const g = Math.round(clamp01(c.g) * 255);
  const b = Math.round(clamp01(c.b) * 255);
  return (r << 16) | (g << 8) | b;
}
