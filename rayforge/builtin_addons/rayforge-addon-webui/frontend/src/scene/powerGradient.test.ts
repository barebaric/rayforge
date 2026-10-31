import { describe, expect, it } from "vitest";
import {
  POWER_HIGH,
  POWER_LOW,
  POWER_MID,
  powerToColor,
  powerToColorHex,
} from "./powerGradient";

describe("powerToColor", () => {
  it("returns the blue stop at power 0", () => {
    expect(powerToColor(0)).toEqual(POWER_LOW);
  });

  it("returns the red stop at power 1", () => {
    expect(powerToColor(1)).toEqual(POWER_HIGH);
  });

  it("returns the yellow-green stop at the midpoint", () => {
    expect(powerToColor(0.5)).toEqual(POWER_MID);
  });

  it("interpolates within each half of the gradient", () => {
    const quarter = powerToColor(0.25);
    expect(quarter.r).toBeCloseTo((POWER_LOW.r + POWER_MID.r) / 2);
    expect(quarter.g).toBeCloseTo((POWER_LOW.g + POWER_MID.g) / 2);
    expect(quarter.b).toBeCloseTo((POWER_LOW.b + POWER_MID.b) / 2);

    const threeQuarters = powerToColor(0.75);
    expect(threeQuarters.r).toBeCloseTo((POWER_MID.r + POWER_HIGH.r) / 2);
  });

  it("clamps out-of-range powers", () => {
    expect(powerToColor(-2)).toEqual(POWER_LOW);
    expect(powerToColor(2)).toEqual(POWER_HIGH);
  });

  it("maps to hex numbers usable as three.js colors", () => {
    expect(powerToColorHex(0)).toBe(
      (Math.round(POWER_LOW.r * 255) << 16) |
        (Math.round(POWER_LOW.g * 255) << 8) |
        Math.round(POWER_LOW.b * 255),
    );
    expect(powerToColorHex(1)).toBe(
      (Math.round(POWER_HIGH.r * 255) << 16) |
        (Math.round(POWER_HIGH.g * 255) << 8) |
        Math.round(POWER_HIGH.b * 255),
    );
  });
});
