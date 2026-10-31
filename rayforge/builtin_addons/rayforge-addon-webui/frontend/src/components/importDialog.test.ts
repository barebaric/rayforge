import { describe, expect, it } from "vitest";
import { prefersTrace } from "./importDialog";

describe("prefersTrace", () => {
  it("traces bitmaps that cannot be read as vectors", () => {
    expect(prefersTrace(["BITMAP_TRACING"])).toBe(true);
  });

  it("passes through files the importer reads as vectors", () => {
    expect(
      prefersTrace([
        "BITMAP_TRACING",
        "COLOR_LAYERS",
        "DIRECT_VECTOR",
        "LAYER_SELECTION",
      ]),
    ).toBe(false);
  });

  it("does not trace when no features are reported", () => {
    expect(prefersTrace([])).toBe(false);
  });
});
