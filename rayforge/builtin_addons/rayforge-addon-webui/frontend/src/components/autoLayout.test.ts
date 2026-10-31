import { describe, expect, it } from "vitest";
import { decideAutoLayout } from "./autoLayout";

const PIXEL = { id: "pixel-perfect", label: "Auto Layout" };
const NESTING = { id: "nesting", label: "Nesting layout" };

describe("decideAutoLayout", () => {
  it("reports none when no strategy is registered", () => {
    expect(decideAutoLayout([])).toEqual({ kind: "none" });
  });

  it("runs directly when exactly one strategy exists", () => {
    expect(decideAutoLayout([PIXEL])).toEqual({
      kind: "run",
      layout: PIXEL,
    });
  });

  it("opens the picker for several strategies", () => {
    const decision = decideAutoLayout([PIXEL, NESTING]);
    expect(decision).toEqual({ kind: "picker", layouts: [PIXEL, NESTING] });
  });
});
