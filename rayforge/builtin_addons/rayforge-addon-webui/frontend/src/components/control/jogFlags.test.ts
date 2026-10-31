import { describe, expect, it } from "vitest";
import {
  jogPadLayout,
  jogTooltip,
  type AxisFlags,
  type PanelOrientation,
} from "./jogFlags";

const NATIVE_BOTTOM_LEFT: AxisFlags = {
  x_axis_right: false,
  y_axis_down: false,
  x_axis_negative: false,
  y_axis_negative: false,
};

const ROTATED_LEFT: AxisFlags = {
  x_axis_right: true,
  y_axis_down: false,
  x_axis_negative: false,
  y_axis_negative: false,
};

const ROTATED_RIGHT_BOTTOM_LEFT: AxisFlags = {
  x_axis_right: true,
  y_axis_down: true,
  x_axis_negative: false,
  y_axis_negative: false,
};

describe("jogPadLayout", () => {
  it("maps panel directions to screen arrows for a native bed", () => {
    const layout = jogPadLayout(NATIVE_BOTTOM_LEFT);
    expect(layout.right.direction).toBe("+x");
    expect(layout.left.direction).toBe("-x");
    expect(layout.up.direction).toBe("+y");
    expect(layout.down.direction).toBe("-y");
    expect(layout.left.towardOrigin).toBe(true);
    expect(layout.down.towardOrigin).toBe(true);
    expect(layout.right.towardOrigin).toBe(false);
    expect(layout.up.towardOrigin).toBe(false);
  });

  it("flips horizontal arrows when the origin sits on the right", () => {
    const layout = jogPadLayout(ROTATED_LEFT, "rotated_left");
    expect(layout.right.direction).toBe("-x");
    expect(layout.left.direction).toBe("+x");
    expect(layout.up.direction).toBe("+y");
    expect(layout.down.direction).toBe("-y");
    expect(layout.right.towardOrigin).toBe(true);
    expect(layout.left.towardOrigin).toBe(false);
    expect(layout.right.nativeAxis).toBe("y");
    expect(layout.up.nativeAxis).toBe("x");
  });

  it("flips vertical arrows when the origin sits at the top", () => {
    const layout = jogPadLayout(
      ROTATED_RIGHT_BOTTOM_LEFT,
      "rotated_right",
    );
    expect(layout.up.direction).toBe("-y");
    expect(layout.down.direction).toBe("+y");
    expect(layout.up.towardOrigin).toBe(true);
    expect(layout.down.towardOrigin).toBe(false);
    expect(layout.right.direction).toBe("-x");
    expect(layout.right.towardOrigin).toBe(true);
    expect(layout.right.nativeAxis).toBe("y");
    expect(layout.up.nativeAxis).toBe("x");
  });

  it("marks native reversal from the axis negative flags", () => {
    const layout = jogPadLayout({
      x_axis_right: false,
      y_axis_down: false,
      x_axis_negative: true,
      y_axis_negative: false,
    });
    expect(layout.right.nativeNegative).toBe(true);
    expect(layout.left.nativeNegative).toBe(false);
    expect(layout.up.nativeNegative).toBe(false);
    expect(layout.down.nativeNegative).toBe(true);
  });

  it("swaps reversal tracking to the native Y axis when rotated", () => {
    // rotated_left over a bottom-left origin + native reverse_y: the
    // displayed X axis carries the native Y reversal (x_axis_negative).
    const layout = jogPadLayout(
      {
        x_axis_right: true,
        y_axis_down: false,
        x_axis_negative: true,
        y_axis_negative: false,
      },
      "rotated_left",
    );
    expect(layout.right.direction).toBe("-x");
    expect(layout.right.nativeAxis).toBe("y");
    expect(layout.right.nativeNegative).toBe(true);
    expect(layout.left.nativeNegative).toBe(false);
    expect(layout.up.direction).toBe("+y");
    expect(layout.up.nativeAxis).toBe("x");
    expect(layout.up.nativeNegative).toBe(false);
    expect(layout.down.nativeNegative).toBe(true);
  });

  it("labels native jog signs for every orientation variant", () => {
    // Button -> native axis/sign, derived from MachinePanel.calculate_jog
    // across all origin/reversal/orientation combinations; the rows are
    // the generator variants (scripts/gen_oriented_configs.py).
    const cases: Array<{
      name: string;
      flags: AxisFlags;
      orientation: PanelOrientation;
      expected: { right: string; left: string; up: string; down: string };
    }> = [
      {
        name: "bottom_left",
        flags: {
          x_axis_right: false,
          y_axis_down: false,
          x_axis_negative: false,
          y_axis_negative: false,
        },
        orientation: "native",
        expected: { right: "X+", left: "X-", up: "Y+", down: "Y-" },
      },
      {
        name: "top_left",
        flags: {
          x_axis_right: false,
          y_axis_down: true,
          x_axis_negative: false,
          y_axis_negative: false,
        },
        orientation: "native",
        expected: { right: "X+", left: "X-", up: "Y+", down: "Y-" },
      },
      {
        name: "top_right",
        flags: {
          x_axis_right: true,
          y_axis_down: true,
          x_axis_negative: false,
          y_axis_negative: false,
        },
        orientation: "native",
        expected: { right: "X+", left: "X-", up: "Y+", down: "Y-" },
      },
      {
        name: "bottom_right",
        flags: {
          x_axis_right: true,
          y_axis_down: false,
          x_axis_negative: false,
          y_axis_negative: false,
        },
        orientation: "native",
        expected: { right: "X+", left: "X-", up: "Y+", down: "Y-" },
      },
      {
        name: "top_right_reverse_x",
        flags: {
          x_axis_right: true,
          y_axis_down: true,
          x_axis_negative: true,
          y_axis_negative: false,
        },
        orientation: "native",
        expected: { right: "X-", left: "X+", up: "Y+", down: "Y-" },
      },
      {
        name: "bottom_left_reverse_y",
        flags: {
          x_axis_right: false,
          y_axis_down: false,
          x_axis_negative: false,
          y_axis_negative: true,
        },
        orientation: "native",
        expected: { right: "X+", left: "X-", up: "Y-", down: "Y+" },
      },
      {
        name: "bottom_left_rotated_left",
        flags: {
          x_axis_right: true,
          y_axis_down: false,
          x_axis_negative: false,
          y_axis_negative: false,
        },
        orientation: "rotated_left",
        expected: { right: "Y+", left: "Y-", up: "X+", down: "X-" },
      },
      {
        name: "top_left_rotated_right",
        flags: {
          x_axis_right: true,
          y_axis_down: true,
          x_axis_negative: false,
          y_axis_negative: false,
        },
        orientation: "rotated_right",
        expected: { right: "Y+", left: "Y-", up: "X+", down: "X-" },
      },
    ];
    for (const { name, flags, orientation, expected } of cases) {
      const layout = jogPadLayout(flags, orientation);
      for (const screen of ["right", "left", "up", "down"] as const) {
        const btn = layout[screen];
        const native = `${btn.nativeAxis.toUpperCase()}${
          btn.nativeNegative ? "-" : "+"
        }`;
        expect(`${name} ${screen}: ${native}`).toBe(
          `${name} ${screen}: ${expected[screen]}`,
        );
      }
    }
  });

  it("moves toward the origin exactly on the minus directions", () => {
    const variants = [
      NATIVE_BOTTOM_LEFT,
      ROTATED_LEFT,
      ROTATED_RIGHT_BOTTOM_LEFT,
    ];
    for (const flags of variants) {
      const layout = jogPadLayout(flags);
      const byDirection = new Map([
        [layout.right.direction, layout.right],
        [layout.left.direction, layout.left],
        [layout.up.direction, layout.up],
        [layout.down.direction, layout.down],
      ]);
      expect(byDirection.get("-x")!.towardOrigin).toBe(true);
      expect(byDirection.get("-y")!.towardOrigin).toBe(true);
      expect(byDirection.get("+x")!.towardOrigin).toBe(false);
      expect(byDirection.get("+y")!.towardOrigin).toBe(false);
    }
  });

  it("describes direction and native axis in the tooltip", () => {
    const layout = jogPadLayout({
      x_axis_right: true,
      y_axis_down: false,
      x_axis_negative: true,
      y_axis_negative: false,
    }, "rotated_left");
    expect(jogTooltip(layout.right)).toBe(
      "Jog right (panel -x, machine Y-, toward origin)",
    );
    expect(jogTooltip(layout.up)).toBe(
      "Jog up (panel +y, machine X+, away from origin)",
    );
  });
});
