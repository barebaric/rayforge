/**
 * Pure derivation of the jog pad layout from the viewport contract's
 * axis flags.
 *
 * All jog inputs are PANEL-space directions ("+x"/"-x"/"+y"/"-y"); the
 * backend converts them to native axis deltas. The panel origin is the
 * displayed origin corner, so the flags decide which on-screen arrow
 * corresponds to which panel direction:
 *
 * - `x_axis_right`: the displayed origin sits on the right, panel X
 *   grows toward the left.
 * - `y_axis_down`: the displayed origin sits at the top, panel Y grows
 *   downward.
 * - `x_axis_negative` / `y_axis_negative`: the displayed axis reflects
 *   a reversed native axis.
 */

export type PanelDirection = "+x" | "-x" | "+y" | "-y";

export type ScreenDirection = "right" | "left" | "up" | "down";

export interface AxisFlags {
  x_axis_right: boolean;
  y_axis_down: boolean;
  x_axis_negative: boolean;
  y_axis_negative: boolean;
}

export interface JogButton {
  /** Panel-space direction sent to POST /machine/jog. */
  direction: PanelDirection;
  screen: ScreenDirection;
  /** Panel axis the button drives. */
  axis: "x" | "y";
  /** True when the motion approaches the displayed origin corner. */
  towardOrigin: boolean;
  /** Native axis driven (swapped when the panel is rotated). */
  nativeAxis: "x" | "y";
  /** True when the motion decreases the native coordinate. */
  nativeNegative: boolean;
}

export interface JogPadLayout {
  right: JogButton;
  left: JogButton;
  up: JogButton;
  down: JogButton;
}

export type PanelOrientation = "native" | "rotated_left" | "rotated_right";

function axisButton(
  axis: "x" | "y",
  positive: boolean,
  screen: ScreenDirection,
  flags: AxisFlags,
  orientation: PanelOrientation,
): JogButton {
  const direction: PanelDirection = `${positive ? "+" : "-"}${axis}` as PanelDirection;
  return {
    direction,
    screen,
    axis,
    towardOrigin: !positive,
    nativeAxis:
      orientation === "native" ? axis : axis === "x" ? "y" : "x",
    nativeNegative: nativeNegativeOf(axis, positive, flags),
  };
}

/**
 * True when the motion decreases the native coordinate. Under rotation
 * the displayed axis drives the other native axis, but the sign still
 * follows the displayed axis itself: direction, its reversal flag, and
 * the origin-corner polarity multiply (verified against
 * MachinePanel.calculate_jog across all origin/reversal/orientation
 * combinations).
 */
function nativeNegativeOf(
  axis: "x" | "y",
  positive: boolean,
  flags: AxisFlags,
): boolean {
  const originPolarity =
    axis === "x" ? flags.x_axis_right : flags.y_axis_down;
  const reversed = axis === "x" ? flags.x_axis_negative : flags.y_axis_negative;
  let sign = positive ? 1 : -1;
  if (reversed) sign = -sign;
  if (originPolarity) sign = -sign;
  return sign < 0;
}

/** Builds the four directional buttons for the given contract flags. */
export function jogPadLayout(
  flags: AxisFlags,
  orientation: PanelOrientation = "native",
): JogPadLayout {
  return {
    right: axisButton("x", !flags.x_axis_right, "right", flags, orientation),
    left: axisButton("x", flags.x_axis_right, "left", flags, orientation),
    up: axisButton("y", !flags.y_axis_down, "up", flags, orientation),
    down: axisButton("y", flags.y_axis_down, "down", flags, orientation),
  };
}

/** Human-readable tooltip for a jog button. */
export function jogTooltip(btn: JogButton): string {
  const native = `${btn.nativeAxis.toUpperCase()}${
    btn.nativeNegative ? "-" : "+"
  }`;
  const origin = btn.towardOrigin ? "toward origin" : "away from origin";
  return `Jog ${btn.screen} (panel ${btn.direction}, machine ${native}, ${origin})`;
}
