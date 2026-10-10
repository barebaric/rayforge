---
description:
  "Find the best line interval for engraving and the focal point of your laser with the Interval
  Test and the Focus Test."
---

# Interval and Focus Tests

Next to the [Material Test Grid](material-test-grid.md), the Tools menu offers two more calibration
jobs. Both are created as an ordinary layer with workpieces and operations: you can move it, frame
it and preview it like any other content, and change the settings of each cell or line afterwards
in its operation.

## Interval Test

**Tools → Create Interval Test** engraves a row of filled squares. Every square has its own
**Engrave** operation with its own line interval, spread evenly from the smallest to the largest
interval you enter, while power and speed stay the same for all of them. The labels under each
square show the interval in millimeters and the matching line density in lines per inch (LPI).

Pick the square that is filled evenly without visible lines and without burning too deep, and use
its interval for engravings on that material. The labels are cut with a separate low-power
operation before the squares.

## Focus Test

**Tools → Create Focus Test** finds the height at which the beam is sharpest. A positive offset
means more distance between the head and the material. The thinnest line marks the best focus.

| Method                           | How the height changes                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Z axis steps**                 | The head moves to each offset with relative Z moves and returns to its starting height at the end      |
| **Manual (pause between lines)** | The job pauses (`M0`) before each line; you move the head by hand and press Resume                     |
| **Ramp (tilted material)**       | One long line with distance ticks; you prop up one end of a flat strip so it rises under the line      |

Z axis steps are only offered on machines with a Z axis, and both Z axis steps and the manual method
need a G-code controller, because they use [Command](command.md) operations between the lines. The
offsets are limited to ±10 mm from the starting height.

For the manual method, focus the laser as usual first. The labels are engraved at that height.
At the first pause, set the head to the first offset; at every following pause, move it by one
step. Check that your controller stops on `M0` and that the Resume button continues the job before
you rely on it, for example with a dry run at 0% power.

For the ramp, the height under any point of the line follows from the rise of the strip: at a
distance _d_ along a line of length _L_ on a strip that rises _h_, the material is _h_ × _d_ / _L_
higher than at the start.
