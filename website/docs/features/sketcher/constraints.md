---
description: "Learn about geometric and dimensional constraints in the Rayforge parametric 2D sketcher."
---

# Parametric Constraint System

The constraint system is the core of the parametric sketcher, allowing you to
define precise geometric relationships:

## Geometric Constraints

- **Coincident**: Forces two points to occupy the same location
- **Vertical**: Constrains a line to be perfectly vertical
- **Horizontal**: Constrains a line to be perfectly horizontal
- **Tangent**: Makes a line tangent to a circle or arc
- **Perpendicular**: Forces two lines, a line and an arc/circle, or two
  arcs/circles to meet at 90 degrees
- **Point on Line/Shape**: Constrains a point to lie on a line, arc, or circle
- **Collinear**: Forces two or more lines to lie on the same infinite line
- **Symmetry**: Creates symmetrical relationships between elements. Supports
  two modes:
  - **Point Symmetry**: Select 3 points (first is the center)
  - **Line Symmetry**: Select 2 points and 1 line (the line is the axis)

## Dimensional Constraints

- **Distance**: Sets the exact distance between two points or along a line
- **Diameter**: Defines the diameter of a circle
- **Radius**: Sets the radius of a circle or arc
- **Angle**: Enforces a specific angle between two lines
- **Aspect Ratio**: Forces the ratio between two distances to be equal to a
  specified value
- **Equal Length/Radius**: Forces multiple elements (lines, arcs, ellipses, or
  circles) to have the same length or radius
- **Equal Distance**: Makes two line segments the same length (different from
  Equal Length/Radius, which can also apply to arcs and circles)
