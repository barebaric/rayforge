use crate::constants::*;

/// A 2D point represented as (x, y) coordinates.
pub type Point = (f64, f64);

/// A 3D point represented as (x, y, z) coordinates.
pub type Point3D = (f64, f64, f64);

/// A 2D axis-aligned bounding box represented as (x_min, y_min, x_max, y_max).
pub type Rect = (f64, f64, f64, f64);

/// A cubic Bezier curve defined by four control points: (p0, c1, c2, p1).
/// - p0: Start point
/// - c1: First control point
/// - c2: Second control point
/// - p1: End point
pub type CubicBezier = (Point, Point, Point, Point);

/// Control points for a cubic Bezier curve: (c1, c2, p1).
/// c1 and c2 are the control points, p1 is the end point.
pub type BezierControls = (Point, Point, Point);

/// Result of splitting a cubic Bezier curve: (first_half, second_half).
pub type BezierSplit = (CubicBezier, CubicBezier);

/// A pair of geometry vectors: (inner_contours, outer_contours).
pub type GeometryPair<T> = (Vec<T>, Vec<T>);

/// A 2D polygon represented as a list of vertices in order.
pub type Polygon = Vec<Point>;

/// A 3D polygon represented as a list of 3D vertices in order.
pub type Polygon3D = Vec<Point3D>;

/// An edge represented as a pair of points: (start, end).
pub type Edge = (Point, Point);

/// A line segment in 3D space represented as (start, end).
pub type Segment3D = (Point3D, Point3D);

/// A row in the geometry command array: [type, x, y, z, aux1, aux2, aux3, aux4].
/// type is the command type (move/line/arc/bezier), x/y/z are the endpoint,
/// and aux1-4 are additional parameters (e.g., control points, arc center offsets).
pub type CommandRow = (f64, f64, f64, f64, f64, f64, f64, f64);

/// Typed view over a single `[f64; 8]` geometry command row.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    Move {
        end: Point3D,
    },
    Line {
        end: Point3D,
    },
    Arc {
        end: Point3D,
        center_offset: Point,
        clockwise: bool,
    },
    Bezier {
        end: Point3D,
        control1: Point,
        control2: Point,
    },
}

impl Command {
    pub fn from_row(row: &[f64; 8]) -> Result<Command, String> {
        let cmd_type = row[COL_TYPE] as i32;
        let end = (row[COL_X], row[COL_Y], row[COL_Z]);
        match cmd_type {
            t if t == CMD_TYPE_MOVE as i32 => Ok(Command::Move { end }),
            t if t == CMD_TYPE_LINE as i32 => Ok(Command::Line { end }),
            t if t == CMD_TYPE_ARC as i32 => Ok(Command::Arc {
                end,
                center_offset: (row[COL_I], row[COL_J]),
                clockwise: row[COL_CW] != 0.0,
            }),
            t if t == CMD_TYPE_BEZIER as i32 => Ok(Command::Bezier {
                end,
                control1: (row[COL_C1X], row[COL_C1Y]),
                control2: (row[COL_C2X], row[COL_C2Y]),
            }),
            _ => Err(format!("unknown command type: {}", cmd_type)),
        }
    }

    pub fn to_row(&self) -> [f64; 8] {
        match self {
            Command::Move { end } => {
                [CMD_TYPE_MOVE, end.0, end.1, end.2, 0.0, 0.0, 0.0, 0.0]
            }
            Command::Line { end } => {
                [CMD_TYPE_LINE, end.0, end.1, end.2, 0.0, 0.0, 0.0, 0.0]
            }
            Command::Arc {
                end,
                center_offset,
                clockwise,
            } => [
                CMD_TYPE_ARC,
                end.0,
                end.1,
                end.2,
                center_offset.0,
                center_offset.1,
                if *clockwise { 1.0 } else { 0.0 },
                0.0,
            ],
            Command::Bezier {
                end,
                control1,
                control2,
            } => [
                CMD_TYPE_BEZIER,
                end.0,
                end.1,
                end.2,
                control1.0,
                control1.1,
                control2.0,
                control2.1,
            ],
        }
    }

    pub fn end_point(&self) -> Point3D {
        match self {
            Command::Move { end } => *end,
            Command::Line { end } => *end,
            Command::Arc { end, .. } => *end,
            Command::Bezier { end, .. } => *end,
        }
    }
}

pub trait CommandSlice {
    fn iter_commands(&self) -> impl Iterator<Item = Command> + '_;
    fn try_iter_commands(
        &self,
    ) -> impl Iterator<Item = Result<Command, String>> + '_;
}

impl CommandSlice for [[f64; 8]] {
    fn iter_commands(&self) -> impl Iterator<Item = Command> + '_ {
        self.iter()
            .map(|r| Command::from_row(r).expect("invalid command"))
    }

    fn try_iter_commands(
        &self,
    ) -> impl Iterator<Item = Result<Command, String>> + '_ {
        self.iter().map(Command::from_row)
    }
}

/// A 2D integer point for grid-based operations.
pub type IntPoint = (i64, i64);

/// A 2D integer polygon for grid-based operations.
pub type IntPolygon = Vec<IntPoint>;

/// A 3D axis-aligned bounding box with separate min/max bounds for each axis.
#[derive(Clone, Debug, Default)]
pub struct Rect3D {
    /// Minimum x coordinate (left face).
    pub x_min: f64,
    /// Maximum x coordinate (right face).
    pub x_max: f64,
    /// Minimum y coordinate (bottom face).
    pub y_min: f64,
    /// Maximum y coordinate (top face).
    pub y_max: f64,
    /// Minimum z coordinate (front face).
    pub z_min: f64,
    /// Maximum z coordinate (back face).
    pub z_max: f64,
}

/// Container for contour/path data with geometric and topological information.
#[derive(Clone, Debug)]
pub struct ContourData {
    /// The geometric data of the contour.
    pub geo: super::Geometry,
    /// Whether the contour forms a closed path.
    pub is_closed: bool,
    /// List of vertices defining the contour.
    pub vertices: Polygon,
    /// The signed area of the contour (positive for CCW, negative for CW).
    pub area: f64,
    /// Winding order: "cw", "ccw", or "unknown".
    pub winding_order: String,
}
