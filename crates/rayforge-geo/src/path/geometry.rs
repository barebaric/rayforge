//! Geometry: Core geometric path structure.
//!
//! This module provides the main `Geometry` struct for building and manipulating
//! geometric paths. Paths are constructed using command-based operations like
//! `move_to`, `line_to`, `arc_to`, and `bezier_to`. Commands are stored in a
//! flat array format and can be queried for various properties.

use crate::types::{BezierControls, Command, Point3D, Rect};

/// A geometric path consisting of move, line, arc, and bezier commands.
///
/// The Geometry struct maintains two data stores:
/// - `pending_data`: Commands added since the last sync
/// - `data`: Committed commands that have been synchronized
///
/// This allows for batched command submission via `sync_to_data()`.
#[derive(Clone, Debug)]
pub struct Geometry {
    /// Committed command data array, each row is [type, x, y, z, aux1, aux2, aux3, aux4].
    pub(crate) data: Vec<[f64; 8]>,
    /// Pending command data awaiting synchronization.
    pub(crate) pending_data: Vec<[f64; 8]>,
    /// The position where the last MOVE command was issued.
    pub last_move_to: Point3D,
    /// Whether the geometry can be uniformly scaled without distortion (false if arcs present).
    pub uniform_scalable: bool,
}

impl PartialEq for Geometry {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.last_move_to == other.last_move_to
            && self.uniform_scalable == other.uniform_scalable
    }
}

impl Default for Geometry {
    fn default() -> Self {
        Self::new()
    }
}

impl Geometry {
    /// Creates a new empty Geometry.
    pub fn new() -> Self {
        Geometry {
            data: Vec::new(),
            pending_data: Vec::new(),
            last_move_to: (0.0, 0.0, 0.0),
            uniform_scalable: true,
        }
    }

    /// Moves the current position to the specified point.
    /// Starts a new subpath; subsequent commands will continue from this point.
    pub fn move_to(&mut self, x: f64, y: f64, z: f64) {
        self.last_move_to = (x, y, z);
        self.pending_data
            .push(Command::Move { end: (x, y, z) }.to_row());
    }

    /// Draws a straight line from the current position to the specified point.
    pub fn line_to(&mut self, x: f64, y: f64, z: f64) {
        self.pending_data
            .push(Command::Line { end: (x, y, z) }.to_row());
    }

    /// Closes the current subpath by drawing a line back to the starting point.
    /// The starting point is the position of the last `move_to` command.
    pub fn close_path(&mut self) {
        self.line_to(
            self.last_move_to.0,
            self.last_move_to.1,
            self.last_move_to.2,
        );
    }

    /// Draws an arc from the current position to the specified endpoint.
    ///
    /// The arc is defined by:
    /// - `(x, y, z)`: The endpoint coordinates
    /// - `(i, j)`: The offset from the start point to the arc center
    /// - `clockwise`: Whether to draw the arc in clockwise direction
    pub fn arc_to(
        &mut self,
        x: f64,
        y: f64,
        i: f64,
        j: f64,
        clockwise: bool,
        z: f64,
    ) {
        self.uniform_scalable = false;
        self.pending_data.push(
            Command::Arc {
                end: (x, y, z),
                center_offset: (i, j),
                clockwise,
            }
            .to_row(),
        );
    }

    /// Draws a cubic Bezier curve from the current position to the endpoint.
    ///
    /// The curve is defined by three control points:
    /// - `c1`: First control point
    /// - `c2`: Second control point
    /// - `p1`: End point (the start point is the current position)
    pub fn bezier_to(&mut self, controls: BezierControls, z: f64) {
        let (c1, c2, p1) = controls;
        self.pending_data.push(
            Command::Bezier {
                end: (p1.0, p1.1, z),
                control1: c1,
                control2: c2,
            }
            .to_row(),
        );
    }

    /// Commits all pending commands to the data array.
    /// After calling this, pending_data will be empty.
    pub fn sync_to_data(&mut self) {
        if self.pending_data.is_empty() {
            return;
        }
        self.data.append(&mut self.pending_data);
    }

    /// Returns a reference to the committed data, flushing any pending
    /// commands first.  Use this instead of accessing `.data` directly
    /// to guarantee the sync has happened.
    pub fn synced_data(&mut self) -> &Vec<[f64; 8]> {
        self.sync_to_data();
        &self.data
    }

    /// Returns a mutable reference to the committed data, flushing any
    /// pending commands first.
    pub fn synced_data_mut(&mut self) -> &mut Vec<[f64; 8]> {
        self.sync_to_data();
        &mut self.data
    }

    /// Returns a shared reference to the pending (unsynced) commands.
    pub fn pending_data(&self) -> &Vec<[f64; 8]> {
        &self.pending_data
    }

    /// Returns a shared reference to the committed data without syncing.
    /// Only use when you know the data is already synced.
    pub fn data(&self) -> &Vec<[f64; 8]> {
        &self.data
    }

    /// Returns a mutable reference to the pending (unsynced) commands.
    pub fn pending_data_mut(&mut self) -> &mut Vec<[f64; 8]> {
        &mut self.pending_data
    }

    /// Creates a new geometry from a list of 3D points connected by line segments.
    pub fn from_points(points: &[(f64, f64, f64)], close: bool) -> Self {
        let mut geo = Self::new();
        if points.is_empty() {
            return geo;
        }
        geo.move_to(points[0].0, points[0].1, points[0].2);
        for &(x, y, z) in points.iter().skip(1) {
            geo.line_to(x, y, z);
        }
        if close && points.len() > 1 {
            geo.close_path();
        }
        geo.sync_to_data();
        geo
    }

    /// Returns the total number of commands (both pending and committed).
    pub fn len(&self) -> usize {
        self.data.len() + self.pending_data.len()
    }

    /// Returns true if there are no commands (both pending and committed).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty() && self.pending_data.is_empty()
    }

    /// Clears all data (both pending and committed) and resets the geometry.
    pub fn clear(&mut self) {
        self.pending_data.clear();
        self.data.clear();
        self.uniform_scalable = true;
    }

    /// Returns the axis-aligned bounding rectangle of the geometry.
    /// Returns (0, 0, 0, 0) if the geometry is empty.
    pub fn rect(&self) -> Rect {
        if self.data.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        crate::query::get_bounding_rect_from_array(&self.data)
    }

    /// Returns the total path length (sum of all segment lengths).
    /// Returns 0.0 if the geometry is empty.
    pub fn distance(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        crate::query::get_total_distance_from_array(&self.data)
    }

    /// Returns the total area enclosed by the geometry (absolute value).
    /// Returns 0.0 if the geometry is empty.
    pub fn area(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        crate::analysis::get_area_from_array(&self.data)
    }

    /// Returns true if the geometry forms a closed path within the given tolerance.
    /// A closed path starts and ends at the same point (within tolerance).
    pub fn is_closed(&self, tolerance: f64) -> bool {
        if self.data.len() < 2 {
            return false;
        }
        let first = Command::from_row(&self.data[0]).expect("invalid command");
        if !matches!(first, Command::Move { .. }) {
            return false;
        }
        let start = first.end_point();
        let last = Command::from_row(&self.data[self.data.len() - 1])
            .expect("invalid command");
        let end = last.end_point();
        let dist_sq = (start.0 - end.0).powi(2)
            + (start.1 - end.1).powi(2)
            + (start.2 - end.2).powi(2);
        dist_sq < tolerance * tolerance
    }

    /// Creates a deep copy of the geometry.
    pub fn copy(&self) -> Geometry {
        let mut new_geo = Geometry::new();
        new_geo.last_move_to = self.last_move_to;
        new_geo.uniform_scalable = self.uniform_scalable;
        new_geo.data = self.data.clone();
        new_geo.pending_data = self.pending_data.clone();
        new_geo
    }

    /// Applies an affine transformation matrix to the geometry in place.
    /// The matrix is a 4x4 transformation matrix.
    pub fn transform(&mut self, matrix: &[[f64; 4]; 4]) {
        if self.data.is_empty() {
            return;
        }
        self.data = crate::transform::apply_affine_transform_to_array(
            &self.data, matrix,
        );
        let last_move_vec = [
            self.last_move_to.0,
            self.last_move_to.1,
            self.last_move_to.2,
            1.0,
        ];
        self.last_move_to = (
            matrix[0][0] * last_move_vec[0]
                + matrix[0][1] * last_move_vec[1]
                + matrix[0][2] * last_move_vec[2]
                + matrix[0][3] * last_move_vec[3],
            matrix[1][0] * last_move_vec[0]
                + matrix[1][1] * last_move_vec[1]
                + matrix[1][2] * last_move_vec[2]
                + matrix[1][3] * last_move_vec[3],
            matrix[2][0] * last_move_vec[0]
                + matrix[2][1] * last_move_vec[1]
                + matrix[2][2] * last_move_vec[2]
                + matrix[2][3] * last_move_vec[3],
        );
    }

    /// Extends this geometry by appending all commands from another geometry.
    /// Both committed data and pending data are appended.
    pub fn extend(&mut self, other: &Geometry) {
        if !other.data.is_empty() {
            self.sync_to_data();
            self.data.extend(other.data.clone());
        }
        if !other.pending_data.is_empty() {
            self.pending_data.extend(other.pending_data.clone());
        }
        self.uniform_scalable = self.uniform_scalable && other.uniform_scalable;
    }

    /// Returns the geometry decomposed into continuous segments.
    /// Each segment is a vector of points representing a continuous path
    /// between MOVE commands.
    pub fn segments(&self) -> Vec<Vec<Point3D>> {
        if self.data.is_empty() {
            return Vec::new();
        }

        let mut all_segments: Vec<Vec<Point3D>> = Vec::new();
        let mut current_segment: Vec<Point3D> = Vec::new();
        let mut last_point: Point3D = (0.0, 0.0, 0.0);

        for row in &self.data {
            let cmd = Command::from_row(row).expect("invalid command");
            let end_point = cmd.end_point();

            match cmd {
                Command::Move { .. } => {
                    if !current_segment.is_empty() {
                        all_segments.push(current_segment);
                        current_segment = Vec::new();
                    }
                    current_segment.push(end_point);
                }
                _ => {
                    if current_segment.is_empty() {
                        current_segment.push(last_point);
                    }
                    current_segment.push(end_point);
                }
            }
            last_point = end_point;
        }

        if !current_segment.is_empty() {
            all_segments.push(current_segment);
        }

        all_segments
    }

    /// Returns the command at the given index, if it exists.
    /// Only returns committed data (not pending).
    pub fn get_command_at(&self, index: usize) -> Option<[f64; 8]> {
        if index < self.data.len() {
            Some(self.data[index])
        } else {
            None
        }
    }

    /// Returns an iterator over all committed commands.
    pub fn iter_commands(&self) -> impl Iterator<Item = [f64; 8]> + '_ {
        self.data.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_geometry() {
        let geo = Geometry::new();
        assert!(geo.is_empty());
        assert!(geo.uniform_scalable);
    }

    #[test]
    fn test_move_to() {
        let mut geo = Geometry::new();
        geo.move_to(1.0, 2.0, 0.0);
        assert_eq!(geo.len(), 1);
        assert_eq!(geo.last_move_to, (1.0, 2.0, 0.0));
    }

    #[test]
    fn test_line_to() {
        let mut geo = Geometry::new();
        geo.move_to(0.0, 0.0, 0.0);
        geo.line_to(1.0, 0.0, 0.0);
        geo.line_to(1.0, 1.0, 0.0);
        assert_eq!(geo.len(), 3);
    }

    #[test]
    fn test_arc_to() {
        let mut geo = Geometry::new();
        geo.move_to(0.0, 0.0, 0.0);
        geo.arc_to(1.0, 0.0, 0.5, 0.0, true, 0.0);
        assert_eq!(geo.len(), 2);
        assert!(!geo.uniform_scalable);
    }

    #[test]
    fn test_bezier_to() {
        let mut geo = Geometry::new();
        geo.move_to(0.0, 0.0, 0.0);
        geo.bezier_to(((0.25, 0.5), (0.75, 0.5), (1.0, 1.0)), 0.0);
        assert_eq!(geo.len(), 2);
    }

    #[test]
    fn test_sync_to_data() {
        let mut geo = Geometry::new();
        geo.move_to(1.0, 2.0, 0.0);
        geo.line_to(3.0, 4.0, 0.0);
        assert_eq!(geo.pending_data.len(), 2);
        assert!(geo.data.is_empty());
        geo.sync_to_data();
        assert!(geo.pending_data.is_empty());
        assert_eq!(geo.data.len(), 2);
    }

    #[test]
    fn test_rect() {
        let mut geo = Geometry::new();
        geo.move_to(0.0, 0.0, 0.0);
        geo.line_to(10.0, 0.0, 0.0);
        geo.line_to(10.0, 5.0, 0.0);
        geo.sync_to_data();
        let rect = geo.rect();
        assert_eq!(rect, (0.0, 0.0, 10.0, 5.0));
    }

    #[test]
    fn test_is_closed() {
        let mut geo = Geometry::new();
        geo.move_to(0.0, 0.0, 0.0);
        geo.line_to(10.0, 0.0, 0.0);
        geo.line_to(10.0, 10.0, 0.0);
        geo.close_path();
        geo.sync_to_data();
        assert!(geo.is_closed(1e-6));
    }

    #[test]
    fn test_area() {
        let mut geo = Geometry::new();
        geo.move_to(0.0, 0.0, 0.0);
        geo.line_to(10.0, 0.0, 0.0);
        geo.line_to(10.0, 10.0, 0.0);
        geo.close_path();
        geo.sync_to_data();
        let area = geo.area();
        assert!((area - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_copy() {
        let mut geo = Geometry::new();
        geo.move_to(1.0, 2.0, 0.0);
        geo.line_to(3.0, 4.0, 0.0);
        let copy = geo.copy();
        assert_eq!(geo.data, copy.data);
        assert_eq!(geo.last_move_to, copy.last_move_to);
        assert_eq!(geo.uniform_scalable, copy.uniform_scalable);
    }

    #[test]
    fn test_segments() {
        let mut geo = Geometry::new();
        geo.move_to(0.0, 0.0, 0.0);
        geo.line_to(1.0, 0.0, 0.0);
        geo.move_to(2.0, 0.0, 0.0);
        geo.line_to(3.0, 0.0, 0.0);
        geo.sync_to_data();
        let segments = geo.segments();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].len(), 2);
        assert_eq!(segments[1].len(), 2);
    }
}
