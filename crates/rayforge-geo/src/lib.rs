//! # RayForge Geometry Library
//!
//! A 2D/3D geometry library for CAD/CAM applications. Provides structures and functions
//! for creating, manipulating, and analyzing geometric shapes including lines, arcs,
//! Bezier curves, polygons, and complex paths.
//!
//! ## Core Concepts
//!
//! - **Geometry**: A path-based geometric structure supporting Move, Line, Arc, and Bezier commands
//! - **Primitives**: Basic geometric operations like point-in-polygon, line intersections
//! - **Analysis**: Path analysis including area calculation, winding order, and tangents
//! - **Query**: Path queries for bounding boxes, distances, and closest points
//!
//! ## Usage
//!
//! ```rust
//! use rayforge_geo::{Geometry, Point};
//!
//! let mut geo = Geometry::new();
//! geo.move_to(0.0, 0.0, 0.0);
//! geo.line_to(10.0, 0.0, 0.0);
//! geo.line_to(10.0, 10.0, 0.0);
//! geo.close_path();
//! geo.sync_to_data();
//!
//! let area = geo.area();
//! let rect = geo.rect();
//! ```

pub mod algo;
pub mod constants;
pub mod path;
pub mod shape;
pub mod types;

pub use algo::*;
pub use constants::*;
pub use path::*;
pub use shape::*;
pub use types::*;
