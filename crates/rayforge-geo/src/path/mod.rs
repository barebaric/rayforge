//! Path: Operations on Geometry (command arrays).
//!
//! This module provides functions for analyzing, manipulating, and transforming
//! geometric paths represented as command arrays.

pub mod analysis;
pub mod cleanup;
pub mod geometry;
pub mod intersect;
pub mod query;
pub mod split;
pub mod transform;

pub use analysis::*;
pub use cleanup::*;
pub use geometry::Geometry;
pub use intersect::*;
pub use query::*;
pub use split::*;
pub use transform::*;
