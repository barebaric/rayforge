//! Shape: Base geometric entities.
//!
//! This module provides fundamental geometric shapes and their operations.

pub mod arc;
pub mod bezier;
pub mod circle;
pub mod line;
pub mod point;
pub mod polygon;

pub use arc::*;
pub use bezier::*;
pub use circle::*;
pub use line::*;
pub use point::*;
pub use polygon::*;
