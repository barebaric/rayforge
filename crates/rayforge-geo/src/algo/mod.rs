//! Algo: Complex mathematical operations.
//!
//! This module provides advanced geometric algorithms including clipping,
//! curve fitting, Minkowski sums, simplification, and smoothing.

pub mod clipping;
pub mod fitting;
pub mod minkowski;
pub mod simplify;
pub mod smooth;

pub use clipping::*;
pub use fitting::*;
pub use minkowski::*;
pub use simplify::*;
pub use smooth::*;
