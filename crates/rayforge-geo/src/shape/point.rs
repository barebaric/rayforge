//! Point: Point operations.
//!
//! This module provides basic point manipulation functions.

use crate::types::Point3D;

/// Computes the midpoint between two 3D points.
pub fn midpoint(a: Point3D, b: Point3D) -> Point3D {
    ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0, (a.2 + b.2) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_midpoint() {
        let a: Point3D = (0.0, 0.0, 0.0);
        let b: Point3D = (2.0, 4.0, 6.0);
        let result = midpoint(a, b);
        assert_eq!(result, (1.0, 2.0, 3.0));
    }
}
