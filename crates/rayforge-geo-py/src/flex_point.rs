use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use rayforge_geo::Point;

/// A 2D point that accepts tuples `(x, y)`, `(x, y, z)`,
/// or lists `[x, y]`, `[x, y, z]`, discarding the z coordinate.
#[derive(Clone, Copy, Debug)]
pub struct PyPoint2D(pub f64, pub f64);

impl<'a, 'py> FromPyObject<'a, 'py> for PyPoint2D {
    type Error = PyErr;
    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::types::PyAny>) -> Result<Self, Self::Error> {
        if let Ok((x, y)) = ob.extract::<(f64, f64)>() {
            return Ok(PyPoint2D(x, y));
        }
        if let Ok((x, y, _)) = ob.extract::<(f64, f64, f64)>() {
            return Ok(PyPoint2D(x, y));
        }
        let iter = ob.try_iter()?;
        let items: Vec<f64> = iter
            .take(3)
            .map(|i| i?.extract::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        if items.len() >= 2 {
            return Ok(PyPoint2D(items[0], items[1]));
        }
        Err(pyo3::exceptions::PyValueError::new_err(
            "expected a sequence of 2 or 3 floats",
        ))
    }
}

impl From<PyPoint2D> for (f64, f64) {
    fn from(p: PyPoint2D) -> Self {
        (p.0, p.1)
    }
}

impl From<&PyPoint2D> for (f64, f64) {
    fn from(p: &PyPoint2D) -> Self {
        (p.0, p.1)
    }
}

pub fn poly_to_points(poly: Vec<PyPoint2D>) -> Vec<(f64, f64)> {
    poly.into_iter().map(|p| (p.0, p.1)).collect()
}

/// A 3D point that accepts both 2-tuple `(x, y)` (z defaults to 0.0)
/// and 3-tuple `(x, y, z)`.
#[derive(Clone, Copy, Debug)]
pub struct PyPoint3D(pub f64, pub f64, pub f64);

impl<'a, 'py> FromPyObject<'a, 'py> for PyPoint3D {
    type Error = PyErr;
    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::types::PyAny>) -> Result<Self, Self::Error> {
        if let Ok((x, y, z)) = ob.extract::<(f64, f64, f64)>() {
            return Ok(PyPoint3D(x, y, z));
        }
        if let Ok((x, y)) = ob.extract::<(f64, f64)>() {
            return Ok(PyPoint3D(x, y, 0.0));
        }
        let iter = ob.try_iter()?;
        let items: Vec<f64> = iter
            .take(3)
            .map(|i| i?.extract::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        match items.len() {
            3 => Ok(PyPoint3D(items[0], items[1], items[2])),
            2 => Ok(PyPoint3D(items[0], items[1], 0.0)),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "expected a sequence of 2 or 3 floats",
            )),
        }
    }
}

impl From<PyPoint3D> for (f64, f64, f64) {
    fn from(p: PyPoint3D) -> Self {
        (p.0, p.1, p.2)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PyIntPoint2D(pub i64, pub i64);

impl<'a, 'py> FromPyObject<'a, 'py> for PyIntPoint2D {
    type Error = PyErr;
    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::types::PyAny>) -> Result<Self, Self::Error> {
        if let Ok((x, y)) = ob.extract::<(i64, i64)>() {
            return Ok(PyIntPoint2D(x, y));
        }
        let iter = ob.try_iter()?;
        let items: Vec<i64> = iter
            .take(2)
            .map(|i| i?.extract::<i64>())
            .collect::<Result<Vec<_>, _>>()?;
        if items.len() >= 2 {
            return Ok(PyIntPoint2D(items[0], items[1]));
        }
        Err(pyo3::exceptions::PyValueError::new_err(
            "expected a sequence of 2 integers",
        ))
    }
}

pub fn int_poly_to_points(poly: Vec<PyIntPoint2D>) -> Vec<(i64, i64)> {
    poly.into_iter().map(|p| (p.0, p.1)).collect()
}

/// Extract a single polygon (list of 2D points) from a Python object.
/// Accepts either a list of (x, y) tuples or an (N, 2) numpy array.
/// Returns `Vec<Point>` directly, avoiding `PyPoint2D` intermediate.
pub fn extract_polygon(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Point>> {
    if let Ok(arr) = ob.extract::<Bound<'_, PyArray2<f64>>>() {
        return Ok(polygon_from_numpy(&arr));
    }
    let mut points = Vec::new();
    for item in ob.try_iter()? {
        let item = item?;
        if let Ok(p) = item.extract::<PyPoint2D>() {
            points.push((p.0, p.1));
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "polygon elements must be (x, y) tuples or numpy array",
            ));
        }
    }
    Ok(points)
}

/// Extract a list of polygons from a Python object.
/// Accepts either a list of lists of (x, y) tuples or a list of (N, 2)
/// numpy arrays.
pub fn extract_polygons(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Point>>> {
    let mut result = Vec::new();
    for item in ob.try_iter()? {
        let item = item?;
        result.push(extract_polygon(&item)?);
    }
    Ok(result)
}

/// Zero-copy-friendly extraction of a polygon from a numpy (N, 2) array.
fn polygon_from_numpy(arr: &Bound<'_, PyArray2<f64>>) -> Vec<Point> {
    let readonly = arr.readonly();
    let view = readonly.as_array();
    let nrows = view.nrows();
    let mut points = Vec::with_capacity(nrows);
    for i in 0..nrows {
        points.push((view[[i, 0]], view[[i, 1]]));
    }
    points
}
