use crate::flex_point::{PyPoint2D, PyPoint3D};
use crate::geometry::{Geometry, PyCommand};
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use rayforge_geo::algo::fitting::{
    convert_arc_to_beziers_from_array, create_arc_cmd, create_line_cmd,
    flatten_to_points, linearize_geometry,
};
use rayforge_geo::path::analysis::{
    get_point_and_tangent_at_from_array, get_subpath_area_from_array,
    get_subpath_vertices_from_array, partial_segment_from_row,
    segment_length_from_row_flat,
};
use rayforge_geo::path::cleanup::{are_points_equal, get_segment_key};
use rayforge_geo::path::split::get_valid_contours_data;
use rayforge_geo::path::transform::{
    apply_affine_transform_to_array, map_geometry_to_frame,
};
use rayforge_geo::{
    check_intersection_from_array, check_self_intersection_from_array,
    close_all_contours, close_geometry_gaps_from_array, does_enclose,
    filter_to_external_contours, fit_curves, grow_geometry,
    normalize_winding_orders, remove_duplicate_segments, remove_inner_edges,
    reverse_contour, split_inner_and_outer_contours, split_into_components,
    split_into_contours, Geometry as CoreGeometry, Point, CMD_TYPE_ARC,
    CMD_TYPE_BEZIER, CMD_TYPE_LINE,
};

type GeometryVecPair = (Vec<Py<Geometry>>, Vec<Py<Geometry>>);

fn to_data_array(data: Vec<Vec<f64>>) -> Vec<[f64; 8]> {
    data.into_iter()
        .map(|r| {
            let mut a = [0.0; 8];
            let len = r.len().min(8);
            a[..len].copy_from_slice(&r[..len]);
            a
        })
        .collect()
}

#[pyfunction(name = "grow_geometry")]
fn grow_geometry_py(geometry: &Geometry, offset: f64) -> Geometry {
    let mut inner = geometry.inner.clone();
    inner.sync_to_data();
    let result = grow_geometry(&inner, offset);
    Geometry { inner: result }
}

#[pyfunction(name = "split_into_contours")]
fn split_into_contours_py(geometry: &Geometry) -> Vec<Geometry> {
    let mut inner = geometry.inner.clone();
    inner.sync_to_data();
    split_into_contours(&inner)
        .into_iter()
        .map(|g| Geometry { inner: g })
        .collect()
}

#[pyfunction(name = "split_into_components")]
fn split_into_components_py(geometry: &Geometry) -> Vec<Geometry> {
    let mut inner = geometry.inner.clone();
    inner.sync_to_data();
    split_into_components(&inner)
        .into_iter()
        .map(|g| Geometry { inner: g })
        .collect()
}

#[pyfunction]
fn get_bounding_rect_from_array(data: Vec<Vec<f64>>) -> (f64, f64, f64, f64) {
    let arr = to_data_array(data);
    rayforge_geo::get_bounding_rect_from_array(&arr)
}

#[pyfunction]
fn get_total_distance_from_array(data: Vec<Vec<f64>>) -> f64 {
    let arr = to_data_array(data);
    rayforge_geo::get_total_distance_from_array(&arr)
}

#[pyfunction]
fn extract_overcut_rows(
    py: Python<'_>,
    data: Option<Vec<Vec<f64>>>,
    max_length: f64,
) -> Option<Bound<'_, pyo3::types::PyAny>> {
    let data = data?;
    let arr = to_data_array(data);
    rayforge_geo::extract_overcut_rows(&arr, max_length).map(|rows| {
        let vecs: Vec<Vec<f64>> =
            rows.into_iter().map(|r| r.to_vec()).collect();
        PyArray2::<f64>::from_vec2(py, &vecs)
            .expect("failed to create numpy array")
            .as_any()
            .clone()
    })
}

#[pyfunction(name = "get_subpath_vertices_from_array")]
fn get_subpath_vertices_from_array_py(
    data: Vec<Vec<f64>>,
    subpath_index: usize,
) -> Vec<Point> {
    let arr = to_data_array(data);
    get_subpath_vertices_from_array(&arr, subpath_index)
}

#[pyfunction(name = "get_subpath_area_from_array")]
fn get_subpath_area_from_array_py(
    data: Vec<Vec<f64>>,
    subpath_index: usize,
) -> f64 {
    let arr = to_data_array(data);
    get_subpath_area_from_array(&arr, subpath_index)
}

#[pyfunction]
fn get_area_from_array(data: Vec<Vec<f64>>) -> f64 {
    let arr = to_data_array(data);
    rayforge_geo::get_area_from_array(&arr)
}

#[pyfunction]
fn get_path_winding_order_from_array(
    data: Vec<Vec<f64>>,
    start_cmd_index: usize,
) -> String {
    let arr = to_data_array(data);
    rayforge_geo::analysis::get_path_winding_order_from_array(
        &arr,
        start_cmd_index,
    )
    .to_string()
}

#[pyfunction]
fn get_point_tangent_at_py(
    data: Vec<Vec<f64>>,
    row_index: usize,
    t: f64,
) -> Option<(Point, Point)> {
    let arr = to_data_array(data);
    get_point_and_tangent_at_from_array(&arr, row_index, t)
}

#[pyfunction]
fn optimize_path_from_array(
    py: Python<'_>,
    data: Option<Vec<Vec<f64>>>,
    tolerance: f64,
    fit_arcs: bool,
) -> Bound<'_, pyo3::types::PyAny> {
    let Some(data) = data else {
        return PyArray2::<f64>::zeros(py, [0, 0], false).as_any().clone();
    };
    if data.is_empty() {
        return PyArray2::<f64>::zeros(py, [0, 0], false).as_any().clone();
    }
    let arr = to_data_array(data);
    let result = rayforge_geo::algo::fitting::optimize_path_from_array(
        &arr, tolerance, fit_arcs,
    );
    let vecs: Vec<Vec<f64>> = result.into_iter().map(|r| r.to_vec()).collect();
    if vecs.is_empty() {
        return PyArray2::<f64>::zeros(py, [0, 0], false).as_any().clone();
    }
    let np_arr = PyArray2::<f64>::from_vec2(py, &vecs)
        .expect("failed to create numpy array");
    np_arr.as_any().clone()
}

#[pyfunction(name = "does_enclose")]
fn does_enclose_py(container: &Geometry, content: &Geometry) -> PyResult<bool> {
    let mut c = container.inner.clone();
    c.sync_to_data();
    let mut ct = content.inner.clone();
    ct.sync_to_data();
    Ok(does_enclose(&c, &ct))
}

#[pyfunction]
#[pyo3(signature = (data, tolerance, progress_callback=None))]
fn fit_arcs(
    data: Option<Vec<Vec<f64>>>,
    tolerance: f64,
    progress_callback: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<Vec<f64>>>> {
    match data {
        Some(rows) => {
            let arr = to_data_array(rows);
            let result = rayforge_geo::fit_arcs(&arr, tolerance);
            if let Some(ref cb) = progress_callback {
                let _ = cb.call1((1.0f64,));
            }
            Ok(Some(result.iter().map(|r| r.to_vec()).collect()))
        }
        None => Ok(None),
    }
}

#[pyfunction(name = "reverse_contour")]
fn reverse_contour_py(contour: &Geometry) -> PyResult<Geometry> {
    let mut c = contour.inner.clone();
    c.sync_to_data();
    Ok(Geometry {
        inner: reverse_contour(&c),
    })
}

#[pyfunction(name = "split_inner_and_outer_contours")]
fn split_inner_and_outer_contours_py(
    py: Python<'_>,
    contours: Bound<'_, PyList>,
) -> PyResult<GeometryVecPair> {
    let mut original_geos: Vec<Py<Geometry>> = Vec::new();
    let mut geos: Vec<CoreGeometry> = Vec::new();
    for item in contours.iter() {
        let py_geo: Py<Geometry> =
            item.extract::<Py<Geometry>>().map_err(PyErr::from)?;
        original_geos.push(py_geo.clone_ref(py));
        let g = py_geo.borrow(py);
        let mut inner = g.inner.clone();
        inner.sync_to_data();
        geos.push(inner);
    }
    let (inner_indices, outer_indices) = split_inner_and_outer_contours(&geos);

    let inner: Vec<Py<Geometry>> = inner_indices
        .into_iter()
        .map(|i| original_geos[i].clone_ref(py))
        .collect();
    let outer: Vec<Py<Geometry>> = outer_indices
        .into_iter()
        .map(|i| original_geos[i].clone_ref(py))
        .collect();

    Ok((inner, outer))
}

#[pyfunction(name = "close_all_contours")]
fn close_all_contours_py(geometry: &Geometry) -> PyResult<Geometry> {
    let mut g = geometry.inner.clone();
    g.sync_to_data();
    Ok(Geometry {
        inner: close_all_contours(&g),
    })
}

#[pyfunction(name = "normalize_winding_orders")]
fn normalize_winding_orders_py(
    contours: Bound<'_, PyList>,
) -> PyResult<Vec<Geometry>> {
    let mut geos: Vec<CoreGeometry> = Vec::new();
    for item in contours.iter() {
        let g: PyRef<Geometry> = item.extract()?;
        let mut inner = g.inner.clone();
        inner.sync_to_data();
        geos.push(inner);
    }
    let result = normalize_winding_orders(&geos);
    Ok(result.into_iter().map(|g| Geometry { inner: g }).collect())
}

#[pyfunction(name = "filter_to_external_contours")]
fn filter_to_external_contours_py(
    py: Python<'_>,
    contours: Bound<'_, PyList>,
) -> PyResult<Vec<Py<Geometry>>> {
    let mut original_geos: Vec<Py<Geometry>> = Vec::new();
    let mut geos: Vec<CoreGeometry> = Vec::new();
    for item in contours.iter() {
        let py_geo: Py<Geometry> =
            item.extract::<Py<Geometry>>().map_err(PyErr::from)?;
        original_geos.push(py_geo.clone_ref(py));
        let g = py_geo.borrow(py);
        let mut inner = g.inner.clone();
        inner.sync_to_data();
        geos.push(inner);
    }
    let result = filter_to_external_contours(&geos);

    let synced_geos: Vec<_> = geos.iter().map(|g| {
        let mut gc = g.clone();
        gc.sync_to_data();
        gc
    }).collect();

    let output: Vec<Py<Geometry>> = result
        .into_iter()
        .map(|mut g| {
            g.sync_to_data();
            let match_idx =
                synced_geos.iter().position(|sg| sg.data() == g.data());
            match match_idx {
                Some(i) => original_geos[i].clone_ref(py),
                None => Py::new(py, Geometry { inner: g }).unwrap(),
            }
        })
        .collect();

    Ok(output)
}

#[pyfunction(name = "remove_inner_edges")]
fn remove_inner_edges_py(geometry: &Geometry) -> PyResult<Geometry> {
    let mut g = geometry.inner.clone();
    g.sync_to_data();
    Ok(Geometry {
        inner: remove_inner_edges(&g),
    })
}

#[pyfunction(name = "get_valid_contours_data")]
fn get_valid_contours_data_py<'py>(
    py: Python<'py>,
    contour_geometries: Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    let mut original_geos: Vec<Py<Geometry>> = Vec::new();
    let mut geos: Vec<CoreGeometry> = Vec::new();
    for item in contour_geometries.iter() {
        let py_geo: Py<Geometry> =
            item.extract::<Py<Geometry>>().map_err(PyErr::from)?;
        original_geos.push(py_geo.clone_ref(py));
        let g = py_geo.borrow(py);
        let mut inner = g.inner.clone();
        inner.sync_to_data();
        geos.push(inner);
    }
    let mut out: Vec<Bound<'py, pyo3::types::PyAny>> = Vec::new();
    for (orig_idx, geo) in geos.iter().enumerate() {
        let single_result = get_valid_contours_data(std::slice::from_ref(geo));
        if single_result.is_empty() {
            continue;
        }
        let (_, pts, closed) = single_result.into_iter().next().unwrap();
        let dict = PyDict::new(py);
        dict.set_item("geo", original_geos[orig_idx].clone_ref(py))?;
        let py_pts: Vec<(f64, f64)> = pts;
        dict.set_item("vertices", py_pts)?;
        dict.set_item("is_closed", closed)?;
        dict.set_item("original_index", orig_idx)?;
        out.push(dict.into_any());
    }
    PyList::new(py, out)
}

#[pyfunction]
fn close_geometry_gaps(
    geometry: &Geometry,
    tolerance: f64,
) -> PyResult<Geometry> {
    let mut g = geometry.inner.clone();
    g.sync_to_data();
    if !g.synced_data().is_empty() {
        let closed = close_geometry_gaps_from_array(g.synced_data(), tolerance);
        *g.synced_data_mut() = closed;
    }
    Ok(Geometry { inner: g })
}

#[pyfunction]
fn check_self_intersection(
    data: Option<Vec<Vec<f64>>>,
    fail_on_t_junction: bool,
) -> PyResult<bool> {
    match data {
        Some(rows) => {
            let arr = to_data_array(rows);
            Ok(check_self_intersection_from_array(&arr, fail_on_t_junction))
        }
        None => Ok(false),
    }
}

#[pyfunction]
fn check_intersection(
    data1: Option<Vec<Vec<f64>>>,
    data2: Option<Vec<Vec<f64>>>,
    fail_on_t_junction: bool,
) -> PyResult<bool> {
    match (data1, data2) {
        (Some(rows1), Some(rows2)) => {
            let arr1 = to_data_array(rows1);
            let arr2 = to_data_array(rows2);
            Ok(check_intersection_from_array(
                &arr1,
                &arr2,
                fail_on_t_junction,
            ))
        }
        _ => Ok(false),
    }
}

#[pyfunction]
fn check_self_intersection_from_array_py(
    data: Vec<Vec<f64>>,
    fail_on_t_junction: bool,
) -> bool {
    let arr = to_data_array(data);
    check_self_intersection_from_array(&arr, fail_on_t_junction)
}

#[pyfunction(name = "check_intersection_from_array")]
fn check_intersection_from_array_py(
    data1: Vec<Vec<f64>>,
    data2: Vec<Vec<f64>>,
    fail_on_t_junction: bool,
) -> bool {
    let arr1 = to_data_array(data1);
    let arr2 = to_data_array(data2);
    check_intersection_from_array(&arr1, &arr2, fail_on_t_junction)
}

#[pyfunction]
fn _partial_segment_from_row(
    row: Vec<f64>,
    start_point: (f64, f64, f64),
    t: f64,
) -> Option<Vec<f64>> {
    let arr = to_data_array(vec![row]);
    partial_segment_from_row(&arr[0], start_point, t).map(|r| r.to_vec())
}

#[pyfunction]
fn _segment_length_from_row(
    row: Vec<f64>,
    start_point: (f64, f64, f64),
) -> f64 {
    let arr = to_data_array(vec![row]);
    segment_length_from_row_flat(&arr[0], start_point)
}

#[pyfunction(name = "apply_affine_transform_to_array")]
fn apply_affine_transform_to_array_py(
    py: Python<'_>,
    data: Vec<Vec<f64>>,
    matrix: Vec<Vec<f64>>,
) -> Bound<'_, pyo3::types::PyAny> {
    let arr = to_data_array(data);
    let mat: [[f64; 4]; 4] = [
        [matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]],
        [matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]],
        [matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]],
        [matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]],
    ];
    let result = apply_affine_transform_to_array(&arr, &mat);
    let vecs: Vec<Vec<f64>> = result.into_iter().map(|r| r.to_vec()).collect();
    if vecs.is_empty() {
        return PyArray2::<f64>::zeros(py, [0, 8], false).as_any().clone();
    }
    PyArray2::<f64>::from_vec2(py, &vecs)
        .expect("failed to create numpy array")
        .as_any()
        .clone()
}

#[pyfunction(name = "map_geometry_to_frame")]
#[pyo3(signature = (geometry, origin, p_width, p_height, anchor_y=None, stable_src_height=None, anchor_x=None, stable_src_width=None))]
fn map_geometry_to_frame_py(
    geometry: &Geometry,
    origin: (f64, f64),
    p_width: (f64, f64),
    p_height: (f64, f64),
    anchor_y: Option<f64>,
    stable_src_height: Option<f64>,
    anchor_x: Option<f64>,
    stable_src_width: Option<f64>,
) -> Geometry {
    let result = map_geometry_to_frame(
        &geometry.inner,
        origin,
        p_width,
        p_height,
        anchor_y,
        stable_src_height,
        anchor_x,
        stable_src_width,
    );
    Geometry { inner: result }
}

#[pyfunction(name = "remove_duplicate_segments")]
#[pyo3(signature = (data, tolerance=1e-6))]
fn remove_duplicate_segments_py(
    py: Python<'_>,
    data: Option<Vec<Vec<f64>>>,
    tolerance: f64,
) -> Py<PyAny> {
    let Some(data) = data else {
        return py.None();
    };
    if data.is_empty() {
        return PyArray2::<f64>::zeros(py, [0, 0], false)
            .as_any()
            .clone()
            .unbind();
    }
    let arr = to_data_array(data);
    let result = remove_duplicate_segments(&arr, tolerance);
    let vecs: Vec<Vec<f64>> = result.into_iter().map(|r| r.to_vec()).collect();
    let np_arr = PyArray2::<f64>::from_vec2(py, &vecs)
        .expect("failed to create numpy array");
    np_arr.as_any().clone().unbind()
}

#[pyfunction(name = "flatten_to_points")]
#[pyo3(signature = (data, tolerance))]
fn flatten_to_points_py(
    data: Option<Vec<Vec<f64>>>,
    tolerance: f64,
) -> Vec<Vec<(f64, f64, f64)>> {
    match data {
        Some(rows) => {
            let arr = to_data_array(rows);
            flatten_to_points(&arr, tolerance)
        }
        None => Vec::new(),
    }
}

#[pyfunction(name = "linearize_geometry")]
#[pyo3(signature = (data, tolerance))]
fn linearize_geometry_py(
    py: Python<'_>,
    data: Option<Vec<Vec<f64>>>,
    tolerance: f64,
) -> Bound<'_, pyo3::types::PyAny> {
    let Some(data) = data else {
        return PyArray2::<f64>::zeros(py, [0, 8], false).as_any().clone();
    };
    if data.is_empty() {
        return PyArray2::<f64>::zeros(py, [0, 8], false).as_any().clone();
    }
    let arr = to_data_array(data);
    let result = linearize_geometry(&arr, tolerance);
    let vecs: Vec<Vec<f64>> = result.into_iter().map(|r| r.to_vec()).collect();
    let np_arr = PyArray2::<f64>::from_vec2(py, &vecs)
        .expect("failed to create numpy array");
    np_arr.as_any().clone()
}

#[pyfunction(name = "create_line_cmd")]
fn create_line_cmd_py(end_point: PyPoint3D) -> Vec<f64> {
    create_line_cmd((end_point.0, end_point.1, end_point.2)).to_vec()
}

#[pyfunction(name = "create_arc_cmd")]
fn create_arc_cmd_py(
    end: (f64, f64, f64),
    center: (f64, f64),
    start: (f64, f64, f64),
) -> Vec<f64> {
    create_arc_cmd(end, center, start).to_vec()
}

#[pyfunction(name = "convert_arc_to_beziers_from_array")]
fn convert_arc_to_beziers_from_array_py(
    start: (f64, f64, f64),
    end: (f64, f64, f64),
    center_offset: (f64, f64),
    clockwise: bool,
) -> Vec<Vec<f64>> {
    convert_arc_to_beziers_from_array(start, end, center_offset, clockwise)
        .into_iter()
        .map(|r| r.to_vec())
        .collect()
}

#[pyfunction(name = "fit_curves")]
fn fit_curves_py(
    py: Python<'_>,
    data: Option<Vec<Vec<f64>>>,
    tolerance: f64,
    preserve_beziers: bool,
    preserve_arcs: bool,
) -> Bound<'_, pyo3::types::PyAny> {
    let Some(data) = data else {
        return PyArray2::<f64>::zeros(py, [0, 0], false).as_any().clone();
    };
    if data.is_empty() {
        return PyArray2::<f64>::zeros(py, [0, 0], false).as_any().clone();
    }
    let arr = to_data_array(data);
    let result = fit_curves(&arr, tolerance, preserve_beziers, preserve_arcs);
    let vecs: Vec<Vec<f64>> = result.into_iter().map(|r| r.to_vec()).collect();
    let np_arr = PyArray2::<f64>::from_vec2(py, &vecs)
        .expect("failed to create numpy array");
    np_arr.as_any().clone()
}

#[pyfunction]
fn _are_points_equal(
    p1: (f64, f64, f64),
    p2: (f64, f64, f64),
    tolerance: f64,
) -> bool {
    let arr1 = [p1.0, p1.1, p1.2];
    let arr2 = [p2.0, p2.1, p2.2];
    are_points_equal(&arr1, &arr2, tolerance)
}

#[pyfunction]
fn _get_segment_key(
    py: Python<'_>,
    data: Vec<Vec<f64>>,
    index: usize,
    _tolerance: f64,
) -> Option<Py<PyAny>> {
    let row = data.get(index)?;
    let arr = to_data_array(vec![row.clone()]);
    let internal = get_segment_key(&arr[0])?;
    let cmd_type = internal.0;
    let end = (internal.1[0], internal.1[1], internal.1[2]);
    let params = internal.2;
    let result: Py<PyAny> = if cmd_type == CMD_TYPE_LINE as u32 {
        let a: [Bound<'_, PyAny>; 2] = [
            "LINE".into_pyobject(py).unwrap().into_any(),
            end.into_pyobject(py).unwrap().into_any(),
        ];
        PyTuple::new(py, a).unwrap().into_any().unbind()
    } else if cmd_type == CMD_TYPE_ARC as u32 {
        let a: [Bound<'_, PyAny>; 4] = [
            "ARC".into_pyobject(py).unwrap().into_any(),
            end.into_pyobject(py).unwrap().into_any(),
            (params[0], params[1]).into_pyobject(py).unwrap().into_any(),
            pyo3::types::PyBool::new(py, params[2] > 0.5)
                .as_any()
                .clone(),
        ];
        PyTuple::new(py, a).unwrap().into_any().unbind()
    } else if cmd_type == CMD_TYPE_BEZIER as u32 {
        let a: [Bound<'_, PyAny>; 4] = [
            "BEZIER".into_pyobject(py).unwrap().into_any(),
            end.into_pyobject(py).unwrap().into_any(),
            (params[0], params[1]).into_pyobject(py).unwrap().into_any(),
            (params[2], params[3]).into_pyobject(py).unwrap().into_any(),
        ];
        PyTuple::new(py, a).unwrap().into_any().unbind()
    } else {
        return None;
    };
    Some(result)
}

fn _extract_point3(key: &Bound<'_, PyAny>, idx: usize) -> PyResult<[f64; 3]> {
    let p: PyPoint3D = key.get_item(idx)?.extract()?;
    Ok([p.0, p.1, p.2])
}

fn _extract_point2(key: &Bound<'_, PyAny>, idx: usize) -> PyResult<(f64, f64)> {
    let p: (f64, f64) = key.get_item(idx)?.extract()?;
    Ok(p)
}

#[pyfunction]
fn _are_segments_equal(
    key1: &Bound<'_, PyAny>,
    key2: &Bound<'_, PyAny>,
    tolerance: f64,
) -> PyResult<bool> {
    let type1: String = key1.get_item(0)?.extract()?;
    let type2: String = key2.get_item(0)?.extract()?;
    if type1 != type2 {
        return Ok(false);
    }
    match type1.as_str() {
        "LINE" => {
            let a1 = _extract_point3(key1, 1)?;
            let a2 = _extract_point3(key2, 1)?;
            let b1 = _extract_point3(key1, 2)?;
            let b2 = _extract_point3(key2, 2)?;
            Ok(are_points_equal(&a1, &a2, tolerance)
                && are_points_equal(&b1, &b2, tolerance))
        }
        "ARC" => {
            let key1_len = key1.len()?;
            let key2_len = key2.len()?;
            if key1_len == 4 && key2_len == 4 {
                let p1a = _extract_point3(key1, 1)?;
                let p1b = _extract_point3(key2, 1)?;
                let ca = _extract_point2(key1, 2)?;
                let cb = _extract_point2(key2, 2)?;
                let cwa: bool = key1.get_item(3)?.extract()?;
                let cwb: bool = key2.get_item(3)?.extract()?;
                Ok(are_points_equal(&p1a, &p1b, tolerance)
                    && (ca.0 - cb.0).abs() < tolerance
                    && (ca.1 - cb.1).abs() < tolerance
                    && cwa == cwb)
            } else {
                let a1 = _extract_point3(key1, 1)?;
                let a2 = _extract_point3(key2, 1)?;
                let b1 = _extract_point3(key1, 2)?;
                let b2 = _extract_point3(key2, 2)?;
                let ca = _extract_point2(key1, 3)?;
                let cb = _extract_point2(key2, 3)?;
                let cwa: bool = key1.get_item(4)?.extract()?;
                let cwb: bool = key2.get_item(4)?.extract()?;
                Ok(are_points_equal(&a1, &a2, tolerance)
                    && are_points_equal(&b1, &b2, tolerance)
                    && (ca.0 - cb.0).abs() < tolerance
                    && (ca.1 - cb.1).abs() < tolerance
                    && cwa == cwb)
            }
        }
        "BEZIER" => {
            let p1a = _extract_point3(key1, 1)?;
            let p1b = _extract_point3(key2, 1)?;
            let c1a: (f64, f64) = key1.get_item(2)?.extract()?;
            let c1b: (f64, f64) = key2.get_item(2)?.extract()?;
            let c2a: (f64, f64) = key1.get_item(3)?.extract()?;
            let c2b: (f64, f64) = key2.get_item(3)?.extract()?;
            Ok(are_points_equal(&p1a, &p1b, tolerance)
                && (c1a.0 - c1b.0).abs() < tolerance
                && (c1a.1 - c1b.1).abs() < tolerance
                && (c2a.0 - c2b.0).abs() < tolerance
                && (c2a.1 - c2b.1).abs() < tolerance)
        }
        _ => Ok(false),
    }
}

#[pyfunction(name = "get_angle_at_vertex")]
fn get_angle_at_vertex_py(
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
) -> f64 {
    rayforge_geo::path::analysis::get_angle_at_vertex(p0, p1, p2)
}

#[pyfunction(name = "remove_duplicates")]
fn remove_duplicates_py(points: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    rayforge_geo::path::analysis::remove_duplicates(&points)
}

#[pyfunction(name = "is_clockwise")]
fn is_clockwise_py(points: Vec<PyPoint2D>) -> bool {
    let pts: Vec<(f64, f64)> = points.iter().map(|p| (p.0, p.1)).collect();
    rayforge_geo::path::analysis::is_clockwise(&pts)
}

#[pyfunction(name = "is_closed")]
#[pyo3(signature = (commands, tolerance=1e-6))]
fn is_closed_py(commands: Vec<Vec<f64>>, tolerance: f64) -> bool {
    let arr = to_data_array(commands);
    rayforge_geo::path::analysis::is_closed(&arr, tolerance)
}

#[pyfunction(name = "get_outward_normal_at_from_array")]
fn get_outward_normal_at_from_array_py(
    data: Vec<Vec<f64>>,
    row_index: usize,
    t: f64,
) -> Option<(f64, f64)> {
    let arr = to_data_array(data);
    rayforge_geo::path::analysis::get_outward_normal_at_from_array(
        &arr, row_index, t,
    )
}

pub fn build_path_module(
    py: Python,
    parent: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let path_mod = PyModule::new(py, "path")?;

    path_mod.add_class::<Geometry>()?;
    path_mod.add_class::<PyCommand>()?;

    path_mod.add_function(wrap_pyfunction!(
        remove_duplicate_segments_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        flatten_to_points_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        linearize_geometry_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        create_line_cmd_py,
        path_mod.clone()
    )?)?;
    path_mod
        .add_function(wrap_pyfunction!(create_arc_cmd_py, path_mod.clone())?)?;
    path_mod.add_function(wrap_pyfunction!(
        convert_arc_to_beziers_from_array_py,
        path_mod.clone()
    )?)?;
    path_mod
        .add_function(wrap_pyfunction!(fit_curves_py, path_mod.clone())?)?;
    path_mod
        .add_function(wrap_pyfunction!(_are_points_equal, path_mod.clone())?)?;
    path_mod
        .add_function(wrap_pyfunction!(_get_segment_key, path_mod.clone())?)?;
    path_mod.add_function(wrap_pyfunction!(
        _are_segments_equal,
        path_mod.clone()
    )?)?;

    path_mod
        .add_function(wrap_pyfunction!(grow_geometry_py, path_mod.clone())?)?;
    path_mod.add_function(wrap_pyfunction!(
        split_into_contours_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        split_into_components_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        get_bounding_rect_from_array,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        get_total_distance_from_array,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        extract_overcut_rows,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        get_subpath_vertices_from_array_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        get_subpath_area_from_array_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        get_area_from_array,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        get_path_winding_order_from_array,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        get_point_tangent_at_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        optimize_path_from_array,
        path_mod.clone()
    )?)?;
    path_mod
        .add_function(wrap_pyfunction!(does_enclose_py, path_mod.clone())?)?;
    path_mod.add_function(wrap_pyfunction!(fit_arcs, path_mod.clone())?)?;
    path_mod.add_function(wrap_pyfunction!(
        reverse_contour_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        split_inner_and_outer_contours_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        close_all_contours_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        normalize_winding_orders_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        filter_to_external_contours_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        remove_inner_edges_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        get_valid_contours_data_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        close_geometry_gaps,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        check_self_intersection,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        check_intersection,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        check_self_intersection_from_array_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        check_intersection_from_array_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        _partial_segment_from_row,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        _segment_length_from_row,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        apply_affine_transform_to_array_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        map_geometry_to_frame_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        get_angle_at_vertex_py,
        path_mod.clone()
    )?)?;
    path_mod.add_function(wrap_pyfunction!(
        remove_duplicates_py,
        path_mod.clone()
    )?)?;
    path_mod
        .add_function(wrap_pyfunction!(is_clockwise_py, path_mod.clone())?)?;
    path_mod.add_function(wrap_pyfunction!(is_closed_py, path_mod.clone())?)?;
    path_mod.add_function(wrap_pyfunction!(
        get_outward_normal_at_from_array_py,
        path_mod.clone()
    )?)?;

    parent.add_submodule(&path_mod)?;

    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("rayforge.core.geo.path", &path_mod)?;

    Ok(())
}
