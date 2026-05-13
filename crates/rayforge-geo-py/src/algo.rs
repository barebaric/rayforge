use pyo3::prelude::*;
use crate::flex_point::{PyPoint2D, PyPoint3D, poly_to_points, extract_polygons};
use rayforge_geo::algo::clipping::{
    clip_line_segment_with_polygons, clip_line_segment_with_rect,
    subtract_polygons_from_line_segment,
};
use rayforge_geo::algo::minkowski::{
    calculate_input_scale, convolve_point_sequences, convolve_two_segments,
    get_inner_fit_polygon, get_no_fit_polygon,
    get_polygon_minkowski_sum_convex,
};
use rayforge_geo::algo::simplify::simplify_polyline;
use rayforge_geo::algo::fitting::{
    are_points_collinear, convert_arc_to_beziers_from_array, create_arc_cmd,
    create_line_cmd, fit_circle_to_3_points, fit_circle_to_points,
    fit_points_recursive, fit_points_with_primitives, flatten_to_points,
    get_polyline_arc_deviation, get_polyline_line_deviation,
    linearize_geometry, project_circle_center_to_bisector,
};
use rayforge_geo::algo::smooth::{
    compute_gaussian_kernel, resample_polyline, smooth_circularly,
    smooth_polyline, smooth_sub_segment,
};
use rayforge_geo::Segment3D;

pub fn build_algo_module(py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let algo_mod = PyModule::new(py, "algo")?;

    let minkowski_mod = PyModule::new(py, "minkowski")?;
    let simplify_mod = PyModule::new(py, "simplify")?;
    let clipping_mod = PyModule::new(py, "clipping")?;
    let smooth_mod = PyModule::new(py, "smooth")?;

    clipping_mod.add_function(wrap_pyfunction!(clip_line_segment_py, clipping_mod.clone())?)?;
    clipping_mod.add_function(wrap_pyfunction!(clip_line_segment_to_regions_py, clipping_mod.clone())?)?;
    clipping_mod.add_function(wrap_pyfunction!(subtract_polygons_from_line_segment_py, clipping_mod.clone())?)?;

    minkowski_mod.add_function(wrap_pyfunction!(minkowski_sum_convex_py, minkowski_mod.clone())?)?;
    minkowski_mod.add_function(wrap_pyfunction!(get_inner_fit_polygon_py, minkowski_mod.clone())?)?;
    minkowski_mod.add_function(wrap_pyfunction!(get_no_fit_polygon_py, minkowski_mod.clone())?)?;
    minkowski_mod.add_function(wrap_pyfunction!(calculate_input_scale_py, minkowski_mod.clone())?)?;
    minkowski_mod.add_function(wrap_pyfunction!(convolve_two_segments_py, minkowski_mod.clone())?)?;
    minkowski_mod.add_function(wrap_pyfunction!(convolve_point_sequences_py, minkowski_mod.clone())? )?;

    simplify_mod.add_function(wrap_pyfunction!(simplify_polyline_py, simplify_mod.clone())?)?;
    simplify_mod.add_function(wrap_pyfunction!(simplify_polyline_to_array_py, simplify_mod.clone())?)?;

    smooth_mod.add_function(wrap_pyfunction!(compute_gaussian_kernel_py, smooth_mod.clone())?)?;
    smooth_mod.add_function(wrap_pyfunction!(smooth_circularly_py, smooth_mod.clone())?)?;
    smooth_mod.add_function(wrap_pyfunction!(smooth_polyline_algo_py, smooth_mod.clone())?)?;
    smooth_mod.add_function(wrap_pyfunction!(smooth_sub_segment_py, smooth_mod.clone())?)?;
    smooth_mod.add_function(wrap_pyfunction!(resample_polyline_py, smooth_mod.clone())?)?;

    let fitting_mod = PyModule::new(py, "fitting")?;
    fitting_mod.add_function(wrap_pyfunction!(are_points_collinear_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(fit_circle_to_3_points_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(fit_circle_to_points_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(project_circle_center_to_bisector_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(flatten_to_points_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(linearize_geometry_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(create_line_cmd_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(create_arc_cmd_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(convert_arc_to_beziers_from_array_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(fit_points_recursive_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(fit_points_with_primitives_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(get_polyline_line_deviation_py, fitting_mod.clone())?)?;
    fitting_mod.add_function(wrap_pyfunction!(get_polyline_arc_deviation_py, fitting_mod.clone())?)?;

    algo_mod.add_submodule(&minkowski_mod)?;
    algo_mod.add_submodule(&simplify_mod)?;
    algo_mod.add_submodule(&clipping_mod)?;
    algo_mod.add_submodule(&smooth_mod)?;
    algo_mod.add_submodule(&fitting_mod)?;

    parent.add_submodule(&algo_mod)?;

    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("rayforge.core.geo.algo", &algo_mod)?;
    sys_modules.set_item("rayforge.core.geo.algo.minkowski", &minkowski_mod)?;
    sys_modules.set_item("rayforge.core.geo.algo.simplify", &simplify_mod)?;
    sys_modules.set_item("rayforge.core.geo.algo.clipping", &clipping_mod)?;
    sys_modules.set_item("rayforge.core.geo.algo.smooth", &smooth_mod)?;
    sys_modules.set_item("rayforge.core.geo.algo.fitting", &fitting_mod)?;

    Ok(())
}

type Point = (f64, f64);

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

#[pyfunction(name = "are_points_collinear")]
#[pyo3(signature = (points, tolerance=1e-6))]
fn are_points_collinear_py(points: Vec<(f64, f64, f64)>, tolerance: f64) -> bool {
    are_points_collinear(&points, tolerance)
}

#[pyfunction(name = "fit_circle_to_3_points")]
fn fit_circle_to_3_points_py(
    p1: PyPoint3D,
    p2: PyPoint3D,
    p3: PyPoint3D,
) -> Option<((f64, f64), f64)> {
    fit_circle_to_3_points((p1.0, p1.1, p1.2), (p2.0, p2.1, p2.2), (p3.0, p3.1, p3.2))
}

#[pyfunction(name = "fit_circle_to_points")]
fn fit_circle_to_points_py(points: Vec<(f64, f64, f64)>) -> Option<((f64, f64), f64, f64)> {
    fit_circle_to_points(&points)
}

#[pyfunction(name = "project_circle_center_to_bisector")]
fn project_circle_center_to_bisector_py(
    p1: PyPoint3D,
    p2: PyPoint3D,
    center: (f64, f64),
) -> (f64, f64) {
    project_circle_center_to_bisector((p1.0, p1.1, p1.2), (p2.0, p2.1, p2.2), center)
}

#[pyfunction(name = "flatten_to_points")]
fn flatten_to_points_py(data: Vec<Vec<f64>>, tolerance: f64) -> Vec<Vec<(f64, f64, f64)>> {
    let arr = to_data_array(data);
    flatten_to_points(&arr, tolerance)
}

#[pyfunction(name = "linearize_geometry")]
fn linearize_geometry_py(data: Vec<Vec<f64>>, tolerance: f64) -> Vec<Vec<f64>> {
    let arr = to_data_array(data);
    linearize_geometry(&arr, tolerance).into_iter().map(|r| r.to_vec()).collect()
}

#[pyfunction(name = "create_line_cmd")]
fn create_line_cmd_py(end_point: PyPoint3D) -> Vec<f64> {
    create_line_cmd((end_point.0, end_point.1, end_point.2)).to_vec()
}

#[pyfunction(name = "create_arc_cmd")]
fn create_arc_cmd_py(
    end: PyPoint3D,
    center: (f64, f64),
    start: PyPoint3D,
) -> Vec<f64> {
    create_arc_cmd((end.0, end.1, end.2), center, (start.0, start.1, start.2)).to_vec()
}

#[pyfunction(name = "convert_arc_to_beziers_from_array")]
#[pyo3(signature = (start, end, center_offset, clockwise))]
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

#[pyfunction(name = "fit_points_recursive")]
fn fit_points_recursive_py(
    points: Vec<(f64, f64, f64)>,
    tolerance: f64,
    start_idx: usize,
    end_idx: usize,
) -> Vec<Vec<f64>> {
    fit_points_recursive(&points, tolerance, start_idx, end_idx)
        .into_iter()
        .map(|r| r.to_vec())
        .collect()
}

#[pyfunction(name = "fit_points_with_primitives")]
fn fit_points_with_primitives_py(
    points: Vec<(f64, f64, f64)>,
    tolerance: f64,
) -> Vec<Vec<f64>> {
    fit_points_with_primitives(&points, tolerance)
        .into_iter()
        .map(|r| r.to_vec())
        .collect()
}

#[pyfunction(name = "get_polyline_line_deviation")]
fn get_polyline_line_deviation_py(
    points: Vec<(f64, f64, f64)>,
    start: usize,
    end: usize,
) -> (f64, usize) {
    get_polyline_line_deviation(&points, start, end)
}

#[pyfunction(name = "get_polyline_arc_deviation")]
fn get_polyline_arc_deviation_py(
    points: Vec<(f64, f64, f64)>,
    center: (f64, f64),
    radius: f64,
) -> f64 {
    get_polyline_arc_deviation(&points, center, radius)
}

#[pyfunction(name = "resample_polyline")]
fn resample_polyline_py(
    points: Vec<(f64, f64, f64)>,
    max_segment_length: f64,
    is_closed: bool,
) -> Vec<(f64, f64, f64)> {
    resample_polyline(&points, max_segment_length, is_closed)
}

#[pyfunction(name = "clip_line_segment_with_rect")]
fn clip_line_segment_py(
    p1: (f64, f64, f64),
    p2: (f64, f64, f64),
    rect: (f64, f64, f64, f64),
) -> Option<Segment3D> {
    clip_line_segment_with_rect(p1, p2, rect)
}

#[pyfunction(name = "subtract_polygons_from_line_segment")]
fn subtract_polygons_from_line_segment_py(
    p1: (f64, f64, f64),
    p2: (f64, f64, f64),
    regions: &Bound<'_, PyAny>,
) -> PyResult<Vec<Segment3D>> {
    let regions = extract_polygons(regions)?;
    Ok(subtract_polygons_from_line_segment(p1, p2, &regions))
}

#[pyfunction(name = "clip_line_segment_with_polygons")]
fn clip_line_segment_to_regions_py(
    p1: (f64, f64, f64),
    p2: (f64, f64, f64),
    regions: &Bound<'_, PyAny>,
) -> PyResult<Vec<Segment3D>> {
    let regions = extract_polygons(regions)?;
    Ok(clip_line_segment_with_polygons(p1, p2, &regions))
}

#[pyfunction(name = "get_polygon_minkowski_sum_convex")]
fn minkowski_sum_convex_py(
    poly_a: Vec<(i64, i64)>,
    poly_b: Vec<(i64, i64)>,
) -> Vec<Vec<(i64, i64)>> {
    get_polygon_minkowski_sum_convex(&poly_a, &poly_b)
}

#[pyfunction(name = "get_inner_fit_polygon")]
fn get_inner_fit_polygon_py(
    outer: Vec<PyPoint2D>,
    inner: Vec<PyPoint2D>,
) -> Vec<Vec<Point>> {
    get_inner_fit_polygon(&poly_to_points(outer), &poly_to_points(inner))
}

#[pyfunction(name = "get_no_fit_polygon")]
fn get_no_fit_polygon_py(
    subject: Vec<PyPoint2D>,
    tool: Vec<PyPoint2D>,
) -> Vec<Vec<Point>> {
    get_no_fit_polygon(&poly_to_points(subject), &poly_to_points(tool))
}

#[pyfunction(name = "calculate_input_scale")]
#[pyo3(signature = (polygons, max_int=2147483647))]
fn calculate_input_scale_py(polygons: &Bound<'_, PyAny>, max_int: i64) -> PyResult<f64> {
    let polys = extract_polygons(polygons)?;
    Ok(calculate_input_scale(&polys, max_int))
}

#[pyfunction(name = "convolve_two_segments")]
fn convolve_two_segments_py(
    a1: (i64, i64),
    a2: (i64, i64),
    b1: (i64, i64),
    b2: (i64, i64),
) -> Vec<(i64, i64)> {
    convolve_two_segments(a1, a2, b1, b2)
}

#[pyfunction(name = "convolve_point_sequences")]
fn convolve_point_sequences_py(
    seq_a: Vec<(i64, i64)>,
    seq_b: Vec<(i64, i64)>,
) -> Vec<Vec<(i64, i64)>> {
    convolve_point_sequences(&seq_a, &seq_b)
}

#[pyfunction(name = "simplify_polyline")]
fn simplify_polyline_py(points: Vec<PyPoint2D>, tolerance: f64) -> Vec<Point> {
    let pts = poly_to_points(points);
    let points_3d: Vec<rayforge_geo::Point3D> = pts.iter().map(|p| (p.0, p.1, 0.0)).collect();
    let result = simplify_polyline(&points_3d, tolerance);
    result.iter().map(|p| (p.0, p.1)).collect()
}

#[pyfunction(name = "simplify_polyline_to_array")]
fn simplify_polyline_to_array_py(data: Vec<Vec<f64>>, tolerance: f64) -> Vec<Vec<f64>> {
    let num_cols = data.first().map(|r| r.len()).unwrap_or(2);
    let points: Vec<rayforge_geo::Point3D> = data.iter().map(|r| (r[0], r[1], r.get(2).copied().unwrap_or(0.0))).collect();
    let result = simplify_polyline(&points, tolerance);
    result.iter().map(|p| {
        let mut row = Vec::with_capacity(num_cols);
        row.push(p.0);
        row.push(p.1);
        for _ in 2..num_cols {
            row.push(p.2);
        }
        row
    }).collect()
}

#[pyfunction(name = "compute_gaussian_kernel")]
fn compute_gaussian_kernel_py(amount: i32) -> (Vec<f64>, f64) {
    compute_gaussian_kernel(amount)
}

#[pyfunction(name = "smooth_circularly")]
fn smooth_circularly_py(
    points: Vec<(f64, f64, f64)>,
    kernel: Vec<f64>,
) -> Vec<(f64, f64, f64)> {
    smooth_circularly(&points, &kernel)
}

#[pyfunction(name = "smooth_polyline")]
#[pyo3(signature = (points, amount, corner_angle_threshold, is_closed=None))]
fn smooth_polyline_algo_py(
    points: Vec<(f64, f64, f64)>,
    amount: i32,
    corner_angle_threshold: f64,
    is_closed: Option<bool>,
) -> Vec<(f64, f64, f64)> {
    smooth_polyline(&points, amount, corner_angle_threshold, is_closed)
}

#[pyfunction(name = "smooth_sub_segment")]
fn smooth_sub_segment_py(
    points: Vec<(f64, f64, f64)>,
    kernel: Vec<f64>,
) -> Vec<(f64, f64, f64)> {
    smooth_sub_segment(&points, &kernel)
}