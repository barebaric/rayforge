use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::flex_point::{PyPoint2D, PyPoint3D, poly_to_points, extract_polygons};
use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use rayforge_geo::path::analysis::is_arc_clockwise;
use rayforge_geo::{BezierSplit, Point, Segment3D, CMD_TYPE_ARC};
use rayforge_geo::shape::arc::{
    _linearize_arc_from_array, does_arc_intersect_circle,
    does_arc_intersect_rect, get_arc_angles, get_arc_bounds,
    get_arc_closest_point, get_arc_direction, get_arc_midpoint,
    is_angle_between, is_arc_inside_polygons, linearize_arc, normalize_angle,
};
use rayforge_geo::shape::bezier::{
    bezier_flatness_sq, clip_bezier_with_rect, convert_cubic_bezier_to_quadratic,
    flatten_bezier, get_bezier_bounds, get_bezier_point_at,
    get_bezier_rect_intersections, is_bezier_inside_polygons, linearize_bezier,
    linearize_bezier_adaptive, linearize_bezier_from_array, linearize_bezier_segment,
    perp_dist_sq, split_bezier,
};
use rayforge_geo::shape::circle::{
    does_circle_intersect_rect, get_circle_circle_intersections,
    is_circle_inside_rect, line_segment_intersects_circle,
    project_point_onto_circle,
};
use rayforge_geo::shape::line::{
    does_line_segment_intersect_circle, does_line_segment_intersect_rect,
    does_rect_contain_rect, get_line_closest_point, get_line_line_intersection,
    get_line_segment_closest_point, get_line_segment_intersection,
    get_line_segment_polygon_intersections, get_point_line_distance,
    is_point_inside_rect, is_point_on_segment,
};
use rayforge_geo::shape::point::midpoint;
use rayforge_geo::shape::polygon::{
    clean_polygon, flip_polygon, flip_polygons, get_polygon_bounds,
    get_polygon_centroid, get_polygon_convex_hull, get_polygon_edges,
    get_polygon_group_bounds, get_polygon_perimeter, get_polygon_signed_area,
    get_polygons_difference, get_polygons_intersection, get_polygons_union,
    is_almost_equal, is_point_inside_polygon, is_polygon_convex,
    normalize_polygons, offset_polygon, point_line_distance, polygons_intersect,
    rotate_polygon, rotate_polygons, scale_polygon, to_clipper_from_points,
    translate_bounds, translate_polygon, translate_polygons,
};

fn _arc_row_from_any(arc_cmd: &Bound<'_, PyAny>) -> PyResult<[f64; 8]> {
    if let Ok(row) = arc_cmd.extract::<Vec<f64>>() {
        let mut arr = [0.0; 8];
        let len = row.len().min(8);
        arr[..len].copy_from_slice(&row[..len]);
        return Ok(arr);
    }
    if let Ok(end) = arc_cmd.getattr("end") {
        let end: (f64, f64, f64) = end.extract()?;
        let center_offset: (f64, f64) = arc_cmd.getattr("center_offset")?.extract()?;
        let clockwise: bool = arc_cmd.getattr("clockwise")?.extract()?;
        let arr: [f64; 8] = [
            CMD_TYPE_ARC,
            end.0, end.1, end.2,
            center_offset.0, center_offset.1,
            if clockwise { 1.0 } else { 0.0 },
            0.0,
        ];
        return Ok(arr);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a command row or a MockArc-like namedtuple with end, center_offset, clockwise",
    ))
}

pub fn build_shape_module(
    py: Python,
    parent: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let shape_mod = PyModule::new(py, "shape")?;

    let arc_mod = PyModule::new(py, "arc")?;
    arc_mod
        .add_function(wrap_pyfunction!(get_arc_bounds_py, arc_mod.clone())?)?;
    arc_mod.add_function(wrap_pyfunction!(
        get_arc_direction_py,
        arc_mod.clone()
    )?)?;
    arc_mod.add_function(wrap_pyfunction!(
        get_arc_closest_point_py,
        arc_mod.clone()
    )?)?;
    arc_mod.add_function(wrap_pyfunction!(
        get_arc_midpoint_py,
        arc_mod.clone()
    )?)?;
    arc_mod
        .add_function(wrap_pyfunction!(get_arc_angles_py, arc_mod.clone())?)?;
    arc_mod.add_function(wrap_pyfunction!(
        does_arc_intersect_rect_py,
        arc_mod.clone()
    )?)?;
    arc_mod.add_function(wrap_pyfunction!(
        does_arc_intersect_circle_py,
        arc_mod.clone()
    )?)?;
    arc_mod.add_function(wrap_pyfunction!(
        is_arc_clockwise_py,
        arc_mod.clone()
    )?)?;
    arc_mod.add_function(wrap_pyfunction!(
        is_arc_inside_polygons_py,
        arc_mod.clone()
    )?)?;
    arc_mod.add_function(wrap_pyfunction!(
        is_angle_between_py,
        arc_mod.clone()
    )?)?;
    arc_mod
        .add_function(wrap_pyfunction!(normalize_angle_py, arc_mod.clone())?)?;
    arc_mod
        .add_function(wrap_pyfunction!(linearize_arc_py, arc_mod.clone())?)?;
    arc_mod.add_function(wrap_pyfunction!(
        linearize_arc_from_array_py,
        arc_mod.clone()
    )?)?;
    shape_mod.add_submodule(&arc_mod)?;

    let bezier_mod = PyModule::new(py, "bezier")?;
    bezier_mod.add_function(wrap_pyfunction!(
        get_bezier_point_at_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod
        .add_function(wrap_pyfunction!(split_bezier_py, bezier_mod.clone())?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        get_bezier_bounds_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        get_bezier_rect_intersections_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        clip_bezier_with_rect_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        convert_cubic_bezier_to_quadratic_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        is_bezier_inside_polygons_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        linearize_bezier_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        linearize_bezier_adaptive_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        linearize_bezier_from_array_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        linearize_bezier_segment_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        flatten_bezier_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        bezier_flatness_sq_py,
        bezier_mod.clone()
    )?)?;
    bezier_mod.add_function(wrap_pyfunction!(
        perp_dist_sq_py,
        bezier_mod.clone()
    )?)?;
    shape_mod.add_submodule(&bezier_mod)?;

    let circle_mod = PyModule::new(py, "circle")?;
    circle_mod.add_function(wrap_pyfunction!(
        get_circle_circle_intersections_py,
        circle_mod.clone()
    )?)?;
    circle_mod.add_function(wrap_pyfunction!(
        is_circle_inside_rect_py,
        circle_mod.clone()
    )?)?;
    circle_mod.add_function(wrap_pyfunction!(
        does_circle_intersect_rect_py,
        circle_mod.clone()
    )?)?;
    circle_mod.add_function(wrap_pyfunction!(
        line_segment_intersects_circle_py,
        circle_mod.clone()
    )?)?;
    circle_mod.add_function(wrap_pyfunction!(
        project_point_onto_circle_py,
        circle_mod.clone()
    )?)?;
    shape_mod.add_submodule(&circle_mod)?;

    let polygon_mod = PyModule::new(py, "polygon")?;
    polygon_mod.add_function(wrap_pyfunction!(
        clean_polygon_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        is_almost_equal_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        normalize_polygons_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        translate_bounds_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        translate_polygons_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        point_line_distance_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        get_polygon_area_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        get_polygon_signed_area_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        get_polygon_perimeter_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        get_polygon_bounds_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        get_polygon_group_bounds_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        get_polygon_centroid_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        is_polygon_convex_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        get_polygon_convex_hull_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        get_polygon_edges_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        is_point_inside_polygon_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        offset_polygon_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        get_polygons_union_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        get_polygons_intersection_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        get_polygons_difference_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        polygons_intersect_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        flip_polygon_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        flip_polygons_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        rotate_polygon_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        rotate_polygons_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        scale_polygon_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        translate_polygon_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        polygon_area_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        polygon_bounds_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        polygon_perimeter_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        polygon_group_bounds_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        flip_polygon_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        flip_polygons_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        normalize_polygons_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        point_in_polygon_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        polygons_intersect_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        rotate_polygon_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        rotate_polygons_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        translate_polygon_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        translate_polygons_numpy_py,
        polygon_mod.clone()
    )?)?;
    polygon_mod.add_function(wrap_pyfunction!(
        to_clipper_numpy_py,
        polygon_mod.clone()
    )?)?;
    shape_mod.add_submodule(&polygon_mod)?;

    let line_mod = PyModule::new(py, "line")?;
    line_mod.add_function(wrap_pyfunction!(
        get_line_line_intersection_py,
        line_mod.clone()
    )?)?;
    line_mod.add_function(wrap_pyfunction!(
        get_line_segment_intersection_py,
        line_mod.clone()
    )?)?;
    line_mod.add_function(wrap_pyfunction!(
        get_line_closest_point_py,
        line_mod.clone()
    )?)?;
    line_mod.add_function(wrap_pyfunction!(
        get_line_segment_closest_point_py,
        line_mod.clone()
    )?)?;
    line_mod.add_function(wrap_pyfunction!(
        get_point_line_distance_py,
        line_mod.clone()
    )?)?;
    line_mod.add_function(wrap_pyfunction!(
        is_point_on_line_segment_py,
        line_mod.clone()
    )?)?;
    line_mod.add_function(wrap_pyfunction!(
        does_line_segment_intersect_rect_py,
        line_mod.clone()
    )?)?;
    line_mod.add_function(wrap_pyfunction!(
        does_line_segment_intersect_circle_py,
        line_mod.clone()
    )?)?;
    line_mod.add_function(wrap_pyfunction!(
        get_line_segment_polygon_intersections_py,
        line_mod.clone()
    )?)?;
    shape_mod.add_submodule(&line_mod)?;

    let rect_mod = PyModule::new(py, "rect")?;
    rect_mod.add_function(wrap_pyfunction!(
        is_point_inside_rect_py,
        rect_mod.clone()
    )?)?;
    rect_mod.add_function(wrap_pyfunction!(
        does_rect_contain_rect_py,
        rect_mod.clone()
    )?)?;
    rect_mod.add_function(wrap_pyfunction!(
        does_rect_intersect_rect_py,
        rect_mod.clone()
    )?)?;
    shape_mod.add_submodule(&rect_mod)?;

    let point_mod = PyModule::new(py, "point")?;
    point_mod
        .add_function(wrap_pyfunction!(midpoint_py, point_mod.clone())?)?;
    shape_mod.add_submodule(&point_mod)?;

    parent.add_submodule(&shape_mod)?;

    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("rayforge.core.geo.shape", shape_mod)?;
    sys_modules.set_item("rayforge.core.geo.shape.arc", arc_mod)?;
    sys_modules.set_item("rayforge.core.geo.shape.bezier", bezier_mod)?;
    sys_modules.set_item("rayforge.core.geo.shape.circle", circle_mod)?;
    sys_modules.set_item("rayforge.core.geo.shape.polygon", polygon_mod)?;
    sys_modules.set_item("rayforge.core.geo.shape.line", line_mod)?;
    sys_modules.set_item("rayforge.core.geo.shape.rect", rect_mod)?;
    sys_modules.set_item("rayforge.core.geo.shape.point", point_mod)?;

    Ok(())
}

#[pyfunction(name = "get_arc_bounds")]
#[pyo3(signature = (start, end, center, clockwise))]
fn get_arc_bounds_py(
    start: Point,
    end: Point,
    center: Point,
    clockwise: bool,
) -> (f64, f64, f64, f64) {
    get_arc_bounds(start, end, center, clockwise)
}

#[pyfunction(name = "get_arc_direction")]
fn get_arc_direction_py(center: Point, start: Point, mouse: Point) -> bool {
    get_arc_direction(center, start, mouse)
}

#[pyfunction(name = "get_arc_closest_point")]
fn get_arc_closest_point_py(
    arc_cmd: &Bound<'_, PyAny>,
    start_pos: (f64, f64, f64),
    x: f64,
    y: f64,
) -> PyResult<Option<(f64, Point, f64)>> {
    let arr = _arc_row_from_any(arc_cmd)?;
    Ok(get_arc_closest_point(&arr, start_pos, x, y))
}

#[pyfunction(name = "get_arc_midpoint")]
#[pyo3(signature = (start, end, center, clockwise))]
fn get_arc_midpoint_py(
    start: Point,
    end: Point,
    center: Point,
    clockwise: bool,
) -> Point {
    get_arc_midpoint(start, end, center, clockwise)
}

#[pyfunction(name = "get_arc_angles")]
#[pyo3(signature = (start, end, center, clockwise))]
fn get_arc_angles_py(
    start: Point,
    end: Point,
    center: Point,
    clockwise: bool,
) -> (f64, f64, f64) {
    get_arc_angles(start, end, center, clockwise)
}

#[pyfunction(name = "does_arc_intersect_rect")]
#[pyo3(signature = (arc_start, arc_end, arc_center, clockwise, rect))]
fn does_arc_intersect_rect_py(
    arc_start: Point,
    arc_end: Point,
    arc_center: Point,
    clockwise: bool,
    rect: (f64, f64, f64, f64),
) -> bool {
    does_arc_intersect_rect(arc_start, arc_end, arc_center, clockwise, rect)
}

#[pyfunction(name = "does_arc_intersect_circle")]
#[pyo3(signature = (arc_start, arc_end, arc_center, clockwise, circle_center, circle_radius))]
fn does_arc_intersect_circle_py(
    arc_start: Point,
    arc_end: Point,
    arc_center: Point,
    clockwise: bool,
    circle_center: Point,
    circle_radius: f64,
) -> bool {
    does_arc_intersect_circle(
        arc_start,
        arc_end,
        arc_center,
        clockwise,
        circle_center,
        circle_radius,
    )
}

#[pyfunction(name = "is_arc_clockwise")]
fn is_arc_clockwise_py(points: Vec<PyPoint2D>, center: PyPoint2D) -> bool {
    let points_2d: Vec<(f64, f64)> = points.iter().map(|p| (p.0, p.1)).collect();
    is_arc_clockwise(&points_2d, (center.0, center.1))
}

#[pyfunction(name = "is_arc_inside_polygons")]
#[pyo3(signature = (arc_start, arc_end, arc_center, clockwise, polygons))]
fn is_arc_inside_polygons_py(
    arc_start: Point,
    arc_end: Point,
    arc_center: Point,
    clockwise: bool,
    polygons: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let polygons_2d = extract_polygons(polygons)?;
    Ok(is_arc_inside_polygons(arc_start, arc_end, arc_center, clockwise, &polygons_2d))
}

#[pyfunction(name = "is_angle_between")]
#[pyo3(signature = (angle, start, end, clockwise))]
fn is_angle_between_py(angle: f64, start: f64, end: f64, clockwise: bool) -> bool {
    is_angle_between(angle, start, end, clockwise)
}

#[pyfunction(name = "normalize_angle")]
fn normalize_angle_py(angle: f64) -> f64 {
    normalize_angle(angle)
}

#[pyfunction(name = "linearize_arc")]
#[pyo3(signature = (arc_cmd, start_point, resolution=0.1))]
fn linearize_arc_py(
    arc_cmd: &Bound<'_, PyAny>,
    start_point: (f64, f64, f64),
    resolution: f64,
) -> PyResult<Vec<Segment3D>> {
    let arr = _arc_row_from_any(arc_cmd)?;
    Ok(linearize_arc(&arr, start_point, resolution))
}

#[pyfunction(name = "linearize_arc_from_array")]
fn linearize_arc_from_array_py(
    data: Vec<f64>,
    start_point: (f64, f64, f64),
    max_seg_length: f64,
) -> Vec<Vec<f64>> {
    let mut arr = [0.0; 8];
    let len = data.len().min(8);
    arr[..len].copy_from_slice(&data[..len]);
    let result = _linearize_arc_from_array(&arr, start_point, max_seg_length);
    result.into_iter().map(|(p1, p2)| vec![p1.0, p1.1, p1.2, p2.0, p2.1, p2.2]).collect()
}

#[pyfunction(name = "get_bezier_point_at")]
fn get_bezier_point_at_py(
    p0: PyPoint2D,
    p1: PyPoint2D,
    p2: PyPoint2D,
    p3: PyPoint2D,
    t: f64,
) -> Point {
    get_bezier_point_at((p0.0, p0.1), (p1.0, p1.1), (p2.0, p2.1), (p3.0, p3.1), t)
}

#[pyfunction(name = "split_bezier")]
fn split_bezier_py(
    p0: PyPoint2D,
    p1: PyPoint2D,
    p2: PyPoint2D,
    p3: PyPoint2D,
    t: f64,
) -> BezierSplit {
    split_bezier((p0.0, p0.1), (p1.0, p1.1), (p2.0, p2.1), (p3.0, p3.1), t)
}

#[pyfunction(name = "get_bezier_bounds")]
fn get_bezier_bounds_py(
    p0: PyPoint2D,
    p1: PyPoint2D,
    p2: PyPoint2D,
    p3: PyPoint2D,
) -> (f64, f64, f64, f64) {
    get_bezier_bounds((p0.0, p0.1), (p1.0, p1.1), (p2.0, p2.1), (p3.0, p3.1))
}

#[pyfunction(name = "get_bezier_rect_intersections")]
fn get_bezier_rect_intersections_py(
    p0: PyPoint2D,
    p1: PyPoint2D,
    p2: PyPoint2D,
    p3: PyPoint2D,
    rect: (f64, f64, f64, f64),
) -> Vec<f64> {
    get_bezier_rect_intersections((p0.0, p0.1), (p1.0, p1.1), (p2.0, p2.1), (p3.0, p3.1), rect)
}

#[pyfunction(name = "clip_bezier_with_rect")]
fn clip_bezier_with_rect_py(
    p0: PyPoint2D,
    p1: PyPoint2D,
    p2: PyPoint2D,
    p3: PyPoint2D,
    rect: (f64, f64, f64, f64),
) -> Vec<(Point, Point, Point, Point)> {
    clip_bezier_with_rect((p0.0, p0.1), (p1.0, p1.1), (p2.0, p2.1), (p3.0, p3.1), rect)
}

#[pyfunction(name = "convert_cubic_bezier_to_quadratic")]
fn convert_cubic_bezier_to_quadratic_py(
    p0: PyPoint2D,
    p1: PyPoint2D,
    p2: PyPoint2D,
    p3: PyPoint2D,
) -> (Point, Point, Point) {
    convert_cubic_bezier_to_quadratic((p0.0, p0.1), (p1.0, p1.1), (p2.0, p2.1), (p3.0, p3.1))
}

#[pyfunction(name = "is_bezier_inside_polygons")]
fn is_bezier_inside_polygons_py(
    p0: PyPoint2D,
    p1: PyPoint2D,
    p2: PyPoint2D,
    p3: PyPoint2D,
    polygons: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let polygons_2d = extract_polygons(polygons)?;
    Ok(is_bezier_inside_polygons(
        (p0.0, p0.1), (p1.0, p1.1), (p2.0, p2.1), (p3.0, p3.1),
        &polygons_2d,
    ))
}

#[pyfunction(name = "linearize_bezier")]
fn linearize_bezier_py(
    p0: PyPoint3D,
    p1: PyPoint3D,
    p2: PyPoint3D,
    p3: PyPoint3D,
    num_steps: usize,
) -> Vec<((f64, f64, f64), (f64, f64, f64))> {
    let result = linearize_bezier((p0.0, p0.1, p0.2), (p1.0, p1.1, p1.2), (p2.0, p2.1, p2.2), (p3.0, p3.1, p3.2), num_steps);
    result
}

#[pyfunction(name = "linearize_bezier_adaptive")]
#[pyo3(signature = (p0, p1, p2, p3, tolerance_sq, max_subdivisions=20))]
fn linearize_bezier_adaptive_py(
    p0: PyPoint2D,
    p1: PyPoint2D,
    p2: PyPoint2D,
    p3: PyPoint2D,
    tolerance_sq: f64,
    max_subdivisions: usize,
) -> Vec<(f64, f64)> {
    linearize_bezier_adaptive((p0.0, p0.1), (p1.0, p1.1), (p2.0, p2.1), (p3.0, p3.1), tolerance_sq, max_subdivisions)
}

#[pyfunction(name = "linearize_bezier_from_array")]
fn linearize_bezier_from_array_py(
    bezier_row: Vec<f64>,
    start_point: (f64, f64, f64),
    max_seg_length: f64,
) -> Vec<Vec<f64>> {
    let mut arr = [0.0; 8];
    let len = bezier_row.len().min(8);
    arr[..len].copy_from_slice(&bezier_row[..len]);
    let result = linearize_bezier_from_array(&arr, start_point, max_seg_length);
    result.into_iter().map(|(p1, p2)| vec![p1.0, p1.1, p1.2, p2.0, p2.1, p2.2]).collect()
}

#[pyfunction(name = "linearize_bezier_segment")]
#[pyo3(signature = (p0, p1, p2, p3, tolerance=0.1))]
fn linearize_bezier_segment_py(
    p0: PyPoint3D,
    p1: PyPoint3D,
    p2: PyPoint3D,
    p3: PyPoint3D,
    tolerance: f64,
) -> Vec<(f64, f64, f64)> {
    let result = linearize_bezier_segment(
        (p0.0, p0.1, p0.2),
        (p1.0, p1.1, p1.2),
        (p2.0, p2.1, p2.2),
        (p3.0, p3.1, p3.2),
        Some(tolerance),
    );
    result
}

#[pyfunction(name = "flatten_bezier")]
fn flatten_bezier_py(
    p0: PyPoint3D,
    p1: PyPoint3D,
    p2: PyPoint3D,
    p3: PyPoint3D,
    tolerance: f64,
    max_subdivisions: usize,
    pts: &Bound<'_, PyList>,
) -> PyResult<()> {
    let mut result = Vec::new();
    flatten_bezier(
        (p0.0, p0.1, p0.2),
        (p1.0, p1.1, p1.2),
        (p2.0, p2.1, p2.2),
        (p3.0, p3.1, p3.2),
        tolerance,
        max_subdivisions,
        &mut result,
    );
    let py = pts.py();
    for p in result {
        let obj = (p.0, p.1, p.2).into_pyobject(py)?;
        pts.append(&obj)?;
    }
    Ok(())
}

#[pyfunction(name = "bezier_flatness_sq")]
fn bezier_flatness_sq_py(
    a: PyPoint2D,
    b: PyPoint2D,
    c: PyPoint2D,
    d: PyPoint2D,
) -> f64 {
    let a3 = (a.0, a.1, 0.0);
    let b3 = (b.0, b.1, 0.0);
    let c3 = (c.0, c.1, 0.0);
    let d3 = (d.0, d.1, 0.0);
    bezier_flatness_sq(a3, b3, c3, d3)
}

#[pyfunction(name = "perp_dist_sq")]
#[pyo3(signature = (pt, origin, vx, vy, vz=0.0, norm_sq=0.0))]
fn perp_dist_sq_py(
    pt: PyPoint3D,
    origin: PyPoint3D,
    vx: f64,
    vy: f64,
    vz: f64,
    norm_sq: f64,
) -> f64 {
    perp_dist_sq((pt.0, pt.1, pt.2), (origin.0, origin.1, origin.2), vx, vy, vz, norm_sq)
}

#[pyfunction(name = "get_circle_circle_intersections")]
fn get_circle_circle_intersections_py(
    c1: Point,
    r1: f64,
    c2: Point,
    r2: f64,
) -> Vec<Point> {
    get_circle_circle_intersections(c1, r1, c2, r2)
}

#[pyfunction(name = "is_circle_inside_rect")]
fn is_circle_inside_rect_py(
    center: Point,
    radius: f64,
    rect: (f64, f64, f64, f64),
) -> bool {
    is_circle_inside_rect(center, radius, rect)
}

#[pyfunction(name = "does_circle_intersect_rect")]
fn does_circle_intersect_rect_py(
    center: Point,
    radius: f64,
    rect: (f64, f64, f64, f64),
) -> bool {
    does_circle_intersect_rect(center, radius, rect)
}

#[pyfunction(name = "line_segment_intersects_circle")]
fn line_segment_intersects_circle_py(
    p1: Point,
    p2: Point,
    circle_center: Point,
    circle_radius: f64,
) -> bool {
    line_segment_intersects_circle(p1, p2, circle_center, circle_radius)
}

#[pyfunction(name = "project_point_onto_circle")]
fn project_point_onto_circle_py(
    point: Point,
    center: Point,
    radius: f64,
) -> Option<Point> {
    project_point_onto_circle(point, center, radius)
}

#[pyfunction(name = "clean_polygon")]
#[pyo3(signature = (polygon, tolerance=None))]
fn clean_polygon_py(polygon: Vec<PyPoint2D>, tolerance: Option<f64>) -> Option<Vec<Point>> {
    clean_polygon(&poly_to_points(polygon), tolerance.unwrap_or(1e-6))
}

#[pyfunction(name = "is_almost_equal")]
#[pyo3(signature = (a, b, tolerance=None))]
fn is_almost_equal_py(a: f64, b: f64, tolerance: Option<f64>) -> bool {
    is_almost_equal(a, b, tolerance.unwrap_or(1e-9))
}

#[pyfunction(name = "normalize_polygons")]
fn normalize_polygons_py(polygons: &Bound<'_, PyAny>) -> PyResult<(Vec<Vec<Point>>, f64, f64)> {
    let p = extract_polygons(polygons)?;
    Ok(normalize_polygons(&p))
}

#[pyfunction(name = "translate_bounds")]
fn translate_bounds_py(bounds: (f64, f64, f64, f64), dx: f64, dy: f64) -> (f64, f64, f64, f64) {
    translate_bounds(bounds, dx, dy)
}

#[pyfunction(name = "translate_polygons")]
fn translate_polygons_py(polygons: &Bound<'_, PyAny>, dx: f64, dy: f64) -> PyResult<Vec<Vec<Point>>> {
    let p = extract_polygons(polygons)?;
    Ok(translate_polygons(&p, dx, dy))
}

#[pyfunction(name = "point_line_distance")]
fn point_line_distance_py(point: Point, line_start: Point, line_end: Point) -> f64 {
    point_line_distance(point, line_start, line_end)
}

#[pyfunction(name = "get_polygon_area")]
fn get_polygon_area_py(polygon: Vec<PyPoint2D>) -> f64 {
    get_polygon_signed_area(&poly_to_points(polygon))
}

#[pyfunction(name = "get_polygon_signed_area")]
fn get_polygon_signed_area_py(polygon: Vec<PyPoint2D>) -> f64 {
    get_polygon_signed_area(&poly_to_points(polygon))
}

#[pyfunction(name = "get_polygon_perimeter")]
fn get_polygon_perimeter_py(polygon: Vec<PyPoint2D>) -> f64 {
    get_polygon_perimeter(&poly_to_points(polygon))
}

#[pyfunction(name = "get_polygon_bounds")]
fn get_polygon_bounds_py(polygon: Vec<PyPoint2D>) -> (f64, f64, f64, f64) {
    get_polygon_bounds(&poly_to_points(polygon))
}

#[pyfunction(name = "get_polygon_group_bounds")]
fn get_polygon_group_bounds_py(
    polygons: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64, f64, f64)> {
    let p = extract_polygons(polygons)?;
    Ok(get_polygon_group_bounds(&p))
}

#[pyfunction(name = "get_polygon_centroid")]
fn get_polygon_centroid_py(polygon: Vec<PyPoint2D>) -> Point {
    get_polygon_centroid(&poly_to_points(polygon))
}

#[pyfunction(name = "is_polygon_convex")]
fn is_polygon_convex_py(polygon: Vec<PyPoint2D>) -> bool {
    is_polygon_convex(&poly_to_points(polygon))
}

#[pyfunction(name = "get_polygon_convex_hull")]
fn get_polygon_convex_hull_py(polygon: Vec<PyPoint2D>) -> Vec<Point> {
    get_polygon_convex_hull(&poly_to_points(polygon))
}

#[pyfunction(name = "get_polygon_edges")]
fn get_polygon_edges_py(polygon: Vec<PyPoint2D>) -> Vec<(Point, Point)> {
    get_polygon_edges(&poly_to_points(polygon))
}

#[pyfunction(name = "is_point_inside_polygon")]
fn is_point_inside_polygon_py(point: Point, polygon: Vec<PyPoint2D>) -> bool {
    is_point_inside_polygon(point, &poly_to_points(polygon))
}

#[pyfunction(name = "offset_polygon")]
fn offset_polygon_py(polygon: Vec<PyPoint2D>, offset: f64) -> Vec<Vec<Point>> {
    offset_polygon(&poly_to_points(polygon), offset)
}

#[pyfunction(name = "get_polygons_union")]
fn get_polygons_union_py(polygons: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Point>>> {
    let p = extract_polygons(polygons)?;
    Ok(get_polygons_union(&p))
}

#[pyfunction(name = "get_polygons_intersection")]
fn get_polygons_intersection_py(
    poly1: Vec<PyPoint2D>,
    poly2: Vec<PyPoint2D>,
) -> Vec<Vec<Point>> {
    get_polygons_intersection(&poly_to_points(poly1), &poly_to_points(poly2))
}

#[pyfunction(name = "get_polygons_difference")]
fn get_polygons_difference_py(
    poly1: Vec<PyPoint2D>,
    poly2: Vec<PyPoint2D>,
) -> Vec<Vec<Point>> {
    get_polygons_difference(&poly_to_points(poly1), &poly_to_points(poly2))
}

#[pyfunction(name = "polygons_intersect")]
#[pyo3(signature = (p1, p2, min_area=0.0))]
fn polygons_intersect_py(
    p1: Vec<PyPoint2D>,
    p2: Vec<PyPoint2D>,
    min_area: f64,
) -> bool {
    polygons_intersect(&poly_to_points(p1), &poly_to_points(p2), min_area)
}

#[pyfunction(name = "flip_polygon")]
fn flip_polygon_py(
    polygon: Vec<PyPoint2D>,
    flip_h: bool,
    flip_v: bool,
) -> Vec<Point> {
    flip_polygon(&poly_to_points(polygon), flip_h, flip_v)
}

#[pyfunction(name = "flip_polygons")]
fn flip_polygons_py(
    polygons: &Bound<'_, PyAny>,
    flip_h: bool,
    flip_v: bool,
) -> PyResult<Vec<Vec<Point>>> {
    let p = extract_polygons(polygons)?;
    Ok(flip_polygons(&p, flip_h, flip_v))
}

#[pyfunction(name = "rotate_polygon")]
fn rotate_polygon_py(polygon: Vec<PyPoint2D>, angle: f64) -> Vec<Point> {
    rotate_polygon(&poly_to_points(polygon), angle)
}

#[pyfunction(name = "rotate_polygons")]
fn rotate_polygons_py(
    polygons: &Bound<'_, PyAny>,
    angle: f64,
) -> PyResult<Vec<Vec<Point>>> {
    let p = extract_polygons(polygons)?;
    Ok(rotate_polygons(&p, angle))
}

#[pyfunction(name = "scale_polygon")]
#[pyo3(signature = (polygon, scale, scale_y=None))]
fn scale_polygon_py(polygon: Vec<PyPoint2D>, scale: f64, scale_y: Option<f64>) -> Vec<Point> {
    scale_polygon(&poly_to_points(polygon), scale, scale_y)
}

#[pyfunction(name = "translate_polygon")]
fn translate_polygon_py(polygon: Vec<PyPoint2D>, dx: f64, dy: f64) -> Vec<Point> {
    translate_polygon(&poly_to_points(polygon), dx, dy)
}

// -- numpy wrapper helpers --

fn _polygon_from_numpy(arr: &Bound<'_, PyArray2<f64>>) -> Vec<(f64, f64)> {
    let readonly = arr.readonly();
    let view = readonly.as_array();
    view.rows().into_iter()
        .map(|row| (row[0], row[1]))
        .collect()
}

fn _polygon_to_numpy(py: Python<'_>, poly: Vec<(f64, f64)>) -> Py<PyAny> {
    let vecs: Vec<Vec<f64>> = poly.into_iter().map(|(x, y)| vec![x, y]).collect();
    let np_arr = PyArray2::<f64>::from_vec2(py, &vecs).expect("failed to create numpy array");
    np_arr.into_any().unbind()
}

fn _polygons_from_numpy_list(polys: Vec<Bound<'_, PyArray2<f64>>>) -> Vec<Vec<(f64, f64)>> {
    polys.into_iter().map(|a| _polygon_from_numpy(&a)).collect()
}

fn _polygons_to_numpy_list(py: Python<'_>, polys: Vec<Vec<(f64, f64)>>) -> Vec<Py<PyAny>> {
    polys.into_iter().map(|p| _polygon_to_numpy(py, p)).collect()
}

#[pyfunction(name = "polygon_area_numpy")]
fn polygon_area_numpy_py(polygon: Bound<'_, PyArray2<f64>>) -> f64 {
    let p = _polygon_from_numpy(&polygon);
    get_polygon_signed_area(&p)
}

#[pyfunction(name = "polygon_bounds_numpy")]
fn polygon_bounds_numpy_py(polygon: Bound<'_, PyArray2<f64>>) -> (f64, f64, f64, f64) {
    let p = _polygon_from_numpy(&polygon);
    get_polygon_bounds(&p)
}

#[pyfunction(name = "polygon_perimeter_numpy")]
fn polygon_perimeter_numpy_py(polygon: Bound<'_, PyArray2<f64>>) -> f64 {
    let p = _polygon_from_numpy(&polygon);
    get_polygon_perimeter(&p)
}

#[pyfunction(name = "polygon_group_bounds_numpy")]
fn polygon_group_bounds_numpy_py(
    polygons: Vec<Bound<'_, PyArray2<f64>>>,
) -> (f64, f64, f64, f64) {
    let p = _polygons_from_numpy_list(polygons);
    get_polygon_group_bounds(&p)
}

#[pyfunction(name = "flip_polygon_numpy")]
fn flip_polygon_numpy_py(
    py: Python<'_>,
    polygon: Bound<'_, PyArray2<f64>>,
    flip_h: bool,
    flip_v: bool,
) -> Py<PyAny> {
    let p = _polygon_from_numpy(&polygon);
    let result = flip_polygon(&p, flip_h, flip_v);
    _polygon_to_numpy(py, result)
}

#[pyfunction(name = "flip_polygons_numpy")]
fn flip_polygons_numpy_py<'py>(
    py: Python<'py>,
    polygons: Bound<'py, PyList>,
    flip_h: bool,
    flip_v: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if !flip_h && !flip_v {
        return Ok(polygons.as_any().clone());
    }
    let mut p = Vec::new();
    for item in polygons.iter() {
        let arr = item.cast::<PyArray2<f64>>()?;
        p.push(_polygon_from_numpy(&arr));
    }
    let result = flip_polygons(&p, flip_h, flip_v);
    let np_list = _polygons_to_numpy_list(py, result);
    Ok(PyList::new(py, np_list)?.as_any().clone())
}

#[pyfunction(name = "normalize_polygons_numpy")]
fn normalize_polygons_numpy_py(
    py: Python<'_>,
    polygons: Vec<Bound<'_, PyArray2<f64>>>,
) -> (Vec<Py<PyAny>>, f64, f64) {
    let p = _polygons_from_numpy_list(polygons);
    let (result, min_x, min_y) = normalize_polygons(&p);
    let result_np = _polygons_to_numpy_list(py, result);
    (result_np, min_x, min_y)
}

#[pyfunction(name = "point_in_polygon_numpy")]
fn point_in_polygon_numpy_py(
    point: (f64, f64),
    polygon: Bound<'_, PyArray2<f64>>,
) -> bool {
    let p = _polygon_from_numpy(&polygon);
    is_point_inside_polygon(point, &p)
}

#[pyfunction(name = "polygons_intersect_numpy")]
#[pyo3(signature = (poly1, poly2, min_area=0.0))]
fn polygons_intersect_numpy_py(
    poly1: Bound<'_, PyArray2<f64>>,
    poly2: Bound<'_, PyArray2<f64>>,
    min_area: f64,
) -> bool {
    let p1 = _polygon_from_numpy(&poly1);
    let p2 = _polygon_from_numpy(&poly2);
    polygons_intersect(&p1, &p2, min_area)
}

#[pyfunction(name = "rotate_polygon_numpy")]
fn rotate_polygon_numpy_py(
    py: Python<'_>,
    polygon: Bound<'_, PyArray2<f64>>,
    angle: f64,
) -> Py<PyAny> {
    let p = _polygon_from_numpy(&polygon);
    let result = rotate_polygon(&p, angle);
    _polygon_to_numpy(py, result)
}

#[pyfunction(name = "rotate_polygons_numpy")]
fn rotate_polygons_numpy_py(
    py: Python<'_>,
    polygons: Vec<Bound<'_, PyArray2<f64>>>,
    angle: f64,
) -> Vec<Py<PyAny>> {
    let p = _polygons_from_numpy_list(polygons);
    let result = rotate_polygons(&p, angle);
    _polygons_to_numpy_list(py, result)
}

#[pyfunction(name = "translate_polygon_numpy")]
fn translate_polygon_numpy_py(
    py: Python<'_>,
    polygon: Bound<'_, PyArray2<f64>>,
    dx: f64,
    dy: f64,
) -> Py<PyAny> {
    let p = _polygon_from_numpy(&polygon);
    let result = translate_polygon(&p, dx, dy);
    _polygon_to_numpy(py, result)
}

#[pyfunction(name = "translate_polygons_numpy")]
fn translate_polygons_numpy_py(
    py: Python<'_>,
    polygons: Vec<Bound<'_, PyArray2<f64>>>,
    dx: f64,
    dy: f64,
) -> Vec<Py<PyAny>> {
    let p = _polygons_from_numpy_list(polygons);
    let result = translate_polygons(&p, dx, dy);
    _polygons_to_numpy_list(py, result)
}

#[pyfunction(name = "to_clipper_numpy")]
#[pyo3(signature = (polygon, scale=10_000_000))]
fn to_clipper_numpy_py(
    polygon: &Bound<'_, PyAny>,
    scale: i64,
) -> PyResult<Vec<(i64, i64)>> {
    let points: Vec<(f64, f64)> = if let Ok(arr) =
        polygon.extract::<Bound<'_, PyArray2<f64>>>()
    {
        let readonly = arr.readonly();
        (0..readonly.shape()[0])
            .map(|i| (*readonly.get([i, 0]).unwrap(), *readonly.get([i, 1]).unwrap()))
            .collect()
    } else if let Ok(pts) = polygon.extract::<Vec<(f64, f64)>>() {
        pts
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "polygon must be an (N, 2) numpy array or list of (x, y) tuples",
        ));
    };
    Ok(to_clipper_from_points(&points, scale as f64))
}

#[pyfunction(name = "get_line_line_intersection")]
fn get_line_line_intersection_py(
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
) -> Option<Point> {
    get_line_line_intersection(p1, p2, p3, p4)
}

#[pyfunction(name = "get_line_segment_intersection")]
fn get_line_segment_intersection_py(
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
) -> Option<Point> {
    get_line_segment_intersection(p1, p2, p3, p4)
}

#[pyfunction(name = "get_line_closest_point")]
fn get_line_closest_point_py(
    line_p1: Point,
    line_p2: Point,
    x: f64,
    y: f64,
) -> Point {
    get_line_closest_point(line_p1, line_p2, x, y)
}

#[pyfunction(name = "get_line_segment_closest_point")]
fn get_line_segment_closest_point_py(
    seg_p1: Point,
    seg_p2: Point,
    x: f64,
    y: f64,
) -> (f64, Point, f64) {
    get_line_segment_closest_point(seg_p1, seg_p2, x, y)
}

#[pyfunction(name = "get_point_line_distance")]
fn get_point_line_distance_py(
    point: Point,
    line_p1: Point,
    line_p2: Point,
) -> f64 {
    get_point_line_distance(point, line_p1, line_p2)
}

#[pyfunction(name = "is_point_on_line_segment")]
fn is_point_on_line_segment_py(
    point: Point,
    seg_p1: Point,
    seg_p2: Point,
) -> bool {
    is_point_on_segment(point, seg_p1, seg_p2)
}

#[pyfunction(name = "does_line_segment_intersect_rect")]
fn does_line_segment_intersect_rect_py(
    p1: Point,
    p2: Point,
    rect: (f64, f64, f64, f64),
) -> bool {
    does_line_segment_intersect_rect(p1, p2, rect)
}

#[pyfunction(name = "does_line_segment_intersect_circle")]
fn does_line_segment_intersect_circle_py(
    p1: Point,
    p2: Point,
    circle_center: Point,
    circle_radius: f64,
) -> bool {
    does_line_segment_intersect_circle(p1, p2, circle_center, circle_radius)
}

#[pyfunction(name = "get_line_segment_polygon_intersections")]
fn get_line_segment_polygon_intersections_py(
    p1: Point,
    p2: Point,
    polygon: Vec<Vec<Point>>,
) -> Vec<f64> {
    get_line_segment_polygon_intersections(p1, p2, &polygon)
}

#[pyfunction(name = "is_point_inside_rect")]
fn is_point_inside_rect_py(point: Point, rect: (f64, f64, f64, f64)) -> bool {
    is_point_inside_rect(point, rect)
}

#[pyfunction(name = "does_rect_contain_rect")]
fn does_rect_contain_rect_py(
    outer: (f64, f64, f64, f64),
    inner: (f64, f64, f64, f64),
) -> bool {
    does_rect_contain_rect(outer, inner)
}

#[pyfunction(name = "does_rect_intersect_rect")]
fn does_rect_intersect_rect_py(
    r1: (f64, f64, f64, f64),
    r2: (f64, f64, f64, f64),
) -> bool {
    use rayforge_geo::shape::line::does_rect_intersect_rect;
    does_rect_intersect_rect(r1, r2)
}

#[pyfunction(name = "midpoint")]
fn midpoint_py(p1: PyPoint3D, p2: PyPoint3D) -> (f64, f64, f64) {
    let p1_3d = (p1.0, p1.1, p1.2);
    let p2_3d = (p2.0, p2.1, p2.2);
    let result = midpoint(p1_3d, p2_3d);
    (result.0, result.1, result.2)
}
