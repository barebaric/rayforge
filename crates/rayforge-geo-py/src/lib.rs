use pyo3::prelude::*;

mod algo;
mod flex_point;
mod geometry;
mod path;
mod shape;
use crate::flex_point::{PyIntPoint2D, int_poly_to_points, extract_polygon, extract_polygons};
use crate::geometry::Geometry;
use rayforge_geo::algo::clipping::clip_line_segment_with_polygons;
use rayforge_geo::algo::fitting::fit_points_with_primitives;
use rayforge_geo::shape::arc::is_arc_inside_polygons;
use rayforge_geo::shape::bezier::is_bezier_inside_polygons;
use rayforge_geo::Segment3D;
use rayforge_geo::{
    CMD_TYPE_ARC, CMD_TYPE_BEZIER, CMD_TYPE_LINE, CMD_TYPE_MOVE, COL_C1X,
    COL_C1Y, COL_C2X, COL_C2Y, COL_CW, COL_I, COL_J, COL_TYPE, COL_X,
    COL_Y, COL_Z, GEO_ARRAY_COLS, Point, Point3D,
};

const CLIPPER_SCALE: i64 = 10_000_000;

#[pymodule(gil_used = false)]
fn geo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__all__",
        vec![
            "Geometry",
            "CMD_TYPE_MOVE",
            "CMD_TYPE_LINE",
            "CMD_TYPE_ARC",
            "CMD_TYPE_BEZIER",
            "COL_TYPE",
            "COL_X",
            "COL_Y",
            "COL_Z",
            "COL_I",
            "COL_J",
            "COL_CW",
            "COL_C1X",
            "COL_C1Y",
            "COL_C2X",
            "COL_C2Y",
            "GEO_ARRAY_COLS",
            "CLIPPER_SCALE",
            "Point",
            "Point3D",
            "Rect",
            "Polygon",
            "IntPolygon",
            "IntPoint",
            "Edge",
            "CubicBezier",
            "Point2DOr3D",
            "Polygon3D",
            "Rect3D",
        ],
    )?;

    add_functions(m)?;
    add_type_aliases(m)?;
    add_submodules(m)?;
    Ok(())
}

fn add_submodules(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    shape::build_shape_module(py, m)?;
    algo::build_algo_module(py, m)?;
    path::build_path_module(py, m)?;

    let types_mod = py.import("types")?;
    let constants = types_mod.call_method0("SimpleNamespace")?;
    {
        use rayforge_geo::{
            CMD_TYPE_ARC, CMD_TYPE_BEZIER, CMD_TYPE_LINE, CMD_TYPE_MOVE,
            COL_C1X, COL_C1Y, COL_C2X, COL_C2Y, COL_CW, COL_I, COL_J, COL_TYPE,
            COL_X, COL_Y, COL_Z, GEO_ARRAY_COLS,
        };
        constants.setattr("CMD_TYPE_MOVE", CMD_TYPE_MOVE)?;
        constants.setattr("CMD_TYPE_LINE", CMD_TYPE_LINE)?;
        constants.setattr("CMD_TYPE_ARC", CMD_TYPE_ARC)?;
        constants.setattr("CMD_TYPE_BEZIER", CMD_TYPE_BEZIER)?;
        constants.setattr("COL_TYPE", COL_TYPE)?;
        constants.setattr("COL_X", COL_X)?;
        constants.setattr("COL_Y", COL_Y)?;
        constants.setattr("COL_Z", COL_Z)?;
        constants.setattr("COL_I", COL_I)?;
        constants.setattr("COL_J", COL_J)?;
        constants.setattr("COL_CW", COL_CW)?;
        constants.setattr("COL_C1X", COL_C1X)?;
        constants.setattr("COL_C1Y", COL_C1Y)?;
        constants.setattr("COL_C2X", COL_C2X)?;
        constants.setattr("COL_C2Y", COL_C2Y)?;
        constants.setattr("GEO_ARRAY_COLS", GEO_ARRAY_COLS)?;
    }
    m.add("constants", constants)?;

    Ok(())
}

fn add_type_aliases(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let typing = py.import("typing")?;
    let tuple_type = typing.getattr("Tuple")?;
    let list_type = typing.getattr("List")?;
    let union_type = typing.getattr("Union")?;
    let float_type = py.get_type::<pyo3::types::PyFloat>();
    let int_type = py.import("builtins")?.getattr("int")?;

    let point =
        tuple_type.get_item((float_type.clone(), float_type.clone()))?;
    let point3d = tuple_type.get_item((
        float_type.clone(),
        float_type.clone(),
        float_type.clone(),
    ))?;
    let rect = tuple_type.get_item((
        float_type.clone(),
        float_type.clone(),
        float_type.clone(),
        float_type.clone(),
    ))?;
    let polygon = list_type.get_item(point.clone())?;
    let polygon3d = list_type.get_item(point3d.clone())?;
    let int_point = tuple_type.get_item((int_type.clone(), int_type))?;
    let int_polygon = list_type.get_item(int_point.clone())?;
    let edge = tuple_type.get_item((point.clone(), point.clone()))?;
    let cubic_bezier = tuple_type.get_item((
        point.clone(),
        point.clone(),
        point.clone(),
        point.clone(),
    ))?;
    let point_2d_or_3d =
        union_type.get_item((point.clone(), point3d.clone()))?;

    m.add("Point", point)?;
    m.add("Point3D", point3d)?;
    m.add("Rect", rect)?;
    m.add("Polygon", polygon)?;
    m.add("Polygon3D", polygon3d)?;
    m.add("IntPoint", int_point)?;
    m.add("IntPolygon", int_polygon)?;
    m.add("Edge", edge)?;
    m.add("CubicBezier", cubic_bezier)?;
    m.add("Point2DOr3D", point_2d_or_3d)?;

    let collections = py.import("collections")?;
    let rect3d = collections.call_method1(
        "namedtuple",
        (
            "Rect3D",
            vec!["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
        ),
    )?;
    m.add("Rect3D", rect3d)?;

    Ok(())
}

fn add_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CMD_TYPE_MOVE", CMD_TYPE_MOVE)?;
    m.add("CMD_TYPE_LINE", CMD_TYPE_LINE)?;
    m.add("CMD_TYPE_ARC", CMD_TYPE_ARC)?;
    m.add("CMD_TYPE_BEZIER", CMD_TYPE_BEZIER)?;
    m.add("COL_TYPE", COL_TYPE)?;
    m.add("COL_X", COL_X)?;
    m.add("COL_Y", COL_Y)?;
    m.add("COL_Z", COL_Z)?;
    m.add("COL_I", COL_I)?;
    m.add("COL_J", COL_J)?;
    m.add("COL_CW", COL_CW)?;
    m.add("COL_C1X", COL_C1X)?;
    m.add("COL_C1Y", COL_C1Y)?;
    m.add("COL_C2X", COL_C2X)?;
    m.add("COL_C2Y", COL_C2Y)?;
    m.add("GEO_ARRAY_COLS", GEO_ARRAY_COLS)?;
    m.add("CLIPPER_SCALE", CLIPPER_SCALE)?;

    m.add_function(wrap_pyfunction!(to_clipper, m)?)?;
    m.add_function(wrap_pyfunction!(from_clipper, m)?)?;
    m.add_function(wrap_pyfunction!(clip_line_segment_with_polygons_top, m)?)?;
    m.add_function(wrap_pyfunction!(is_arc_inside_polygons_top, m)?)?;
    m.add_function(wrap_pyfunction!(is_bezier_inside_polygons_top, m)?)?;
    m.add_function(wrap_pyfunction!(fit_points_with_primitives_top, m)?)?;

    m.add_class::<Geometry>()?;

    Ok(())
}

#[pyfunction(name = "clip_line_segment_with_polygons")]
fn clip_line_segment_with_polygons_top(
    p1: Point3D,
    p2: Point3D,
    regions: &Bound<'_, PyAny>,
) -> PyResult<Vec<Segment3D>> {
    let regions = extract_polygons(regions)?;
    Ok(clip_line_segment_with_polygons(p1, p2, &regions))
}

#[pyfunction(name = "is_arc_inside_polygons")]
fn is_arc_inside_polygons_top(
    arc_start: Point,
    arc_end: Point,
    arc_center: Point,
    clockwise: bool,
    polygons: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let polygons_2d = extract_polygons(polygons)?;
    Ok(is_arc_inside_polygons(arc_start, arc_end, arc_center, clockwise, &polygons_2d))
}

#[pyfunction(name = "is_bezier_inside_polygons")]
fn is_bezier_inside_polygons_top(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    polygons: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let polygons_2d = extract_polygons(polygons)?;
    Ok(is_bezier_inside_polygons(p0, p1, p2, p3, &polygons_2d))
}

#[pyfunction(name = "fit_points_with_primitives")]
fn fit_points_with_primitives_top(
    points: Vec<Point3D>,
    tolerance: f64,
) -> Vec<Vec<f64>> {
    fit_points_with_primitives(&points, tolerance)
        .into_iter()
        .map(|r| r.to_vec())
        .collect()
}

#[pyfunction]
fn to_clipper(polygon: &Bound<'_, PyAny>, scale: Option<i64>) -> PyResult<Vec<(i64, i64)>> {
    let scale = scale.unwrap_or(CLIPPER_SCALE);
    let poly = extract_polygon(polygon)?;
    Ok(poly
        .iter()
        .map(|(x, y)| ((x * scale as f64) as i64, (y * scale as f64) as i64))
        .collect())
}

#[pyfunction]
fn from_clipper(polygon: Vec<PyIntPoint2D>, scale: Option<i64>) -> Vec<Point> {
    let scale = scale.unwrap_or(CLIPPER_SCALE) as f64;
    let poly = int_poly_to_points(polygon);
    poly
        .iter()
        .map(|(x, y)| (*x as f64 / scale, *y as f64 / scale))
        .collect()
}