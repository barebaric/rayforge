use clipper2::{
    difference as clipper_difference, intersect as clipper_intersect,
    simplify as clipper_simplify, union as clipper_union, EndType, FillRule,
    JoinType, Path as GeoPath, Paths as GeoPaths, Point as GeoPoint,
    PointInPolygonResult, PointScaler,
};

use crate::types::{Edge, IntPolygon, Point, Polygon, Rect};

/// Custom point scaler matching Python's CLIPPER_SCALE = 10^7.
#[derive(Debug, Default, Clone, Copy, PartialEq, Hash)]
pub struct GeoScale;

impl PointScaler for GeoScale {
    const MULTIPLIER: f64 = 10_000_000.0;
}

/// A clipper2 path using our custom GeoScale.
pub type ClipperPath = GeoPath<GeoScale>;

/// A clipper2 paths collection using our custom GeoScale.
pub type ClipperPaths = GeoPaths<GeoScale>;

pub fn is_almost_equal(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

pub fn polygon_to_path(polygon: &Polygon) -> ClipperPath {
    ClipperPath::from(polygon.clone())
}

pub fn path_to_polygon(path: &ClipperPath) -> Polygon {
    <Vec<(f64, f64)>>::from(path.clone())
}

pub fn paths_to_polygons(paths: &ClipperPaths) -> Vec<Polygon> {
    <Vec<Vec<(f64, f64)>>>::from(paths.clone())
}

pub fn polygons_to_paths(polygons: &[Polygon]) -> ClipperPaths {
    let v: Vec<Vec<(f64, f64)>> = polygons.to_vec();
    ClipperPaths::from(v)
}

/// Calculate the signed area of a polygon using the shoelace formula.
pub fn get_polygon_signed_area(polygon: &Polygon) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    let n = polygon.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += polygon[i].0 * polygon[j].1;
        area -= polygon[j].0 * polygon[i].1;
    }
    area / 2.0
}

/// Calculate the absolute area of a polygon.
pub fn get_polygon_area(polygon: &Polygon) -> f64 {
    get_polygon_signed_area(polygon).abs()
}

/// Calculate the perimeter of a polygon.
pub fn get_polygon_perimeter(polygon: &Polygon) -> f64 {
    if polygon.len() < 2 {
        return 0.0;
    }
    let n = polygon.len();
    let mut perimeter = 0.0;
    for i in 0..n {
        let p1 = polygon[i];
        let p2 = polygon[(i + 1) % n];
        let dx = p2.0 - p1.0;
        let dy = p2.1 - p1.1;
        perimeter += (dx * dx + dy * dy).sqrt();
    }
    perimeter
}

/// Perpendicular distance from a point to a line segment.
pub fn point_line_distance(
    point: Point,
    line_start: Point,
    line_end: Point,
) -> f64 {
    let line_vec = (line_end.0 - line_start.0, line_end.1 - line_start.1);
    let line_len = (line_vec.0 * line_vec.0 + line_vec.1 * line_vec.1).sqrt();
    if line_len < 1e-6 {
        let dx = point.0 - line_start.0;
        let dy = point.1 - line_start.1;
        return (dx * dx + dy * dy).sqrt();
    }
    let line_unit = (line_vec.0 / line_len, line_vec.1 / line_len);
    let point_vec = (point.0 - line_start.0, point.1 - line_start.1);
    let mut proj_len = point_vec.0 * line_unit.0 + point_vec.1 * line_unit.1;
    proj_len = proj_len.max(0.0).min(line_len);
    let closest_x = line_start.0 + proj_len * line_unit.0;
    let closest_y = line_start.1 + proj_len * line_unit.1;
    let dx = point.0 - closest_x;
    let dy = point.1 - closest_y;
    (dx * dx + dy * dy).sqrt()
}

/// Extract all edges from a polygon as (start, end) point pairs.
pub fn get_polygon_edges(polygon: &Polygon) -> Vec<Edge> {
    if polygon.len() < 2 {
        return vec![];
    }
    let n = polygon.len();
    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        edges.push((polygon[i], polygon[(i + 1) % n]));
    }
    edges
}

/// Get the bounding box of a polygon as (min_x, min_y, max_x, max_y).
pub fn get_polygon_bounds(polygon: &Polygon) -> Rect {
    if polygon.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut min_x = polygon[0].0;
    let mut max_x = polygon[0].0;
    let mut min_y = polygon[0].1;
    let mut max_y = polygon[0].1;
    for &(x, y) in polygon {
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }
    (min_x, min_y, max_x, max_y)
}

/// Calculate bounds of an integer polygon.
pub fn int_get_polygon_bounds(polygon: &IntPolygon) -> (i64, i64, i64, i64) {
    if polygon.is_empty() {
        return (0, 0, 0, 0);
    }
    let mut min_x = polygon[0].0;
    let mut max_x = polygon[0].0;
    let mut min_y = polygon[0].1;
    let mut max_y = polygon[0].1;
    for &(x, y) in polygon {
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }
    (min_x, min_y, max_x, max_y)
}

/// Get the bounding box of multiple polygons.
pub fn get_polygon_group_bounds(polygons: &[Polygon]) -> Rect {
    if polygons.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    let mut has_points = false;
    for poly in polygons {
        for &(x, y) in poly {
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
            has_points = true;
        }
    }
    if !has_points {
        return (0.0, 0.0, 0.0, 0.0);
    }
    (min_x, min_y, max_x, max_y)
}

/// Translate a bounding box by a given offset.
pub fn translate_bounds(bounds: Rect, dx: f64, dy: f64) -> Rect {
    (bounds.0 + dx, bounds.1 + dy, bounds.2 + dx, bounds.3 + dy)
}

/// Normalize polygons so their minimum corner is at the origin.
pub fn normalize_polygons(polygons: &[Polygon]) -> (Vec<Polygon>, f64, f64) {
    if polygons.is_empty() {
        return (polygons.to_vec(), 0.0, 0.0);
    }
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    for poly in polygons {
        for &(x, y) in poly {
            if x < min_x {
                min_x = x;
            }
            if y < min_y {
                min_y = y;
            }
        }
    }
    if min_x == f64::MAX {
        return (polygons.to_vec(), 0.0, 0.0);
    }
    let normalized: Vec<Polygon> = polygons
        .iter()
        .map(|p| translate_polygon(p, -min_x, -min_y))
        .collect();
    (normalized, min_x, min_y)
}

/// Flip a polygon horizontally and/or vertically.
pub fn flip_polygon(polygon: &Polygon, flip_h: bool, flip_v: bool) -> Polygon {
    polygon
        .iter()
        .map(|&(x, y)| {
            (if flip_h { -x } else { x }, if flip_v { -y } else { y })
        })
        .collect()
}

/// Flip multiple polygons horizontally and/or vertically.
pub fn flip_polygons(
    polygons: &[Polygon],
    flip_h: bool,
    flip_v: bool,
) -> Vec<Polygon> {
    polygons
        .iter()
        .map(|p| flip_polygon(p, flip_h, flip_v))
        .collect()
}

/// Calculate the centroid of a polygon.
pub fn get_polygon_centroid(polygon: &Polygon) -> Point {
    if polygon.is_empty() {
        return (0.0, 0.0);
    }
    let n = polygon.len();
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut signed_area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let cross = polygon[i].0 * polygon[j].1 - polygon[j].0 * polygon[i].1;
        signed_area += cross;
        cx += (polygon[i].0 + polygon[j].0) * cross;
        cy += (polygon[i].1 + polygon[j].1) * cross;
    }
    signed_area /= 2.0;
    if signed_area.abs() < 1e-9 {
        let sum_x: f64 = polygon.iter().map(|p| p.0).sum();
        let sum_y: f64 = polygon.iter().map(|p| p.1).sum();
        return (sum_x / n as f64, sum_y / n as f64);
    }
    cx /= 6.0 * signed_area;
    cy /= 6.0 * signed_area;
    (cx, cy)
}

/// Rotate a polygon around the origin.
pub fn rotate_polygon(polygon: &Polygon, angle_degrees: f64) -> Polygon {
    if polygon.is_empty() {
        return polygon.clone();
    }
    let angle_rad = angle_degrees.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    polygon
        .iter()
        .map(|&(x, y)| (x * cos_a - y * sin_a, x * sin_a + y * cos_a))
        .collect()
}

/// Rotate multiple polygons around the origin.
pub fn rotate_polygons(
    polygons: &[Polygon],
    angle_degrees: f64,
) -> Vec<Polygon> {
    polygons
        .iter()
        .map(|p| rotate_polygon(p, angle_degrees))
        .collect()
}

/// Translate a polygon by a given offset.
pub fn translate_polygon(polygon: &Polygon, dx: f64, dy: f64) -> Polygon {
    polygon.iter().map(|&(x, y)| (x + dx, y + dy)).collect()
}

/// Translate multiple polygons by a given offset.
pub fn translate_polygons(
    polygons: &[Polygon],
    dx: f64,
    dy: f64,
) -> Vec<Polygon> {
    polygons
        .iter()
        .map(|p| translate_polygon(p, dx, dy))
        .collect()
}

/// Scale a polygon.
pub fn scale_polygon(polygon: &Polygon, sx: f64, sy: Option<f64>) -> Polygon {
    let sy = sy.unwrap_or(sx);
    polygon.iter().map(|&(x, y)| (x * sx, y * sy)).collect()
}

fn cross(o: Point, a: Point, b: Point) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Check if a polygon is convex.
pub fn is_polygon_convex(polygon: &Polygon) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    if n == 3 {
        return true;
    }
    let mut sign: Option<bool> = None;
    for i in 0..n {
        let c = cross(polygon[i], polygon[(i + 1) % n], polygon[(i + 2) % n]);
        if c.abs() < 1e-10 {
            continue;
        }
        match sign {
            None => sign = Some(c > 0.0),
            Some(s) if (c > 0.0) != s => return false,
            _ => {}
        }
    }
    true
}

/// Compute the convex hull of a polygon using Andrew's monotone chain.
pub fn get_polygon_convex_hull(polygon: &Polygon) -> Polygon {
    if polygon.len() < 3 {
        return polygon.clone();
    }
    let mut sorted = polygon.clone();
    sorted
        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut lower: Vec<Point> = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0
        {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Point> = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0
        {
            upper.pop();
        }
        upper.push(p);
    }
    lower[..lower.len() - 1]
        .iter()
        .chain(&upper[..upper.len() - 1])
        .copied()
        .collect()
}

/// Clean a polygon by removing duplicate and near-collinear points.
pub fn clean_polygon(polygon: &Polygon, tolerance: f64) -> Option<Polygon> {
    if polygon.len() < 3 {
        return None;
    }
    let path = polygon_to_path(polygon);
    let paths = ClipperPaths::from(path);
    let simplified = clipper_simplify(paths, 0.0, false);
    if simplified.is_empty() {
        return None;
    }
    let mut biggest = simplified.get(0).unwrap().clone();
    let mut biggest_area = biggest.signed_area().abs();
    for i in 1..simplified.len() {
        let area = simplified.get(i).unwrap().signed_area().abs();
        if area > biggest_area {
            biggest = simplified.get(i).unwrap().clone();
            biggest_area = area;
        }
    }
    let clean_tol = GeoScale::scale(tolerance);
    let cleaned_paths = clipper_simplify(
        ClipperPaths::from(biggest),
        clean_tol / GeoScale::MULTIPLIER,
        false,
    );
    let cleaned = match cleaned_paths.get(0) {
        Some(p) => p.clone(),
        None => return None,
    };
    let mut result = path_to_polygon(&cleaned);
    if result.len() > 1 {
        let first = result[0];
        let last = result[result.len() - 1];
        if (first.0 - last.0).abs() < 1e-9 && (first.1 - last.1).abs() < 1e-9 {
            result.pop();
        }
    }
    if result.len() < 3 {
        return None;
    }
    Some(result)
}

/// Offset (inflate/deflate) a polygon.
pub fn offset_polygon(polygon: &Polygon, offset: f64) -> Vec<Polygon> {
    if polygon.len() < 3 {
        return vec![];
    }
    if offset.abs() < 1e-9 {
        return vec![polygon.clone()];
    }
    let path = polygon_to_path(polygon);
    let result = path.inflate(offset, JoinType::Miter, EndType::Polygon, 2.0);
    let mut output = Vec::new();
    for p in result.iter() {
        let poly = path_to_polygon(p);
        if poly.len() >= 3 {
            output.push(poly);
        }
    }
    output
}

/// Compute the union of multiple polygons.
pub fn get_polygons_union(polygons: &[Polygon]) -> Vec<Polygon> {
    if polygons.is_empty() {
        return vec![];
    }
    let clipper_paths = polygons_to_paths(polygons);
    if clipper_paths.is_empty() {
        return vec![];
    }
    let result =
        clipper_union(clipper_paths.clone(), clipper_paths, FillRule::NonZero)
            .unwrap_or_default();
    paths_to_polygons(&result)
        .into_iter()
        .filter(|p| p.len() >= 3)
        .collect()
}

/// Compute the intersection of two polygons.
pub fn get_polygons_intersection(
    poly1: &Polygon,
    poly2: &Polygon,
) -> Vec<Polygon> {
    if poly1.len() < 3 || poly2.len() < 3 {
        return vec![];
    }
    let path1 = polygons_to_paths(std::slice::from_ref(poly1));
    let path2 = polygons_to_paths(std::slice::from_ref(poly2));
    let result =
        clipper_intersect(path1, path2, FillRule::NonZero).unwrap_or_default();
    paths_to_polygons(&result)
        .into_iter()
        .filter(|p| p.len() >= 3)
        .collect()
}

/// Compute the difference of two polygons (poly1 - poly2).
pub fn get_polygons_difference(
    poly1: &Polygon,
    poly2: &Polygon,
) -> Vec<Polygon> {
    if poly1.len() < 3 || poly2.len() < 3 {
        if poly1.len() >= 3 {
            return vec![poly1.clone()];
        }
        return vec![];
    }
    let path1 = polygons_to_paths(std::slice::from_ref(poly1));
    let path2 = polygons_to_paths(std::slice::from_ref(poly2));
    let result =
        clipper_difference(path1, path2, FillRule::NonZero).unwrap_or_default();
    paths_to_polygons(&result)
        .into_iter()
        .filter(|p| p.len() >= 3)
        .collect()
}

/// Tests if a point is inside a polygon using the ray casting algorithm.
/// Uses a bounding box early-out for performance, and handles edge cases
/// where the point lies exactly on a polygon edge.
pub fn is_point_in_polygon(point: Point, polygon: &Polygon) -> bool {
    let (x, y) = point;
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut min_x = polygon[0].0;
    let mut max_x = polygon[0].0;
    let mut min_y = polygon[0].1;
    let mut max_y = polygon[0].1;

    for &(px, py) in polygon {
        if px < min_x {
            min_x = px;
        } else if px > max_x {
            max_x = px;
        }
        if py < min_y {
            min_y = py;
        } else if py > max_y {
            max_y = py;
        }
    }

    if x < min_x || x > max_x || y < min_y || y > max_y {
        return false;
    }

    for i in 0..n {
        let p1 = polygon[i];
        let p2 = polygon[(i + 1) % n];
        let (p1x, p1y) = p1;
        let (p2x, p2y) = p2;

        let cross_product = (y - p1y) * (p2x - p1x) - (x - p1x) * (p2y - p1y);
        if cross_product.abs() < 1e-9
            && p1x.min(p2x) <= x
            && x <= p1x.max(p2x)
            && p1y.min(p2y) <= y
            && y <= p1y.max(p2y)
        {
            return true;
        }
    }

    let mut inside = false;
    let (mut p1x, mut p1y) = polygon[0];
    for i in 0..=n {
        let (p2x, p2y) = polygon[i % n];
        if p1y == p2y {
            p1x = p2x;
            p1y = p2y;
            continue;
        }

        if p1y.min(p2y) < y && y <= p1y.max(p2y) {
            let x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x;
            if x_intersect > x {
                inside = !inside;
            }
        }
        p1x = p2x;
        p1y = p2y;
    }

    inside
}

/// Alias for is_point_in_polygon for compatibility.
pub fn is_point_inside_polygon(point: Point, polygon: &Polygon) -> bool {
    is_point_in_polygon(point, polygon)
}

/// Check if a point is inside a polygon using clipper2.
pub fn point_in_polygon_clipper(point: Point, polygon: &Polygon) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    // Check if point is on any edge (clipper2 does not count edge points as inside)
    for i in 0..polygon.len() {
        let j = (i + 1) % polygon.len();
        if super::line::is_point_on_segment(point, polygon[i], polygon[j]) {
            return true;
        }
    }
    let path = polygon_to_path(polygon);
    let geo_point = GeoPoint::<GeoScale>::new(point.0, point.1);
    path.is_point_inside(geo_point) == PointInPolygonResult::IsInside
}

/// Check if two polygons intersect.
pub fn polygons_intersect(
    poly1: &Polygon,
    poly2: &Polygon,
    min_area: f64,
) -> bool {
    if poly1.len() < 3 || poly2.len() < 3 {
        return false;
    }
    let intersection = get_polygons_intersection(poly1, poly2);
    if intersection.is_empty() {
        return false;
    }
    if min_area <= 0.0 {
        return true;
    }
    // min_area is specified in clipper integer coordinates (scale^2),
    // convert to float area for comparison
    let scale = 10_000_000.0;
    let min_area_float = min_area / (scale * scale);
    for poly in &intersection {
        if get_polygon_area(poly) > min_area_float {
            return true;
        }
    }
    false
}

pub fn to_clipper_from_points(points: &[(f64, f64)], scale: f64) -> IntPolygon {
    points
        .iter()
        .map(|(x, y)| ((*x * scale) as i64, (*y * scale) as i64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_geo_scale() {
        assert_eq!(GeoScale::MULTIPLIER, 10_000_000.0);
        let pt = GeoPoint::<GeoScale>::new(1.5, 2.75);
        assert_eq!(pt.x_scaled(), 15_000_000);
        assert_eq!(pt.y_scaled(), 27_500_000);
        assert_relative_eq!(pt.x(), 1.5);
        assert_relative_eq!(pt.y(), 2.75);
    }

    #[test]
    fn test_get_polygon_area() {
        // CCW square: area should be 1.0 (positive)
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert_relative_eq!(get_polygon_signed_area(&poly), 1.0);
        assert_relative_eq!(get_polygon_area(&poly), 1.0);

        // CW square: area should be -1.0
        let poly_cw = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
        assert_relative_eq!(get_polygon_signed_area(&poly_cw), -1.0);
        assert_relative_eq!(get_polygon_area(&poly_cw), 1.0);

        // Degenerate: less than 3 points
        assert_relative_eq!(
            get_polygon_signed_area(&vec![(0.0, 0.0), (1.0, 0.0)]),
            0.0
        );
    }

    #[test]
    fn test_get_polygon_perimeter() {
        // Unit square
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert_relative_eq!(get_polygon_perimeter(&poly), 4.0);

        // Triangle (3-4-5)
        let tri = vec![(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)];
        assert_relative_eq!(get_polygon_perimeter(&tri), 12.0);

        // Degenerate
        assert_relative_eq!(get_polygon_perimeter(&vec![(0.0, 0.0)]), 0.0);
    }

    #[test]
    fn test_point_line_distance() {
        let d = point_line_distance((0.0, 1.0), (0.0, 0.0), (1.0, 0.0));
        assert_relative_eq!(d, 1.0);

        // Point on the line
        let d = point_line_distance((0.5, 0.0), (0.0, 0.0), (1.0, 0.0));
        assert_relative_eq!(d, 0.0);

        // Degenerate segment
        let d = point_line_distance((1.0, 1.0), (0.0, 0.0), (0.0, 0.0));
        assert_relative_eq!(d, 2.0f64.sqrt());
    }

    #[test]
    fn test_get_polygon_edges() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        let edges = get_polygon_edges(&poly);
        assert_eq!(edges.len(), 3);
        assert_eq!(edges[0], ((0.0, 0.0), (1.0, 0.0)));
        assert_eq!(edges[1], ((1.0, 0.0), (1.0, 1.0)));
        assert_eq!(edges[2], ((1.0, 1.0), (0.0, 0.0)));
    }

    #[test]
    fn test_get_polygon_bounds() {
        let poly = vec![(2.0, 3.0), (5.0, 3.0), (5.0, 7.0), (2.0, 7.0)];
        assert_eq!(get_polygon_bounds(&poly), (2.0, 3.0, 5.0, 7.0));
        assert_eq!(get_polygon_bounds(&vec![]), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_get_polygon_group_bounds() {
        let polys = vec![
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            vec![(2.0, 2.0), (3.0, 2.0), (3.0, 3.0)],
        ];
        assert_eq!(get_polygon_group_bounds(&polys), (0.0, 0.0, 3.0, 3.0));
        assert_eq!(
            get_polygon_group_bounds(&[] as &[Polygon]),
            (0.0, 0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn test_translate_bounds() {
        assert_eq!(
            translate_bounds((1.0, 2.0, 3.0, 4.0), 1.0, 2.0),
            (2.0, 4.0, 4.0, 6.0)
        );
    }

    #[test]
    fn test_normalize_polygons() {
        let polys = vec![vec![(2.0, 3.0), (5.0, 3.0), (5.0, 7.0)]];
        let (normalized, dx, dy) = normalize_polygons(&polys);
        assert_relative_eq!(dx, 2.0);
        assert_relative_eq!(dy, 3.0);
        assert_eq!(normalized[0], vec![(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]);
    }

    #[test]
    fn test_get_polygon_centroid() {
        // Square centered at (0.5, 0.5)
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let c = get_polygon_centroid(&poly);
        assert_relative_eq!(c.0, 0.5, epsilon = 1e-10);
        assert_relative_eq!(c.1, 0.5, epsilon = 1e-10);

        // Empty
        assert_eq!(get_polygon_centroid(&vec![]), (0.0, 0.0));
    }

    #[test]
    fn test_rotate_polygon() {
        // Point on unit circle
        let poly = vec![(1.0, 0.0)];
        let rotated = rotate_polygon(&poly, 90.0);
        assert_relative_eq!(rotated[0].0, 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[0].1, 1.0, epsilon = 1e-10);

        // Square centered at origin
        let square = vec![(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];
        let rotated = rotate_polygon(&square, 90.0);
        // (1,1) -> (-1,1), (-1,1) -> (-1,-1), (-1,-1) -> (1,-1), (1,-1) -> (1,1)
        assert_relative_eq!(rotated[0].0, -1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[0].1, 1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1].0, -1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1].1, -1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[2].0, 1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[2].1, -1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[3].0, 1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[3].1, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_translate_polygon() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        let translated = translate_polygon(&poly, 2.0, 3.0);
        assert_eq!(translated, vec![(2.0, 3.0), (3.0, 3.0), (3.0, 4.0)]);
    }

    #[test]
    fn test_scale_polygon() {
        let poly = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)];
        let scaled = scale_polygon(&poly, 0.5, None);
        assert_eq!(scaled, vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]);

        let scaled_nonuniform = scale_polygon(&poly, 0.5, Some(2.0));
        assert_eq!(scaled_nonuniform, vec![(0.0, 0.0), (1.0, 0.0), (1.0, 4.0)]);
    }

    #[test]
    fn test_is_polygon_convex() {
        let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(is_polygon_convex(&square));

        // Concave
        let concave =
            vec![(0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (2.0, 2.0), (0.0, 2.0)];
        assert!(!is_polygon_convex(&concave));

        // Triangle is always convex
        assert!(is_polygon_convex(&vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]));

        // Degenerate
        assert!(!is_polygon_convex(&vec![(0.0, 0.0), (1.0, 1.0)]));
    }

    #[test]
    fn test_get_polygon_convex_hull() {
        let points =
            vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.0, 1.0)];
        let hull = get_polygon_convex_hull(&points);
        // Hull should be the square corners
        assert!(hull.len() == 4);
        assert!(hull.contains(&(0.0, 0.0)));
        assert!(hull.contains(&(1.0, 0.0)));
        assert!(hull.contains(&(1.0, 1.0)));
        assert!(hull.contains(&(0.0, 1.0)));
    }

    #[test]
    fn test_clean_polygon() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let cleaned = clean_polygon(&poly, 0.01);
        assert!(cleaned.is_some());
        let cleaned = cleaned.unwrap();
        assert!(cleaned.len() >= 3);
    }

    #[test]
    fn test_offset_polygon() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let expanded = offset_polygon(&poly, 0.1);
        assert!(!expanded.is_empty());
        assert!(get_polygon_area(&expanded[0]) > 1.0);
    }

    #[test]
    fn test_get_polygons_union() {
        let poly1 = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let poly2 = vec![(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)];
        let union = get_polygons_union(&[poly1, poly2]);
        assert!(!union.is_empty());
    }

    #[test]
    fn test_get_polygons_intersection() {
        let poly1 = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let poly2 = vec![(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)];
        let intersection = get_polygons_intersection(&poly1, &poly2);
        assert!(!intersection.is_empty());
        // Intersection should be a 1x1 square
        assert_relative_eq!(
            get_polygon_area(&intersection[0]),
            1.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_get_polygons_difference() {
        let poly1 = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let poly2 = vec![(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)];
        let diff = get_polygons_difference(&poly1, &poly2);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_point_in_polygon_clipper() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(point_in_polygon_clipper((0.5, 0.5), &poly));
        assert!(!point_in_polygon_clipper((2.0, 2.0), &poly));
    }

    #[test]
    fn test_polygons_intersect() {
        let poly1 = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let poly2 = vec![(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)];
        assert!(polygons_intersect(&poly1, &poly2, 0.0));

        let poly3 = vec![(10.0, 10.0), (11.0, 10.0), (11.0, 11.0)];
        assert!(!polygons_intersect(&poly1, &poly3, 0.0));
    }

    #[test]
    fn test_offset_polygon_zero() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let result = offset_polygon(&poly, 0.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 4);
    }

    #[test]
    fn test_offset_polygon_degenerate() {
        let result = offset_polygon(&vec![(0.0, 0.0), (1.0, 0.0)], 0.1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_clean_polygon_degenerate() {
        assert!(clean_polygon(&vec![(0.0, 0.0), (1.0, 0.0)], 0.01).is_none());
    }

    #[test]
    fn test_get_polygons_union_empty() {
        assert!(get_polygons_union(&[]).is_empty());
    }

    #[test]
    fn test_get_polygons_intersection_non_overlapping() {
        let poly1 = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let poly2 =
            vec![(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0)];
        assert!(get_polygons_intersection(&poly1, &poly2).is_empty());
    }

    #[test]
    fn test_get_polygons_difference_non_overlapping() {
        let poly1 = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let poly2 =
            vec![(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0)];
        let diff = get_polygons_difference(&poly1, &poly2);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0].len(), 4);
    }

    #[test]
    fn test_conversion_roundtrip() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let path = polygon_to_path(&poly);
        let back = path_to_polygon(&path);
        for (a, b) in poly.iter().zip(back.iter()) {
            assert_relative_eq!(a.0, b.0);
            assert_relative_eq!(a.1, b.1);
        }
    }

    #[test]
    fn test_translate_polygons() {
        let polys = vec![
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            vec![(2.0, 2.0), (3.0, 2.0), (3.0, 3.0)],
        ];
        let translated = translate_polygons(&polys, 1.0, 1.0);
        assert_eq!(translated[0][0], (1.0, 1.0));
        assert_eq!(translated[1][0], (3.0, 3.0));
    }

    #[test]
    fn test_rotate_polygons() {
        let polys = vec![vec![(0.0, 1.0)], vec![(1.0, 0.0)]];
        let rotated = rotate_polygons(&polys, 90.0);
        assert_relative_eq!(rotated[0][0].0, -1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[0][0].1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1][0].0, 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1][0].1, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_int_get_polygon_bounds() {
        let poly = vec![(2i64, 3i64), (5i64, 3i64), (5i64, 7i64)];
        assert_eq!(int_get_polygon_bounds(&poly), (2, 3, 5, 7));
        assert_eq!(int_get_polygon_bounds(&vec![]), (0, 0, 0, 0));
    }
}
