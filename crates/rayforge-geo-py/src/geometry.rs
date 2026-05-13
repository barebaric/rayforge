use numpy::ndarray;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyType};

use rayforge_geo::algo::fitting::convert_arc_to_beziers_from_array;
use rayforge_geo::path::analysis::get_point_and_tangent_at_from_array;
use rayforge_geo::path::transform::map_geometry_to_frame;
use rayforge_geo::{
    check_intersection_from_array, check_self_intersection_from_array,
    close_geometry_gaps_from_array, convert_arcs_to_beziers,
    find_closest_point_on_path_from_array,
    fit_curves, get_outward_normal_at_from_array, grow_geometry, linearize_data,
    remove_inner_edges,
    simplify_data, split_inner_and_outer_contours,
    split_into_components, split_into_contours,
    CMD_TYPE_ARC, CMD_TYPE_BEZIER, CMD_TYPE_LINE, CMD_TYPE_MOVE,
    COL_X, COL_Y, COL_Z, Command as CoreCommand, CommandRow,
    Geometry as CoreGeometry, Point,
};

#[pyclass(module = "geo.path", frozen, eq, skip_from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub enum PyCommand {
    Move { end: (f64, f64, f64) },
    Line { end: (f64, f64, f64) },
    Arc {
        end: (f64, f64, f64),
        center_offset: (f64, f64),
        clockwise: bool,
    },
    Bezier {
        end: (f64, f64, f64),
        control1: (f64, f64),
        control2: (f64, f64),
    },
}

impl From<CoreCommand> for PyCommand {
    fn from(cmd: CoreCommand) -> Self {
        match cmd {
            CoreCommand::Move { end } => PyCommand::Move { end },
            CoreCommand::Line { end } => PyCommand::Line { end },
            CoreCommand::Arc {
                end,
                center_offset,
                clockwise,
            } => PyCommand::Arc {
                end,
                center_offset,
                clockwise,
            },
            CoreCommand::Bezier {
                end,
                control1,
                control2,
            } => PyCommand::Bezier {
                end,
                control1,
                control2,
            },
        }
    }
}

#[derive(Clone)]
struct FlexPoint {
    x: f64,
    y: f64,
    z: f64,
}

impl<'a, 'py> FromPyObject<'a, 'py> for FlexPoint {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(p3) = ob.extract::<(f64, f64, f64)>() {
            Ok(FlexPoint {
                x: p3.0,
                y: p3.1,
                z: p3.2,
            })
        } else if let Ok(p2) = ob.extract::<(f64, f64)>() {
            Ok(FlexPoint {
                x: p2.0,
                y: p2.1,
                z: 0.0,
            })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "expected a 2-tuple or 3-tuple of floats",
            ))
        }
    }
}

#[pyclass(module = "geo.path", skip_from_py_object)]
#[derive(Clone)]
pub struct Geometry {
    pub(crate) inner: CoreGeometry,
}

#[pymethods]
impl Geometry {
    #[classattr]
    const COL_TYPE: usize = rayforge_geo::COL_TYPE;

    #[classattr]
    const COL_X: usize = rayforge_geo::COL_X;

    #[classattr]
    const COL_Y: usize = rayforge_geo::COL_Y;

    #[classattr]
    const COL_Z: usize = rayforge_geo::COL_Z;

    #[classattr]
    const COL_I: usize = rayforge_geo::COL_I;

    #[classattr]
    const COL_J: usize = rayforge_geo::COL_J;

    #[classattr]
    const COL_CW: usize = rayforge_geo::COL_CW;

    #[classattr]
    const COL_C1X: usize = rayforge_geo::COL_C1X;

    #[classattr]
    const COL_C1Y: usize = rayforge_geo::COL_C1Y;

    #[classattr]
    const COL_C2X: usize = rayforge_geo::COL_C2X;

    #[classattr]
    const COL_C2Y: usize = rayforge_geo::COL_C2Y;

    #[classattr]
    const CMD_TYPE_MOVE: f64 = rayforge_geo::CMD_TYPE_MOVE as f64;

    #[classattr]
    const CMD_TYPE_LINE: f64 = rayforge_geo::CMD_TYPE_LINE as f64;

    #[classattr]
    const CMD_TYPE_ARC: f64 = rayforge_geo::CMD_TYPE_ARC as f64;

    #[classattr]
    const CMD_TYPE_BEZIER: f64 = rayforge_geo::CMD_TYPE_BEZIER as f64;

    #[new]
    fn new() -> Self {
        Geometry {
            inner: CoreGeometry::new(),
        }
    }

    fn __reduce_ex__<'py>(
        slf: &Bound<'py, Self>,
        protocol: i32,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let _ = protocol;
        let mut borrowed = slf.borrow_mut();
        let data = borrowed.to_dict(py)?;
        let from_dict = slf.get_type().getattr("from_dict")?;
        pyo3::types::PyTuple::new(
            py,
            [from_dict.as_any(), pyo3::types::PyTuple::new(py, [data.as_any()])?.as_any()],
        )
    }

    #[pyo3(signature = (x, y, z=0.0))]
    fn move_to(&mut self, x: f64, y: f64, z: f64) {
        self.inner.move_to(x, y, z);
        self.inner.sync_to_data();
    }

    #[pyo3(signature = (x, y, z=0.0))]
    fn line_to(&mut self, x: f64, y: f64, z: f64) {
        self.inner.line_to(x, y, z);
        self.inner.sync_to_data();
    }

    fn close_path(&mut self) {
        self.inner.close_path();
        self.inner.sync_to_data();
    }

    #[pyo3(signature = (x, y, i=0.0, j=0.0, clockwise=true, z=0.0))]
    fn arc_to(
        &mut self,
        x: f64,
        y: f64,
        i: f64,
        j: f64,
        clockwise: bool,
        z: f64,
    ) {
        self.inner.arc_to(x, y, i, j, clockwise, z);
        self.inner.sync_to_data();
    }

    #[pyo3(signature = (x, y, c1x, c1y, c2x, c2y, z=0.0))]
    fn bezier_to(
        &mut self,
        x: f64,
        y: f64,
        c1x: f64,
        c1y: f64,
        c2x: f64,
        c2y: f64,
        z: f64,
    ) {
        self.inner.bezier_to(((c1x, c1y), (c2x, c2y), (x, y)), z);
        self.inner.sync_to_data();
    }

    fn sync_to_data(&mut self) {
        self.inner.sync_to_data();
    }

    fn __len__(&mut self) -> usize {
        self.inner.sync_to_data();
        self.inner.len()
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        for row in self.inner.data() {
            for &val in row.iter() {
                let normalized = if val == 0.0 { 0.0 } else { val };
                let bits = if normalized.is_nan() {
                    f64::NAN.to_bits()
                } else {
                    normalized.to_bits()
                };
                bits.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    #[getter]
    fn last_move_to(&self) -> (f64, f64, f64) {
        self.inner.last_move_to
    }

    #[setter]
    fn set_last_move_to(&mut self, value: (f64, f64, f64)) {
        self.inner.last_move_to = value;
    }

    #[getter]
    fn uniform_scalable(&self) -> bool {
        self.inner.uniform_scalable
    }

    #[setter(uniform_scalable)]
    fn set_uniform_scalable(&mut self, value: bool) {
        self.inner.uniform_scalable = value;
    }

    fn copy(&self) -> Self {
        Geometry {
            inner: self.inner.copy(),
        }
    }

    fn _get_last_point(&self) -> (f64, f64, f64) {
        let pending = self.inner.pending_data();
        if let Some(last) = pending.last() {
            return (last[COL_X], last[COL_Y], last[COL_Z]);
        }
        (0.0, 0.0, 0.0)
    }

    fn transform(slf: Bound<'_, Self>, matrix: Vec<Vec<f64>>) -> Bound<'_, Self> {
        {
            let mut geo = slf.borrow_mut();
            let mat: [[f64; 4]; 4] = [
                [matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]],
                [matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]],
                [matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]],
                [matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]],
            ];
            geo.inner.transform(&mat);
        }
        slf
    }

    fn extend(&mut self, other: &Geometry) {
        self.inner.extend(&other.inner);
    }

    fn rect(&mut self) -> (f64, f64, f64, f64) {
        self.inner.sync_to_data();
        self.inner.rect()
    }

    fn distance(&mut self) -> f64 {
        self.inner.sync_to_data();
        self.inner.distance()
    }

    fn area(&mut self) -> f64 {
        self.inner.sync_to_data();
        self.inner.area()
    }

    #[pyo3(signature = (tolerance=1e-6))]
    fn is_closed(&mut self, tolerance: f64) -> bool {
        self.inner.sync_to_data();
        self.inner.is_closed(tolerance)
    }

    fn segments(&mut self) -> Vec<Vec<(f64, f64, f64)>> {
        self.inner.sync_to_data();
        self.inner.segments()
    }

    #[getter]
    fn data<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let data = self.inner.synced_data();
        if data.is_empty() {
            return Ok(None);
        }
        let rows = data.len();
        let flat: Vec<f64> = data.iter().flatten().copied().collect();
        match ndarray::Array2::from_shape_vec((rows, 8usize), flat) {
            Ok(arr) => Ok(Some(arr.into_pyarray(py))),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("{}", e),
            )),
        }
    }

    #[setter(data)]
    fn set_data<'py>(
        &mut self,
        _py: Python<'py>,
        value: Option<Bound<'py, PyArray2<f64>>>,
    ) -> PyResult<()> {
        let data = self.inner.synced_data_mut();
        let Some(arr) = value else {
            data.clear();
            return Ok(());
        };
        let readonly = arr.readonly();
        let view = readonly.as_array();
        data.clear();
        for row in view.rows() {
            let mut chunk = [0.0; 8];
            let row_slice: &[f64] = row.as_slice().unwrap();
            chunk.copy_from_slice(row_slice);
            data.push(chunk);
        }
        Ok(())
    }

    #[getter]
    fn _pending_data(&self) -> Vec<Vec<f64>> {
        self.inner.pending_data().iter().map(|r| r.to_vec()).collect()
    }

    fn _sync_to_numpy(&mut self) {
        self.inner.sync_to_data();
    }

    fn get_command_at(&mut self, index: isize) -> Option<CommandRow> {
        if index < 0 {
            return None;
        }
        let data = self.inner.synced_data();
        data.get(index as usize)
            .map(|r| (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
    }

    fn iter_commands(&mut self) -> Vec<CommandRow> {
        let data = self.inner.synced_data();
        data.iter()
            .map(|r| (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
            .collect()
    }

    fn iter_typed_commands(&mut self) -> PyResult<Vec<PyCommand>> {
        let data = self.inner.synced_data();
        data.iter()
            .map(|r| {
                CoreCommand::from_row(r)
                    .map(PyCommand::from)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(e)
                    })
            })
            .collect()
    }

    fn get_typed_command_at(
        &mut self,
        index: isize,
    ) -> PyResult<Option<PyCommand>> {
        if index < 0 {
            return Ok(None);
        }
        let data = self.inner.synced_data();
        match data.get(index as usize) {
            Some(row) => Ok(Some(
                CoreCommand::from_row(row)
                    .map(PyCommand::from)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(e)
                    })?,
            )),
            None => Ok(None),
        }
    }

    fn dump<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let last_move_to = self.inner.last_move_to;
        let uniform_scalable = self.inner.uniform_scalable;
        let data = self.inner.synced_data();
        let dict = PyDict::new(py);
        dict.set_item(
            "last_move_to",
            vec![last_move_to.0, last_move_to.1, last_move_to.2],
        )?;
        dict.set_item("uniform_scalable", uniform_scalable)?;
        let commands = PyList::empty(py);
        for row in data {
            let cmd_type = row[0] as i32;
            let cmd = PyList::empty(py);
            if cmd_type == CMD_TYPE_MOVE as i32 {
                cmd.append("M")?;
                cmd.append(row[1])?;
                cmd.append(row[2])?;
                cmd.append(row[3])?;
            } else if cmd_type == CMD_TYPE_LINE as i32 {
                cmd.append("L")?;
                cmd.append(row[1])?;
                cmd.append(row[2])?;
                cmd.append(row[3])?;
            } else if cmd_type == CMD_TYPE_ARC as i32 {
                cmd.append("A")?;
                cmd.append(row[1])?;
                cmd.append(row[2])?;
                cmd.append(row[3])?;
                cmd.append(row[4])?;
                cmd.append(row[5])?;
                cmd.append(row[6])?;
            } else if cmd_type == CMD_TYPE_BEZIER as i32 {
                cmd.append("B")?;
                cmd.append(row[1])?;
                cmd.append(row[2])?;
                cmd.append(row[3])?;
                cmd.append(row[4])?;
                cmd.append(row[5])?;
                cmd.append(row[6])?;
                cmd.append(row[7])?;
            }
            commands.append(cmd)?;
        }
        dict.set_item("commands", commands)?;
        Ok(dict)
    }

    fn to_dict<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.dump(py)
    }

    #[classmethod]
    fn load<'py>(
        _cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyDict>,
    ) -> PyResult<Self> {
        let mut geo = Self::new();
        if let Some(lmt) = data.get_item("last_move_to")? {
            if let Ok(lmt_list) = lmt.extract::<Vec<f64>>() {
                if lmt_list.len() >= 3 {
                    geo.inner.last_move_to =
                        (lmt_list[0], lmt_list[1], lmt_list[2]);
                }
            }
        }
        if let Some(us) = data.get_item("uniform_scalable")? {
            if let Ok(val) = us.extract::<bool>() {
                geo.inner.uniform_scalable = val;
            }
        }
        if let Some(cmds) = data.get_item("commands")? {
            if let Ok(cmds_list) = cmds.cast::<PyList>() {
                for item in cmds_list.iter() {
                    if let Ok(cmd_list) = item.cast::<PyList>() {
                        let cmd_type: String = match cmd_list.get_item(0) {
                            Ok(val) => match val.extract::<String>() {
                                Ok(s) => s,
                                Err(_) => continue,
                            },
                            Err(_) => continue,
                        };
                        let x: f64 = match cmd_list.get_item(1) {
                            Ok(val) => match val.extract::<f64>() {
                                Ok(v) => v,
                                Err(_) => continue,
                            },
                            Err(_) => continue,
                        };
                        let y: f64 = match cmd_list.get_item(2) {
                            Ok(val) => match val.extract::<f64>() {
                                Ok(v) => v,
                                Err(_) => continue,
                            },
                            Err(_) => continue,
                        };
                        let z: f64 = match cmd_list.get_item(3) {
                            Ok(val) => match val.extract::<f64>() {
                                Ok(v) => v,
                                Err(_) => continue,
                            },
                            Err(_) => continue,
                        };
                        match cmd_type.as_str() {
                            "M" => geo.inner.move_to(x, y, z),
                            "L" => geo.inner.line_to(x, y, z),
                            "A" => {
                                if let (
                                    Some(i_val),
                                    Some(j_val),
                                    Some(cw_val),
                                ) =
                                    (
                                        cmd_list.get_item(4).ok().and_then(
                                            |v| v.extract::<f64>().ok(),
                                        ),
                                        cmd_list.get_item(5).ok().and_then(
                                            |v| v.extract::<f64>().ok(),
                                        ),
                                        cmd_list.get_item(6).ok().and_then(
                                            |v| v.extract::<f64>().ok(),
                                        ),
                                    )
                                {
                                    geo.inner.arc_to(
                                        x,
                                        y,
                                        i_val,
                                        j_val,
                                        cw_val > 0.5,
                                        z,
                                    );
                                }
                            }
                            "B" => {
                                if let (
                                    Some(c1x),
                                    Some(c1y),
                                    Some(c2x),
                                    Some(c2y),
                                ) =
                                    (
                                        cmd_list.get_item(4).ok().and_then(
                                            |v| v.extract::<f64>().ok(),
                                        ),
                                        cmd_list.get_item(5).ok().and_then(
                                            |v| v.extract::<f64>().ok(),
                                        ),
                                        cmd_list.get_item(6).ok().and_then(
                                            |v| v.extract::<f64>().ok(),
                                        ),
                                        cmd_list.get_item(7).ok().and_then(
                                            |v| v.extract::<f64>().ok(),
                                        ),
                                    )
                                {
                                    geo.inner.bezier_to(
                                        ((c1x, c1y), (c2x, c2y), (x, y)),
                                        z,
                                    );
                                }
                            }
                            _ => {}
                        }
                    } else if let Ok(cmd_dict) = item.cast::<PyDict>() {
                        let cmd_type: String = match cmd_dict
                            .get_item("type")
                        {
                            Ok(Some(val)) => match val.extract::<String>() {
                                Ok(s) => s,
                                Err(_) => continue,
                            },
                            _ => continue,
                        };
                        let end = match cmd_dict.get_item("end") {
                            Ok(Some(val)) => val.extract::<(f64, f64, f64)>(),
                            Err(_) => continue,
                            Ok(None) => continue,
                        };
                        let (x, y, z) = match end {
                            Ok(e) => e,
                            Err(_) => {
                                let end2 = match cmd_dict.get_item("end") {
                                    Ok(Some(val)) => {
                                        val.extract::<(f64, f64)>()
                                    }
                                    _ => continue,
                                };
                                match end2 {
                                    Ok((ex, ey)) => (ex, ey, 0.0),
                                    Err(_) => continue,
                                }
                            }
                        };
                        match cmd_type.as_str() {
                            "MoveToCommand" => geo.inner.move_to(x, y, z),
                            "LineToCommand" => geo.inner.line_to(x, y, z),
                            "CurveToCommand" | "BezierToCommand" => {
                                if let (Ok(Some(c1)), Ok(Some(c2))) = (
                                    cmd_dict.get_item("control1"),
                                    cmd_dict.get_item("control2"),
                                ) {
                                    let c1v = c1.extract::<(f64, f64)>();
                                    let c2v = c2.extract::<(f64, f64)>();
                                    if let (Ok((c1x, c1y)), Ok((c2x, c2y))) =
                                        (c1v, c2v)
                                    {
                                        geo.inner.bezier_to(
                                            ((c1x, c1y), (c2x, c2y), (x, y)),
                                            z,
                                        );
                                    }
                                }
                            }
                            "ArcToCommand" => {
                                let i_val: Option<(f64, f64)> =
                                    match cmd_dict.get_item("center") {
                                        Ok(Some(val)) => {
                                            val.extract::<(f64, f64)>().ok()
                                        }
                                        _ => None,
                                    }
                                    .or_else(|| {
                                        match cmd_dict
                                            .get_item("center_offset")
                                        {
                                            Ok(Some(val)) => {
                                                val.extract::<(f64, f64)>().ok()
                                            }
                                            _ => None,
                                        }
                                    });
                                let cw_val: Option<bool> = match cmd_dict
                                    .get_item("clockwise")
                                {
                                    Ok(Some(val)) => {
                                        val.extract::<bool>().ok()
                                    }
                                    _ => None,
                                };
                                if let (Some((ci, cj)), Some(cw)) =
                                    (i_val, cw_val)
                                {
                                    geo.inner.arc_to(
                                        x, y, ci, cj, cw, z,
                                    );
                                }
                            }
                            "ClosePathCommand" => geo.inner.close_path(),
                            _ => {}
                        }
                    }
                }
            }
        }
        geo.inner.sync_to_data();
        Ok(geo)
    }

    #[classmethod]
    fn from_dict<'py>(
        _cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyDict>,
    ) -> PyResult<Self> {
        Self::load(_cls, data)
    }

    #[classmethod]
    #[pyo3(signature = (points, close=true))]
    fn from_points<'py>(
        _cls: &Bound<'py, PyType>,
        points: &Bound<'py, PyAny>,
        close: bool,
    ) -> PyResult<Self> {
        let mut geo = Self::new();
        let points_vec: Vec<FlexPoint> = points
            .try_iter()?
            .map(|p| p?.extract::<FlexPoint>())
            .collect::<Result<Vec<_>, _>>()?;
        if points_vec.is_empty() {
            return Ok(geo);
        }
        let first = &points_vec[0];
        geo.move_to(first.x, first.y, first.z);
        for p in &points_vec[1..] {
            geo.line_to(p.x, p.y, p.z);
        }
        if close && points_vec.len() > 2 {
            geo.close_path();
        }
        geo.inner.sync_to_data();
        Ok(geo)
    }

    fn __eq__(&self, other: &Geometry) -> bool {
        self.inner == other.inner
    }

    fn simplify(&mut self, tolerance: f64) -> Self {
        let data = self.inner.synced_data();
        if data.len() > 2 {
            let simplified = simplify_data(data, tolerance);
            *self.inner.synced_data_mut() = simplified;
        }
        self.clone()
    }

    fn linearize(&mut self, tolerance: f64) -> Self {
        let data = self.inner.synced_data();
        if !data.is_empty() {
            let linearized = linearize_data(data, tolerance);
            *self.inner.synced_data_mut() = linearized;
        }
        self.clone()
    }

    #[pyo3(signature = (tolerance, beziers=true, arcs=true, on_progress=None))]
    fn fit_curves(
        &mut self,
        tolerance: f64,
        beziers: bool,
        arcs: bool,
        on_progress: Option<pyo3::Py<pyo3::PyAny>>,
    ) -> Self {
        let _ = on_progress;
        let data = self.inner.synced_data();
        if !data.is_empty() {
            let fitted = fit_curves(data, tolerance, beziers, arcs);
            *self.inner.synced_data_mut() = fitted;
        }
        self.clone()
    }

    fn fit_arcs(&mut self, tolerance: f64) -> Self {
        self.fit_curves(tolerance, false, true, None)
    }

    fn upgrade_to_scalable(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        {
            let mut geo = slf.borrow_mut();
            let data = geo.inner.synced_data();
            if !data.is_empty() {
                let converted = convert_arcs_to_beziers(data);
                *geo.inner.synced_data_mut() = converted;
                geo.inner.uniform_scalable = true;
            }
        }
        slf
    }

    #[pyo3(signature = (tolerance=None))]
    fn close_gaps(slf: Bound<'_, Self>, tolerance: Option<f64>) -> Bound<'_, Self> {
        {
            let mut geo = slf.borrow_mut();
            let data = geo.inner.synced_data();
            if !data.is_empty() {
                let closed =
                    close_geometry_gaps_from_array(data, tolerance.unwrap_or(0.5));
                *geo.inner.synced_data_mut() = closed;
            }
        }
        slf
    }

    fn cleanup(slf: Bound<'_, Self>, tolerance: f64) -> Bound<'_, Self> {
        {
            let mut geo = slf.borrow_mut();
            let data = geo.inner.synced_data();
            if !data.is_empty() {
                let cleaned = rayforge_geo::remove_duplicate_segments(
                    data,
                    tolerance,
                );
                *geo.inner.synced_data_mut() = cleaned;
            }
        }
        slf
    }

    fn append_data<'py>(
        &mut self,
        py: Python<'py>,
        rows: Option<Py<PyArray2<f64>>>,
    ) -> PyResult<()> {
        let Some(arr) = rows else {
            return Ok(());
        };
        let arr = arr.bind(py);
        let data = self.inner.synced_data_mut();
        let readonly: numpy::PyReadonlyArray<f64, numpy::ndarray::Ix2> =
            arr.readonly();
        let view: numpy::ndarray::ArrayView2<f64> = readonly.as_array();
        for row in view.rows() {
            let mut chunk = [0.0; 8];
            let row_slice: &[f64] = row.as_slice().unwrap();
            chunk.copy_from_slice(row_slice);
            data.push(chunk);
        }
        Ok(())
    }

    fn flip_x(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        {
            let mut geo = slf.borrow_mut();
            for row in geo.inner.synced_data_mut().iter_mut() {
                row[1] = -row[1];
            }
        }
        slf
    }

    fn flip_y(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        {
            let mut geo = slf.borrow_mut();
            for row in geo.inner.synced_data_mut().iter_mut() {
                row[2] = -row[2];
            }
        }
        slf
    }

    fn find_closest_point(
        &mut self,
        x: f64,
        y: f64,
    ) -> Option<(usize, f64, (f64, f64))> {
        let data = self.inner.synced_data();
        if data.is_empty() {
            return None;
        }
        find_closest_point_on_path_from_array(data, x, y)
    }

    fn get_point_and_tangent_at(
        &mut self,
        segment_index: usize,
        t: f64,
    ) -> Option<((f64, f64), (f64, f64))> {
        let data = self.inner.synced_data();
        if data.is_empty() {
            return None;
        }
            get_point_and_tangent_at_from_array(data, segment_index, t)
    }

    fn get_outward_normal_at(
        &mut self,
        segment_index: usize,
        t: f64,
    ) -> Option<(f64, f64)> {
        let data = self.inner.synced_data();
        if data.is_empty() {
            return None;
        }
        get_outward_normal_at_from_array(data, segment_index, t)
    }

    #[pyo3(signature = (x, y, i, j, clockwise=true, z=0.0))]
    fn arc_to_as_bezier(
        &mut self,
        x: f64,
        y: f64,
        i: f64,
        j: f64,
        clockwise: bool,
        z: f64,
    ) {
        let start_point = if let Some(last) =
            self.inner.pending_data().last()
        {
            (last[COL_X], last[COL_Y], last[COL_Z])
        } else if let Some(last) = self.inner.synced_data().last() {
            (last[COL_X], last[COL_Y], last[COL_Z])
        } else {
            self.inner.last_move_to
        };
        let end_point = (x, y, z);
        let center_offset = (i, j);
        let bezier_rows = convert_arc_to_beziers_from_array(
            start_point,
            end_point,
            center_offset,
            clockwise,
        );
        let pending = self.inner.pending_data_mut();
        for row in bezier_rows {
            pending.push(row);
        }
    }

    #[pyo3(signature = (fail_on_t_junction=false))]
    fn has_self_intersections(&mut self, fail_on_t_junction: bool) -> bool {
        let data = self.inner.synced_data();
        if data.is_empty() {
            return false;
        }
        check_self_intersection_from_array(data, fail_on_t_junction)
    }

    fn intersects_with(&mut self, other: &mut Geometry) -> bool {
        let data = self.inner.synced_data();
        let other_data = other.inner.synced_data();
        if data.is_empty() || other_data.is_empty() {
            return false;
        }
        check_intersection_from_array(data, other_data, false)
    }

    #[pyo3(signature = (amount))]
    fn grow(&self, amount: f64) -> Self {
        let result = grow_geometry(&self.inner, amount);
        Geometry { inner: result }
    }

    fn encloses(&mut self, other: &mut Geometry) -> PyResult<bool> {
        self.inner.sync_to_data();
        other.inner.sync_to_data();
        Ok(rayforge_geo::does_enclose(&self.inner, &other.inner))
    }

    fn remove_inner_edges(&mut self) -> PyResult<Geometry> {
        self.inner.sync_to_data();
        Ok(Geometry {
            inner: remove_inner_edges(&self.inner),
        })
    }

    fn split_inner_and_outer_contours(
        &mut self,
    ) -> PyResult<(Vec<Geometry>, Vec<Geometry>)> {
        self.inner.sync_to_data();
        let contours = split_into_contours(&self.inner);
        let (inner_indices, outer_indices) =
            split_inner_and_outer_contours(&contours);
        let inner: Vec<Geometry> = inner_indices
            .into_iter()
            .map(|i| Geometry {
                inner: contours[i].copy(),
            })
            .collect();
        let outer: Vec<Geometry> = outer_indices
            .into_iter()
            .map(|i| Geometry {
                inner: contours[i].copy(),
            })
            .collect();
        Ok((inner, outer))
    }

    #[pyo3(signature = (origin, p_width, p_height, anchor_y=None, stable_src_height=None, anchor_x=None, stable_src_width=None))]
    fn map_to_frame(
        &self,
        origin: (f64, f64),
        p_width: (f64, f64),
        p_height: (f64, f64),
        anchor_y: Option<f64>,
        stable_src_height: Option<f64>,
        anchor_x: Option<f64>,
        stable_src_width: Option<f64>,
    ) -> Geometry {
        let result = map_geometry_to_frame(
            &self.inner,
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

    fn split_into_contours(&mut self) -> Vec<Geometry> {
        self.inner.sync_to_data();
        split_into_contours(&self.inner)
            .into_iter()
            .map(|g| Geometry { inner: g })
            .collect()
    }

    fn split_into_components(&mut self) -> Vec<Geometry> {
        self.inner.sync_to_data();
        split_into_components(&self.inner)
            .into_iter()
            .map(|g| Geometry { inner: g })
            .collect()
    }

    #[pyo3(signature = (tolerance=0.01))]
    fn to_polygons(&self, tolerance: f64) -> Vec<Vec<Point>> {
        let mut linearized = self.inner.copy();
        linearized.sync_to_data();
        if !linearized.synced_data().is_empty() {
            let lin = linearize_data(linearized.synced_data(), tolerance);
            *linearized.synced_data_mut() = lin;
        }
        let segs = linearized.segments();
        let mut result = Vec::new();
        for seg in &segs {
            if seg.len() < 3 {
                continue;
            }
            let poly: Vec<Point> = seg.iter().map(|p| (p.0, p.1)).collect();
            if let Some(cleaned) =
                rayforge_geo::clean_polygon(&poly, 0.01 * tolerance)
            {
                result.push(cleaned);
            } else if poly.len() >= 3 {
                result.push(poly);
            }
        }
        result
    }

    fn __repr__(&mut self) -> String {
        self.inner.sync_to_data();
        let len = self.inner.len();
        let closed = self.inner.is_closed(1e-6);
        format!("<Geometry commands={} closed={}>", len, closed)
    }

}
