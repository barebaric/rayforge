/// Command type for a move operation (start a new path segment).
/// Each command is stored as an 8-element array with command type in the first column.
pub const CMD_TYPE_MOVE: f64 = 1.0;

/// Command type for a line operation (draw a straight line).
pub const CMD_TYPE_LINE: f64 = 2.0;

/// Command type for an arc operation (draw a circular arc).
/// Arc commands use columns I, J for center offset and CW for clockwise flag.
pub const CMD_TYPE_ARC: f64 = 3.0;

/// Command type for a cubic Bezier curve operation.
/// Bezier commands use columns C1X, C1Y, C2X, C2Y for control points.
pub const CMD_TYPE_BEZIER: f64 = 4.0;

/// Column index for command type in the data array.
pub const COL_TYPE: usize = 0;

/// Column index for X coordinate in the data array.
pub const COL_X: usize = 1;

/// Column index for Y coordinate in the data array.
pub const COL_Y: usize = 2;

/// Column index for Z coordinate in the data array.
pub const COL_Z: usize = 3;

/// Column index for I (center offset X) in arc commands.
pub const COL_I: usize = 4;

/// Column index for J (center offset Y) in arc commands.
pub const COL_J: usize = 5;

/// Column index for clockwise flag in arc commands (0.0 = CCW, 1.0 = CW).
pub const COL_CW: usize = 6;

/// Column index for first control point X in Bezier commands.
pub const COL_C1X: usize = 4;

/// Column index for first control point Y in Bezier commands.
pub const COL_C1Y: usize = 5;

/// Column index for second control point X in Bezier commands.
pub const COL_C2X: usize = 6;

/// Column index for second control point Y in Bezier commands.
pub const COL_C2Y: usize = 7;

/// Number of columns in the geometry command array.
pub const GEO_ARRAY_COLS: usize = 8;
