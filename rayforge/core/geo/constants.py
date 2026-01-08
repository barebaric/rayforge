# The geometry data is stored in an (N, 8) float64 array.
# [type, x, y, z, param1, param2, param3, param4]

# Command Types
CMD_TYPE_MOVE = 1.0
CMD_TYPE_LINE = 2.0
CMD_TYPE_ARC = 3.0
CMD_TYPE_BEZIER = 4.0

# Column Indices
COL_TYPE = 0
COL_X = 1
COL_Y = 2
COL_Z = 3
# Arc parameters
COL_I = 4  # Center offset X for arcs
COL_J = 5  # Center offset Y for arcs
COL_CW = 6  # Clockwise flag for arcs (1.0 = True, 0.0 = False)
# Bezier parameters (absolute coordinates)
COL_C1X = 4  # Control point 1 X
COL_C1Y = 5  # Control point 1 Y
COL_C2X = 6  # Control point 2 X
COL_C2Y = 7  # Control point 2 Y

# Array Shape
GEO_ARRAY_COLS = 8
