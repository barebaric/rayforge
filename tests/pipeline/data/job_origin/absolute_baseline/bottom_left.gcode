G21 ;Set units to mm
G90 ;Absolute positioning
G54
T0
G0 X-99.9 Y-39.9
M4 S500
G1 Y-49.9 F3000
G1 X-89.9 F3000
G1 Y-39.9 F3000
G1 X-99.9 F3000
M5
M5 ;Ensure laser is off
G0 X0 Y0 ;Return to origin
