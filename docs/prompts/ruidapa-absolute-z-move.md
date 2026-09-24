# Request: absolute Z positioning support in the ruida-pa backend

## Context

Rayforge issue #452 adds a "move the laser head to an arbitrary X/Y/Z
position" feature. The Rayforge driver API
`Driver.move_to(pos_x, pos_y, pos_z=None)` now carries an optional
absolute Z target (command coordinates, mm).

## Current behavior

`rayforge/machine/driver/ruidarpa/rpa_adapter.py` implements `move_to`
by delegating to `self._backend.jog_xy_to(pos_x, pos_y)`. There is no
backend entry point for an absolute Z move, so `pos_z` is currently
logged and ignored by the Ruida driver.

## Request

Add a backend capability to position Z absolutely, e.g. a
`jog_z_to(pos_z: float)` method (or an extension of `jog_xy_to`)
matching the existing machine-frame mm coordinate convention used by
`jog_xy_to` and the `POSITION_*` status reporting, honoring the same
speed handling as `jog_set_xy_speed` / the recorded jog speed.

Once available, `RpaAdapter.move_to` should pass `pos_z` through to
the backend instead of ignoring it, analogous to how `home(Axis.Z)`
uses `home_z`.
