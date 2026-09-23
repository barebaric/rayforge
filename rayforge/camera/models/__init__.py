"""Camera model package.

Deliberately does not eagerly import :mod:`.camera` here: doing so
would force ``rayforge.camera.source`` (which ``.camera`` needs at
import time) to also finish loading before this package's own
``__init__`` returns, recreating the exact circular-import problem
that :class:`.source_type.CameraSourceType` was split out to avoid.
Import ``rayforge.camera.models.camera.Camera`` directly instead of
``rayforge.camera.models.Camera``.
"""
