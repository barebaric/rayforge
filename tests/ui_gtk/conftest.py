# MUST run before any module in this directory imports OpenGL: GTK4
# creates its GL contexts via EGL on Linux, and PyOpenGL must use the
# matching EGL platform or GetCurrentContext() returns NULL (the same
# setup app.py applies for the application itself).
import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
