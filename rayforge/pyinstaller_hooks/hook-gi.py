"""
This file is used by PyInstaller on Windows to find the gi spec
files to work with Gtk4.

It sets GI_TYPELIB_PATH to the correct directory. Our build installs it
in gi/repository, such that Python code like
`gi.require_version('Gtk', '4.0')`
works and finds the library.
"""
import os
import sys

print("Loading hook-gi.py")

# sys._MEIPASS is the temporary directory where PyInstaller extracts files
base_dir = sys._MEIPASS  # type: ignore
gi_typelib_path = os.path.join(base_dir, 'gi', 'repository')
os.environ['GI_TYPELIB_PATH'] = gi_typelib_path
