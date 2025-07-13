"""
This file is used by PyInstaller on windows and executed before anything else
to set up the environment.
"""
import os
import sys
# import gettext

print("Loading hook-environment.py")

# Only set GI_TYPELIB_PATH when running as a PyInstaller bundle
if getattr(sys, 'frozen', False):
    # sys._MEIPASS is the temporary directory where PyInstaller extracts files
    base_dir = sys._MEIPASS  # type: ignore

    # Set GI_TYPELIB_PATH to the correct directory (the build installs it in
    # gi/repository), such that Python code like
    # `gi.require_version('Gtk', '4.0')` works and finds the library.
    gi_typelib_path = os.path.join(base_dir, 'gi', 'repository')
    os.environ['GI_TYPELIB_PATH'] = gi_typelib_path

    # Set locale directory for gettext
    # locale_dir = os.path.join(base_dir, 'rayforge', 'locale')
    # gettext.bindtextdomain('rayforge', locale_dir)
    # gettext.textdomain('rayforge')
