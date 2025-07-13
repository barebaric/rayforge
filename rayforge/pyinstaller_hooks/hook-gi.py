"""
This file is used by PyInstaller on Windows to find the gi spec
files to work with Gtk4.

It sets GI_TYPELIB_PATH to the correct directory. Our build installs it
in gi/repository, such that Python code like
`gi.require_version('Gtk', '4.0')`
works and finds the library.
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Ensure gi module and all its submodules are included
hiddenimports = collect_submodules('gi')

# Collect non-Python files from the gi package
datas = collect_data_files('gi')
