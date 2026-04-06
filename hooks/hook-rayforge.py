# -----------------------------------------------------------------------------
# Hook for rayforge package - ensures all submodules are collected
# This is needed because builtin addons import from rayforge but are
# loaded dynamically, so PyInstaller's static analysis misses these imports.
# -----------------------------------------------------------------------------

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all submodules from rayforge and its subpackages
hiddenimports = collect_submodules("rayforge")

# Collect data files from rayforge (locale, resources, etc.)
datas = collect_data_files("rayforge", include_py_files=False)
