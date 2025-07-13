"""
This file is used by PyInstaller on Windows to find the gi spec
files to work with Gtk4.

It sets GI_TYPELIB_PATH to the correct directory. Our build installs it
in gi/repository, such that Python code like
`gi.require_version('Gtk', '4.0')`
works and finds the library.
"""
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Ensure gi module and all its submodules are included
hiddenimports = collect_submodules('gi')


def find_typelib_files():
    """Recursively search for *.typelib files in MSYS2 paths and print them."""
    base_paths = [
        os.environ.get('MSYS2_BASE', 'D:/a/_temp/msys64'),
        '/mingw64/lib',
        '/mingw64/share',
        '/usr/lib',
        '/usr/share',
    ]
    found_files = []
    for base_path in base_paths:
        if os.path.exists(base_path):
            print(f"Searching for *.typelib in {base_path}")
            for root, _, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.typelib'):
                        full_path = os.path.join(root, file)
                        print(f"Found typelib: {full_path}")
                        found_files.append((full_path, 'gi/repository'))
        else:
            print(f"Path does not exist: {base_path}")
    return found_files


# Collect non-Python files from the gi package
datas = collect_data_files('gi', include_py_files=False)

# Collect typelib files by searching recursively
datas += find_typelib_files()
