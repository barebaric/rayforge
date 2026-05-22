from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
)

hiddenimports = collect_submodules("raygeo")
hiddenimports += [
    "raygeo.geo",
    "raygeo.geo.algo",
    "raygeo.geo.types",
    "raygeo.ops",
    "raygeo.ops.axis",
    "raygeo.ops.state",
    "raygeo.ops.types",
]

datas = collect_data_files("raygeo", include_py_files=True)
