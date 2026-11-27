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
    "raygeo.driver",
    "raygeo.driver.core",
    "raygeo.driver.grbl",
    "raygeo.driver.marlin",
    "raygeo.driver.smoothie",
    "raygeo.driver.octoprint",
    "raygeo.driver.epilog",
    "raygeo.driver.ruida",
    "raygeo.driver.esp3d",
    "raygeo.driver.xtool_d1",
    "raygeo.driver.xtool_m2",
    "raygeo.driver.simulator",
    "raygeo.driver.remote",
]

datas = collect_data_files("raygeo", include_py_files=True)
