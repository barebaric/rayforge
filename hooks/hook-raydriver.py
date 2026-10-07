from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
)

hiddenimports = collect_submodules("raydriver")
hiddenimports += ["raydriver.grbl", "raydriver.grbl.types"]

datas = collect_data_files("raydriver", include_py_files=True)
