import sys
from pathlib import Path

_root_dir = Path(__file__).parent
_builtin_addons = _root_dir / "rayforge" / "builtin_addons"
_private_addons = _root_dir / "rayforge" / "private_addons"

for _addon_dir in [_builtin_addons, _private_addons]:
    if not _addon_dir.exists():
        continue
    for _addon_path in _addon_dir.iterdir():
        if _addon_path.is_dir():
            _resolved = _addon_path.resolve()
            if str(_resolved) not in sys.path:
                sys.path.insert(0, str(_resolved))
