from .version import (
    get_version_from_file,
    get_version_from_git,
    get_version_from_pkg,
)

__version__ = (
    get_version_from_file() or get_version_from_git() or get_version_from_pkg()
)
