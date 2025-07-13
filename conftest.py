import pytest  # noqa: F401
import gettext


def pytest_configure(config):
    """
    Configure gettext for localization during pytest collection.
    This hook is called early in the pytest process.
    """
    gettext.install('rayforge')
