import os
import sys


def configure_gtk_environment():
    if sys.platform == "win32":
        os.environ.setdefault("GTK_IM_MODULE", "simple")
