import os
import sys

# Get the bundle directory (sys._MEIPASS in PyInstaller)
bundle_dir = getattr(sys, '_MEIPASS')

# Set GI_TYPELIB_PATH to the gi/repository directory in the bundle
typelib_path = os.path.join(bundle_dir, 'gi', 'repository')
os.environ['GI_TYPELIB_PATH'] = typelib_path

# Debug: Print the GI_TYPELIB_PATH to verify
print(f"Set GI_TYPELIB_PATH to: {os.environ['GI_TYPELIB_PATH']}")
