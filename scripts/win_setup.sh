#!/bin/bash
set -e

# --- MSYS2 Path Detection ---
# 1. Use existing MSYS2_PATH (set by CI workflow) if available and non-empty.
if [ -n "$MSYS2_PATH" ]; then
    echo "--- Using pre-set MSYS2_PATH from environment: $MSYS2_PATH"
elif command -v cygpath >/dev/null 2>&1; then
    # 2. If running locally, use cygpath to detect the root installation path reliably.
    #    cygpath -u '/' gives the Unix path of the root (e.g., /c/msys64)
    MSYS2_PATH=$(cygpath -u '/')
    # MSYS2_PATH often ends up as '/' if we are already in the MSYS2 root.
    # If it is '/', we assume the installation is at the common default.
    if [ "$MSYS2_PATH" = "/" ] || [ -z "$MSYS2_PATH" ]; then
        MSYS2_PATH="/c/msys64"
    fi
    echo "--- Auto-detected MSYS2_PATH: $MSYS2_PATH"
else
    # 3. Final fallback (e.g., if cygpath isn't in PATH yet, or running outside msys2_shell)
    MSYS2_PATH="/c/msys64"
    echo "--- Using default MSYS2_PATH fallback: $MSYS2_PATH (Verify locally)"
fi

# 4. Final validation check
if [ ! -d "$MSYS2_PATH/mingw64/bin" ]; then
    echo "FATAL: Detected MSYS2_PATH ($MSYS2_PATH) does not contain mingw64/bin."
    echo "       Please ensure MSYS2 is installed and correctly configured."
    exit 1
fi

# --- Stage 1: Install System (Pacman) Dependencies ---
if [[ "$1" == "pacman" || -z "$1" ]]; then
    echo "--- STAGE 1: Installing System Dependencies (Pacman) ---"
    
    # --- Create Environment File (.msys2_env) ---
    echo "Writing environment variables to .msys2_env..."
    # Note: We use 'export' in the env file so it's ready to be sourced by subsequent scripts (like win_build.sh)
    echo "export MSYS2_PATH=$MSYS2_PATH" > .msys2_env
    echo "export PKG_CONFIG_PATH=$MSYS2_PATH/mingw64/lib/pkgconfig" >> .msys2_env
    echo "export GI_TYPELIB_PATH=$MSYS2_PATH/mingw64/lib/girepository-1.0" >> .msys2_env
    echo "export LD_LIBRARY_PATH=$MSYS2_PATH/mingw64/lib" >> .msys2_env
    
    # --- Permissions Check ---
    # We skip this check in CI because the runner user usually has sufficient privileges.
    if [[ -z "${CI}" ]]; then
        if ! touch "/var/lib/pacman/sync/permission_test" 2>/dev/null; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "!!! PERMISSION ERROR: This script requires Administrator privileges."
            echo "!!! Please run setup.bat using 'Run as administrator' (if applicable)."
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            exit 1
        fi
        rm -f "/var/lib/pacman/sync/permission_test"
    fi

    PACKAGES=(
      # --- Core Build Toolchain ---
      mingw-w64-x86_64-gcc
      mingw-w64-x86_64-cmake
      mingw-w64-x86_64-make
      mingw-w64-x86_64-rust
      mingw-w64-x86_64-meson
      mingw-w64-x86_64-pkgconf
      mingw-w64-x86_64-gettext
      mingw-w64-x86_64-ntldd
      
      # These tools are in the base MSYS repository, not MINGW64
      autoconf
      automake
      libtool

      # Installer & Icon Tools
      mingw-w64-x86_64-nsis
      mingw-w64-x86_64-imagemagick

      # Base Python Environment and Bindings
      mingw-w64-x86_64-python
      mingw-w64-x86_64-python-pip
      mingw-w64-x86_64-python-cffi
      mingw-w64-x86_64-python-gobject
      mingw-w64-x86_64-python-cairo
      
      # GTK4 and related C-level dependencies
      mingw-w64-x86_64-adwaita-icon-theme
      mingw-w64-x86_64-gtk4
      mingw-w64-x86_64-glib2
      mingw-w64-x86_64-libadwaita
      mingw-w64-x86_64-gobject-introspection
      mingw-w64-x86_64-cairo
      mingw-w64-x86_64-librsvg
      mingw-w64-x86_64-poppler
      mingw-w64-x86_64-libvips
      mingw-w64-x86_64-openslide
      mingw-w64-x86_64-angleproject

      # Python C-extension packages and build tools
      mingw-w64-x86_64-cython
      mingw-w64-x86_64-python-maturin
      mingw-w64-x86_64-python-numpy
      mingw-w64-x86_64-python-opencv
      mingw-w64-x86_64-python-pyopengl
      mingw-w64-x86_64-python-pyopengl-accelerate
      mingw-w64-x86_64-python-scipy
      mingw-w64-x86_64-python-svgelements
      mingw-w64-x86_64-pyinstaller 

      # Pure Python dependencies needed for environment
      mingw-w64-x86_64-python-aiohttp
      mingw-w64-x86_64-python-blinker
      mingw-w64-x86_64-python-platformdirs
      mingw-w64-x86_64-python-poetry-core
      mingw-w64-x86_64-python-pytest-asyncio
      mingw-w64-x86_64-python-pytest-cov
      mingw-w64-x86_64-python-pytest-mock
      mingw-w64-x86_64-python-websockets
      mingw-w64-x86_64-python-yaml
    )

    echo "Updating MSYS2 database and system..."
    pacman -Syyu --noconfirm || true

    echo "Installing required system packages..."
    pacman -S --needed --noconfirm "${PACKAGES[@]}"

    echo "✅ Pacman setup complete."
fi

# --- Stage 2: Install Python (Pip) Dependencies ---
if [[ "$1" == "pip" || -z "$1" ]]; then
    echo "--- STAGE 2: Installing Python Dependencies (Pip) ---"
    
    if [ ! -f .msys2_env ]; then
        echo "FATAL: .msys2_env not found. Please run the 'pacman' stage first."
        exit 1
    fi
    source .msys2_env


    # Add Python site-packages path, now that we know Python is installed
    PYTHON_BIN_PATH="$MSYS2_PATH/mingw64/bin/python"
    PYTHON_VERSION=$("$PYTHON_BIN_PATH" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
    echo "export PYTHONPATH=$MSYS2_PATH/mingw64/lib/$PYTHON_VERSION/site-packages:\$PYTHONPATH" >> .msys2_env
    source .msys2_env

    # Export necessary toolchain variables
    export CC=x86_64-w64-mingw32-gcc
    export CXX=x86_w64-w64-mingw32-g++
    export LD=x86_64-w64-mingw32-ld
    export CARGO_BUILD_TARGET=x86_64-pc-windows-gnu

    echo "Installing/updating pip packages..."
    $PYTHON_BIN_PATH -m pip install --upgrade pip --break-system-packages

    $PYTHON_BIN_PATH -m pip install --no-cache-dir --no-build-isolation vtracer==0.6.11 --break-system-packages
    $PYTHON_BIN_PATH -m pip install --no-cache-dir pyclipper==1.3.0.post6 --break-system-packages
    $PYTHON_BIN_PATH -m pip install --no-cache-dir --no-build-isolation --no-deps pyvips==3.0.0 --break-system-packages
    $PYTHON_BIN_PATH -m pip install --no-cache-dir pyserial_asyncio==0.6 ezdxf==1.3.5 pypdf==5.3.1 --break-system-packages

    echo "✅ Windows MSYS2 dependency setup complete."
fi
