#!/bin/bash
set -e

# This logic handles both CI/CD environments (where MSYS2_PATH may be pre-set)
# and local development (where it needs to be auto-detected).

# Priority 1: Check if MSYS2_PATH is already set AND is a valid MINGW64 environment.
# This is the most reliable method and correctly handles properly configured CI runners.
if [ -n "$MSYS2_PATH" ] && [ -d "$MSYS2_PATH/mingw64/bin" ]; then
    echo "--- Using pre-set and validated MSYS2 Path: $MSYS2_PATH"
else
    # Priority 2: If the variable is not set or points to an invalid location,
    # attempt auto-detection. This provides the "just works" local experience.
    if [ -n "$MSYS2_PATH" ]; then
        # This message is helpful for debugging CI issues.
        echo "--- Pre-set MSYS2_PATH '$MSYS2_PATH' is invalid. Attempting auto-detection..."
    else
        echo "--- MSYS2_PATH not set. Attempting auto-detection..."
    fi

    # Use the original auto-detection method. Redirect stderr to suppress noise.
    WIN_MSYS2_PATH=$(cd / && pwd -W 2>/dev/null)

    if [ -n "$WIN_MSYS2_PATH" ]; then
        # Convert the Windows path (e.g., C:\msys64) to a POSIX path (/c/msys64)
        DETECTED_PATH="/$(echo "$WIN_MSYS2_PATH" | sed 's/\\/\//g' | sed 's/://')"
        
        # CRITICAL: Validate the auto-detected path to ensure it's correct.
        if [ -d "$DETECTED_PATH/mingw64/bin" ]; then
            echo "--- Auto-detected and validated MSYS2 Path: $DETECTED_PATH"
            MSYS2_PATH="$DETECTED_PATH" # Set the variable for the rest of the script
        else
            echo "FATAL: Auto-detected path '$DETECTED_PATH' is not a valid MSYS2 MINGW64 installation."
            exit 1
        fi
    else
        echo "FATAL: Could not auto-detect MSYS2 installation path."
        echo "Please run this script from within an MSYS2 MINGW64 shell."
        exit 1
    fi
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
      mingw-w64-x86_64-toolchain
      mingw-w64-x86_64-cmake
      mingw-w64-x86_64-rust
      mingw-w64-x86_64-meson
      mingw-w64-x86_64-pkgconf
      mingw-w64-x86_64-gettext
      mingw-w64-x86_64-ntldd
      
      # These tools are in the base MSYS repository, not MINGW64
      autoconf
      automake
      libtool
      git

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

        export CARGO_BUILD_TARGET=x86_64-pc-windows-gnu

    echo "Installing/updating pip packages..."
    $PYTHON_BIN_PATH -m pip install --upgrade pip --break-system-packages

    $PYTHON_BIN_PATH -m pip install --no-cache-dir --no-build-isolation asyncudp==0.11.0 --break-system-packages
    $PYTHON_BIN_PATH -m pip install --no-cache-dir --no-build-isolation vtracer==0.6.11 --break-system-packages
    $PYTHON_BIN_PATH -m pip install --no-cache-dir pyclipper==1.3.0.post6 --break-system-packages
    $PYTHON_BIN_PATH -m pip install --no-cache-dir --no-build-isolation --no-deps pyvips==3.0.0 --break-system-packages
    $PYTHON_BIN_PATH -m pip install --no-cache-dir pyserial_asyncio==0.6 ezdxf==1.3.5 pypdf==5.3.1 --break-system-packages

    echo "✅ Windows MSYS2 dependency setup complete."
fi
