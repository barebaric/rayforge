# Setup

This guide covers setting up your development environment for Rayforge.

## Linux

### Prerequisites

See the [Installation Guide](../../getting-started/installation#linux-pixi) for Pixi installation instructions.

### Pre-commit Hooks (Optional)

To automatically format and lint your code before each commit, you can install pre-commit hooks:

```bash
pixi run pre-commit-install
```

### Useful Commands

All commands are run via `pixi run`:

-   `pixi run rayforge`: Run the application.
    -   Add `--loglevel=DEBUG` for more verbose output.
-   `pixi run test`: Run the full test suite with `pytest`.
-   `pixi run format`: Format all code using `ruff`.
-   `pixi run lint`: Run all linters (`flake8`, `pyflakes`, `pyright`).

## Windows

### Prerequisites

See the [Installation Guide](../../getting-started/installation#windows-developer) for detailed MSYS2 developer setup instructions.

### Quick Start

Development tasks on Windows are managed via the `run.bat` script, which is a wrapper for the MSYS2 shell.

After cloning the repository and completing the MSYS2 setup, you can use these commands from a standard Windows Command Prompt or PowerShell:

```batch
.\run.bat setup
```

This executes `scripts/win/win_setup.sh` to install all necessary system and Python packages into your MSYS2/MinGW64 environment.

### Pre-commit Hooks (Optional)

To automatically format and lint your code before each commit, run this from the MSYS2 MINGW64 shell:

```bash
bash scripts/win/win_setup_dev.sh
```

:::note

Pre-commit hooks require running git commands from within the MSYS2 MINGW64 shell, not from PowerShell or Command Prompt.

:::

### Useful Commands

All commands are run via the `run.bat` script:

-   `run app`: Run the application from source.
    -   Add `--loglevel=DEBUG` for more verbose output.
-   `run test`: Run the full test suite using `pytest`.
-   `run lint`: Run all linters (`flake8`, `pyflakes`, `pyright`).
-   `run format`: Format and auto-fix code using `ruff`.
-   `run build`: Build the final Windows executable (`.exe`).

Alternatively, you can run the scripts directly from the MSYS2 MINGW64 shell:

-   `bash scripts/win/win_run.sh`: Run the application.
-   `bash scripts/win/win_test.sh`: Run the test suite.
-   `bash scripts/win/win_lint.sh`: Run all linters.
-   `bash scripts/win/win_format.sh`: Format and auto-fix code.
-   `bash scripts/win/win_build.sh`: Build the Windows executable.
