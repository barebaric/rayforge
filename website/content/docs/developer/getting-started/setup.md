# Setup

This guide covers setting up your development environment for Rayforge.

## Linux

### Prerequisites

First, install [Pixi](https://pixi.sh/) by following the instructions on the [official website](https://pixi.sh/latest/installation/).

### Install

Install the Gtk/Adwaita system dependency, and then let Pixi handle the rest:

```bash
sudo apt install gir1.2-adw-1
pixi install
```

This command reads `pixi.toml` and installs all conda and pip dependencies into a local `.pixi` virtual environment.

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

-   [MSYS2](https://www.msys2.org/) (provides the MinGW64 environment).
-   [Git for Windows](https://git-scm.com/download/win).

### Install

Development tasks on Windows are managed via the `run.bat` script, which is a wrapper for the MSYS2 shell.

After cloning the repository, run the setup command from a standard Windows Command Prompt or PowerShell:

```batch
.\run.bat setup
```

This executes `scripts/win_setup.sh` to install all necessary system and Python packages into your MSYS2/MinGW64 environment.

### Useful Commands

All commands are run via the `run.bat` script:

-   `run app`: Run the application from source.
    -   Add `--loglevel=DEBUG` for more verbose output.
-   `run test`: Run the full test suite using `pytest`.
-   `run build`: Build the final Windows executable (`.exe`).
