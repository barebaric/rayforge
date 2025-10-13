# Setup

This guide covers setting up your development environment for Rayforge.

## Prerequisites

First, install [Pixi](https://pixi.sh/) by following the instructions on the [official website](https://pixi.sh/latest/installation/).

## Install System Dependencies

Install Gtk/Adwaita system dependency (Debian/Ubuntu):

```bash
sudo apt install gir1.2-adw-1
```

## Install Project Dependencies

Let Pixi handle the Python dependencies:

```bash
pixi install
```

This command reads `pixi.toml` and installs all conda and pip dependencies into a local `.pixi` virtual environment.

## Activate the Development Shell

For an interactive workflow, you can activate a shell within the project's environment:

```bash
pixi shell
```

Inside this shell, you can run commands directly without the `pixi run` prefix (e.g., `rayforge`, `pytest`). Type `exit` to leave the shell.

## Run the Application

Test that your setup works by running the application:

```bash
pixi run rayforge
```

You may also want to use debugging in development:

```bash
pixi run rayforge --loglevel=DEBUG
```

## Run Tests and Lint

Verify your setup by running the test suite:

```bash
pixi run test
pixi run lint
```

## Project Structure

The main source code is organized in the following directories:

- **`rayforge/core/`**: Document model and geometry handling
- **`rayforge/pipeline/`**: Core processing pipeline for generating machine operations
- **`rayforge/machine/`**: Hardware interface layer, including device drivers
- **`rayforge/doceditor/`**: Main document editor controller and its UI
- **`rayforge/workbench/`**: 2D/3D canvas and visualization systems
- **`rayforge/image/`**: Importers for various file formats (SVG, DXF, etc.)
- **`rayforge/shared/`**: Common utilities, including the tasker for background job management

## Next Steps

After setting up your environment, continue with [Submitting Changes](submitting-changes.md) to learn how to contribute code.