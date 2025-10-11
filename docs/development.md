# Development Setup

This guide will help you set up a development environment for Rayforge.

## Prerequisites

First, install **[Pixi](https://pixi.sh/)** by following the instructions on the [official website](https://pixi.sh/latest/installation/).

## Setup Environment

Clone the repository and install dependencies:

```bash
# Install Gtk/Adwaita system dependency (Debian/Ubuntu)
sudo apt install gir1.2-adw-1

# Let Pixi handle the rest
pixi install
```

This command reads `pixi.toml` and installs all conda and pip dependencies into a local `.pixi` virtual environment.

## Run the Application

```bash
pixi run rayforge
```

## Run Tests and Lint

```bash
pixi run test
pixi run lint
```

## Activate the Development Shell

For an interactive workflow, you can activate a shell within the project's environment:

```bash
pixi shell
```

Inside this shell, you can run commands directly without the `pixi run` prefix (e.g., `rayforge`, `pytest`). Type `exit` to leave the shell.

## Manage Dependencies

Always use Pixi to manage dependencies to ensure the `pixi.lock` file is updated:

```bash
# Add a conda package
pixi add numpy

# Add a PyPI package
pixi add --pypi requests
```

## Translation Workflow

The following tasks are available for managing language translations:

### Update `.po` files from source code

```bash
pixi run update-translations
```

### Compile `.po` files to binary `.mo` files

```bash
pixi run compile-translations
```

## Code Style

- Follow PEP 8 for Python code
- Use descriptive variable and function names
- Add docstrings to functions and classes
- Keep functions focused and single-purpose

## Testing

- Add tests for new features
- Ensure existing tests pass: `pixi run test`
- Test manually on your system
- Test with real hardware if possible

## Commit Messages

Write clear, descriptive commit messages:

```
Add overscan support for raster engraving

- Implement overscan calculation for raster operations
- Add UI controls for overscan settings
- Update G-code generator to handle overscan moves
- Add tests for overscan calculation

Fixes #123
```

## Pull Request Process

1. **Update documentation** if your changes affect user-facing features
2. **Run linter**: `pixi run lint`
3. **Run tests**: `pixi run test`
4. **Update CHANGELOG** (if applicable)
5. **Request review** from maintainers

## Next Steps

- Return to the [Contributing Guide](index.md)
- Learn about [Adding Device Drivers](drivers.md)
