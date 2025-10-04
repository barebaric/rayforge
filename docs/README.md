
## Contributing

We welcome contributions of all kinds! Whether you're fixing a bug, adding
a feature, or improving documentation, your help is appreciated.

- **Report a Bug:** Open an [issue](https://github.com/barebaric/rayforge/issues)
  with a clear description and steps to reproduce.
- **Suggest a Feature:** Start a discussion or open a feature request issue.
- **Submit a Pull Request:** Please discuss major changes in an issue first.
- **Add a Driver:** See the [driver development guide](docs/driver.md) to add
  support for your hardware.

### Packaging Help Wanted

We are actively seeking help to create and maintain a **[Flatpak](flatpak/)** package.
If you have experience with Flatpak, your contribution would be highly valuable!

## Development

This project uses [**Pixi**](https://pixi.sh/) to manage dependencies and
development environments for a reproducible setup.

### Prerequisites

First, install Pixi by following the instructions on the
[official website](https://pixi.sh/latest/installation/).

### 1. Setup Environment

Clone the repository and install dependencies.

```bash
# Install Gtk/Adwaita system dependency (Debian/Ubuntu)
sudo apt install gir1.2-adw-1

# Let Pixi handle the rest
pixi install
```

This command reads `pixi.toml` and installs all conda and pip dependencies into a local
`.pixi` virtual environment.

### 2. Run the Application

```bash
pixi run rayforge
```

### 3. Run Tests and Lint

```bash
pixi run test
pixi run lint
```

### 4. Activate the Development Shell

For an interactive workflow, you can activate a shell within the project's environment.

```bash
pixi shell
```

Inside this shell, you can run commands directly without the `pixi run` prefix
(e.g., `rayforge`, `pytest`). Type `exit` to leave the shell.

### 5. Manage Dependencies

Always use Pixi to manage dependencies to ensure the `pixi.lock` file is updated.

```bash
# Add a conda package
pixi add numpy

# Add a PyPI package
pixi add --pypi requests
```

### 6. Translation Workflow

The following tasks are available for managing language translations:

#### Update `.po` files from source code

```bash
pixi run update-translations
```

#### Compile `.po` files to binary `.mo` files

```bash
pixi run compile-translations
```



### Add Device Drivers

Expand hardware support by adding drivers for new devices. See the [Adding Device Drivers](drivers.md) guide for details.

### Help with Packaging

We need help maintaining packages for different platforms:

- **Flatpak**: We're actively seeking help with Flatpak packaging
- **Other distributions**: Arch, Fedora, etc.
- **macOS**: Help bring Rayforge to Mac users

## Contribution Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use descriptive variable and function names
- Add docstrings to functions and classes
- Keep functions focused and single-purpose

### Testing

- Add tests for new features
- Ensure existing tests pass: `pixi run test`
- Test manually on your system
- Test with real hardware if possible

### Commit Messages

Write clear, descriptive commit messages:

```
Add overscan support for raster engraving

- Implement overscan calculation for raster operations
- Add UI controls for overscan settings
- Update G-code generator to handle overscan moves
- Add tests for overscan calculation

Fixes #123
```

### Pull Request Process

1. **Update documentation** if your changes affect user-facing features
2. **Run linter**: `pixi run lint`
3. **Run tests**: `pixi run test`
4. **Update CHANGELOG** (if applicable)
5. **Request review** from maintainers

## Development Resources

- **[Development Setup](development.md)**: Set up your development environment
- **[Adding Device Drivers](drivers.md)**: Create drivers for new hardware
- **Architecture Overview**: (Coming soon)
- **API Reference**: (Coming soon)

## Community

- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Track bugs and feature requests
- **Pull Requests**: Review and contribute code

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Welcome newcomers
- Focus on what is best for the community
- Show empathy towards others

## Recognition

Contributors are recognized in:

- GitHub contributor list
- Release notes for significant contributions
- About dialog in the application

Thank you for contributing to Rayforge!

---

**Next**: [Development Setup →](development.md) | [Adding Device Drivers →](drivers.md)
