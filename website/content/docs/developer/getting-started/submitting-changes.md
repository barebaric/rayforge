# Submitting Changes

This guide covers the process for contributing code improvements to Rayforge.

## Create a Feature Branch

Create a descriptive branch for your changes:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

## Make Your Changes

- Follow the existing code style and conventions
- Write clean, focused commits with clear messages
- Add tests for new functionality
- Update documentation as needed

## Test Your Changes

Run the full test suite to ensure nothing is broken:

```bash
# Run all tests and linting
pixi run test
pixi run lint
```

## Sync with Upstream

Before creating a pull request, sync with the upstream repository:

```bash
# Fetch the latest changes
git fetch upstream

# Rebase your branch on the latest main
git rebase upstream/main
```

## Submit a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a pull request on GitHub with:
   - A clear title describing the change
   - A detailed description of what you changed and why
   - Reference to any related issues
   - Screenshots if the change affects the UI

## Code Review Process

- All pull requests require review before merging
- Address feedback promptly and make requested changes
- Keep the discussion focused and constructive

## Merge Requirements

Pull requests are merged when they:

- [ ] Pass all automated tests
- [ ] Follow the project's coding style
- [ ] Include appropriate tests for new functionality
- [ ] Have documentation updates if needed
- [ ] Are approved by at least one maintainer

## Additional Guidelines

### Commit Messages

Use clear, descriptive commit messages:

- Start with a capital letter
- Keep the first line under 50 characters
- Use the imperative mood ("Add feature" not "Added feature")
- Include more detail in the body if needed

### Small, Focused Changes

Keep pull requests focused on a single feature or fix. Large changes should be broken into smaller, logical pieces.

!!! tip "Discuss First"
    For major changes, open an [issue](https://github.com/barebaric/rayforge/issues) first to discuss your approach before investing significant time.

!!! note "Need Help?"
    If you're unsure about any part of the contribution process, don't hesitate to ask for help in an issue or discussion.