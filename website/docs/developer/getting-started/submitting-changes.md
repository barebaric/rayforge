---
description:
  "How to submit changes to Rayforge. Follow the contribution workflow for pull requests, code
  review, and merging."
---

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

2. Create a pull request on GitHub and fill out the pull request template:
   - A clear title and a description of the change and the problem it solves
   - Why you chose your specific architectural approach and which alternatives you rejected
   - The exact steps you used to test the change locally
   - Reference to any related issues
   - Screenshots if the change affects the UI
   - Every checkbox in the template's checklist ticked

<!-- prettier-ignore-start -->
:::warning[Template Enforcement]
Pull requests that do not fill out the template are closed automatically by a
bot. Edit the pull request description to answer every section and to tick
every checkbox — the pull request is then reopened automatically.
:::
<!-- prettier-ignore-end -->

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

## AI-Assisted Code

We welcome the use of AI tools to help write code. However, you are strictly responsible for the
code you submit. You MUST fully understand it and must be able to explain the architectural choices.

AI-generated code often introduces subtle bugs, architectural mismatches, and security flaws. Pull
requests that appear to be raw, unreviewed AI output will be closed to respect maintainer time.

## Additional Guidelines

### Commit Messages

Use clear, descriptive commit messages:

- Start with a capital letter
- Keep the first line under 50 characters
- Use the imperative mood ("Add feature" not "Added feature")
- Include more detail in the body if needed

### Small, Focused Changes

Keep pull requests focused on a single feature or fix. Large changes should be broken into smaller,
logical pieces.

<!-- prettier-ignore-start -->
:::tip[Discuss First]
For major changes, open an
[issue](https://github.com/barebaric/rayforge/issues) first to discuss your approach before
investing significant time.
:::
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
:::note[Need Help?]
If you're unsure about any part of the contribution process, don't hesitate to
ask for help in an issue or discussion.
:::
<!-- prettier-ignore-end -->
