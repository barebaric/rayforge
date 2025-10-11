# Rayforge Documentation Setup

This document describes the mkdocs-material documentation setup for Rayforge with versioning support.

## Overview

The documentation system is built using:

- **[MkDocs Material](https://squidfunk.github.io/mkdocs-material/)**: Modern documentation theme
- **GitHub Actions**: Automated deployment

## Features

✅ **Rich content** - Support for screenshots, code blocks, admonitions, tabs, and more
✅ **Search** - Full-text search across all documentation
✅ **Responsive** - Works on desktop, tablet, and mobile
✅ **Navigation** - Multi-level navigation with breadcrumbs
✅ **Extensible** - Easy to add new pages and sections

## Directory Structure

```
rayforge/
├── mkdocs.yml              # MkDocs configuration
├── website/                # Documentation source files
│   ├── index.md           # Homepage
│   ├── getting-started/   # Installation and setup
│   ├── ui/                # User interface documentation
│   ├── features/          # Feature guides
│   ├── machine/           # Machine setup
│   ├── files/             # File format documentation
│   ├── troubleshooting/   # Problem solving
│   ├── reference/         # Technical reference
│   ├── contributing/      # Contribution guide
│   ├── assets/            # Logos, icons, etc.
│   ├── stylesheets/       # Custom CSS
│   └── includes/          # Reusable snippets
├── docs/                  # Screenshots (shared with README)
└── .github/workflows/
    └── docs.yml           # Documentation deployment workflow
```

## Quick Start
This guide shows the simplified commands for working with Rayforge documentation.

## ✅ Simplified Commands (New!)

The documentation tasks have been simplified so you don't need to specify the environment:

### Serve Documentation Locally

Start a local development server with live reload:

```bash
pixi run site-serve
```

- Opens http://127.0.0.1:8000/
- Auto-reloads when you edit files
- Perfect for writing and previewing documentation

### Build Documentation

Build the static site to the `build/website/` directory:

```bash
pixi run site-build
```

- Creates production-ready HTML in `build/website/`
- Validates all links and references
- Use this to test before deploying

### Deploy Documentation

The git-hub action will publish the new site when merged to main.


## Versioning Workflow

### Development Documentation

Updated automatically when pushing to `main` branch:

```bash
git checkout main
# Make documentation changes in website/
git add website/
git commit -m "docs: update feature guide"
git push origin main
```

The GitHub Action will deploy to the `dev` version.

### Release Documentation

Tagged releases automatically create versioned documentation:

```bash
# Create and push a version tag
git tag v1.0.0
git push origin v1.0.0
```

The GitHub Action will:
1. Deploy documentation for version `1.0.0`
2. Update the `latest` alias to point to `1.0.0`
3. Make `latest` the default version

### Manual Deployment

If needed, deploy manually using mike:

```bash
# Deploy development version
mike deploy dev development --update-aliases

# Deploy release version
mike deploy 1.0.0 latest --update-aliases

# Set default version
mike set-default latest

# List all versions
mike list

# Delete a version
mike delete old-version
```

## Adding New Pages

1. Create a new `.md` file in the appropriate directory
2. Add to navigation in `mkdocs.yml`:

```yaml
nav:
  - Section:
      - section/index.md
      - New Page: section/new-page.md
```

3. Write content following the structure guidelines
4. Add internal links to related pages
5. Test locally before committing

## Customization

### Theme Colors

Edit `mkdocs.yml` to change colors:

```yaml
theme:
  palette:
    primary: indigo  # Change to blue, red, green, etc.
    accent: indigo   # Accent color for buttons, links
```


## GitHub Pages Setup

The documentation is deployed to GitHub Pages via the `gh-pages` branch.

### Initial Setup

1. Enable GitHub Pages in repository settings
2. Set source to `gh-pages` branch
3. Wait for first deployment from workflow

### Custom Domain (Optional)

To use a custom domain:

1. Add `CNAME` file to `website/` with your domain
2. Configure DNS with your domain provider
3. Update `site_url` in `mkdocs.yml`

## Automated Deployment

The `.github/workflows/docs.yml` workflow handles automatic deployment:

- **Pull Requests**: Build docs to verify no errors (doesn't deploy)
- **Push to main**: Deploy to `dev` version
- **Version tags**: Deploy to version number and update `latest`

### Workflow Configuration

Key environment variables in the workflow:

- `version`: Extracted from git tag or defaults to `dev`
- `alias`: Set to `latest` for releases, `dev` for development

## Maintenance

### Updating Dependencies

```bash
# Update pixi dependencies
pixi update -e docs

# Or update pip packages
pip install --upgrade mkdocs-material mike
```

### Checking Build Status

Visit the [Actions tab](https://github.com/barebaric/rayforge/actions) on GitHub to monitor deployment status.

