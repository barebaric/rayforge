# Addon Manifest

Every addon needs a `rayforge-addon.yaml` file in its root directory. This manifest tells Rayforge about your addon—its name, what it provides, and how to load it.

## Basic Structure

Here's a complete manifest with all the common fields:

```yaml
name: my_custom_addon
display_name: "My Custom Addon"
description: "Adds support for the XYZ laser cutter."
api_version: 9
url: https://github.com/username/my-custom-addon

author:
  name: Jane Doe
  email: jane@example.com

depends:
  - rayforge>=0.27.0

requires:
  - some-other-addon>=1.0.0

provides:
  backend: my_addon.backend
  frontend: my_addon.frontend
  assets:
    - path: assets/profiles.json
      type: profiles

license:
  name: MIT
```

## Required Fields

### `name`

A unique identifier for your addon. This must be a valid Python module name—only letters, numbers, and underscores, and it cannot start with a number.

```yaml
name: my_custom_addon
```

### `display_name`

A human-readable name shown in the UI. This can contain spaces and special characters.

```yaml
display_name: "My Custom Addon"
```

### `description`

A brief description of what your addon does. This appears in the addon manager.

```yaml
description: "Adds support for the XYZ laser cutter."
```

### `api_version`

The API version your addon targets. This must be at least 1 (the minimum supported version) and at most the current version (9). Using a higher version than supported will cause your addon to fail validation.

```yaml
api_version: 9
```

See the [Hooks](./addon-hooks.md#api-version-history) documentation for what changed in each version.

### `author`

Information about the addon author. The `name` field is required; `email` is optional but recommended for users to contact you.

```yaml
author:
  name: Jane Doe
  email: jane@example.com
```

## Optional Fields

### `url`

A URL to your addon's homepage or repository.

```yaml
url: https://github.com/username/my-custom-addon
```

### `depends`

Version constraints for Rayforge itself. Specify the minimum version your addon requires.

```yaml
depends:
  - rayforge>=0.27.0
```

### `requires`

Dependencies on other addons. List addon names with version constraints.

```yaml
requires:
  - some-other-addon>=1.0.0
```

### `version`

Your addon's version number. This is typically determined automatically from git tags, but you can specify it explicitly. Use semantic versioning (e.g., `1.0.0`).

```yaml
version: 1.0.0
```

## Entry Points

The `provides` section defines what your addon contributes to Rayforge.

### Backend

The backend module loads in both the main process and worker processes. Use this for machine drivers, step types, ops producers, and any core functionality.

```yaml
provides:
  backend: my_addon.backend
```

The value is a dotted Python module path relative to your addon directory.

### Frontend

The frontend module only loads in the main process. Use this for UI components, GTK widgets, and anything that needs the main window.

```yaml
provides:
  frontend: my_addon.frontend
```

### Assets

You can bundle asset files that Rayforge will recognize. Each asset has a path and type:

```yaml
provides:
  assets:
    - path: assets/profiles.json
      type: profiles
    - path: assets/templates
      type: templates
```

The `path` is relative to your addon root and must exist. Asset types are defined by Rayforge and may include things like machine profiles, material libraries, or templates.

## License Information

The `license` field describes how your addon is licensed. For free addons, just specify the license name using an SPDX identifier:

```yaml
license:
  name: MIT
```

Common SPDX identifiers include `MIT`, `Apache-2.0`, `GPL-3.0`, and `BSD-3-Clause`.

## Paid Addons

Rayforge supports paid addons through Gumroad license validation. If you want to sell your addon, you can configure it to require a valid license before it functions.

### Basic Paid Configuration

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
```

When `required` is true, Rayforge will check for a valid license before loading your addon. The `purchase_url` is shown to users who don't have a license.

### Gumroad Product ID

Add your Gumroad product ID to enable license validation:

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_id: "abc123def456"
```

For multiple product IDs (e.g., different pricing tiers):

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_ids:
    - "abc123def456"
    - "xyz789ghi012"
```

### Complete Paid Addon Example

Here's a full manifest for a paid addon:

```yaml
name: premium_laser_pack
display_name: "Premium Laser Pack"
description: "Advanced features for professional laser cutting."
api_version: 9
url: https://example.com/premium-laser-pack

author:
  name: Your Name
  email: you@example.com

depends:
  - rayforge>=0.27.0

provides:
  backend: premium_pack.backend
  frontend: premium_pack.frontend

license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/premium-laser-pack
  product_ids:
    - "standard_tier_id"
    - "pro_tier_id"
```

### Checking License Status in Code

In your addon code, you can check if a license is valid:

```python
@hookimpl
def rayforge_init(context):
    if context.license_validator:
        # Check if user has a valid license for your product
        is_valid = context.license_validator.is_product_valid("your_product_id")
        if not is_valid:
            # Optionally show a message or limit functionality
            logger.warning("License not found - some features disabled")
```

## Validation Rules

Rayforge validates your manifest when loading the addon. Here are the rules:

The `name` must be a valid Python identifier (letters, numbers, underscores, no leading numbers). The `api_version` must be an integer between 1 and the current version. The `author.name` cannot be empty or contain placeholder text like "your-github-username". Entry points must be valid module paths and the modules must exist. Asset paths must be relative (no `..` or leading `/`) and the files must exist.

If validation fails, Rayforge logs an error and skips your addon. Check the console output during development to catch these issues.
