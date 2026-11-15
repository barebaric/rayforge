---
description:
  "Install and manage add-ons in Rayforge. Extend functionality with community-built plugins for
  your laser cutting workflow."
---

# Addons

![Addons Settings](/screenshots/app-settings-addons.webp)

The Addons page shows installed extension addons and allows you to manage them.

## Overview

Addons are extensions that add functionality to Rayforge. They can provide:

- New operations
- Import/export filters
- Machine drivers
- UI enhancements
- Material libraries

## Online Services

The **Enable external addon services** setting controls access to the online registry, remote
installation, and Gumroad and Patreon license services. When disabled, built-in and already
installed local addons remain available, but Rayforge makes no external addon-service requests.

## Installed Addons

The addon list shows all installed addons with:

- **Name**: The addon's display name
- **Version**: Current installed version
- **Description**: Brief description of functionality
- **Status**: Whether enabled or disabled

## Addon Details

![Addon Detail View](/screenshots/app-settings-addons-detail.webp)

Click an addon row — in the installed addons list or in the Addon Registry dialog — to open its
detail view. The detail view shows:

- **Screenshots**: A swipeable gallery, when the addon's author provides screenshots
- **Description**: A longer description of what the addon does
- **Author, version, and license**
- A link to the addon's repository
- The **Install**, **Update**, or **Unlock** action, depending on the addon's state

Screenshots and the long description are loaded from the online registry. When external addon
services are disabled or the registry cannot be reached, the detail view still shows the addon's
local information.

## Managing Addons

### Installing Addons

1. Click the **Install** button
2. Select an addon file (`.zip` or addon repository)
3. Confirm the installation (community addons require confirming a trust warning)
4. Wait for the installation to complete

### Updating Addons

When updates are available, an indicator appears in the Addon Registry dialog. Click **Update** to
install available updates.

### Removing Addons

1. Select the addon to remove
2. Click the **Remove** button
3. Confirm the removal

<!-- prettier-ignore-start -->
:::warning
Removing an addon may affect documents that use its features.
:::
<!-- prettier-ignore-end -->

## Community Addons

Addons that were not created by the Rayforge team are marked as community addons with a dedicated
icon in the addon lists.

When you install a community addon, a confirmation dialog explains that the addon is not provided by
Rayforge itself and shows a link to the addon's repository. Open the link in your browser to review
the code and its author before continuing.

<!-- prettier-ignore-start -->
:::warning
Addons run with the same access to your computer and your files as Rayforge itself. Only install
community addons if you trust their authors.
:::
<!-- prettier-ignore-end -->

## Addon Sources

Addons can be installed from:

- **Repositories**: Git repositories containing addons
- **Official registry**: Curated addons from the Rayforge team

## Related Topics

- [Machines](machines) - Machine configuration
- [Materials](materials) - Material libraries
- [Recipes](recipes) - Operation recipes
