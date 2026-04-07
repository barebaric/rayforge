# Settings

![General Settings](/screenshots/application-general.png)

Customize Rayforge to match your workflow and preferences.

## Accessing Settings

- **Menu**: Edit → Settings
- **Keyboard**: <kbd>ctrl+comma</kbd>

## General

The General page contains application-wide settings.

### Appearance

- **Theme**: Choose between System, Light, or Dark theme

### Operations

- **Auto-update operations**: When enabled (the default), operations are
  recalculated automatically after every change. Disable this if you prefer
  to trigger recalculation manually via the toolbar button — this can be
  helpful on slower machines or with very complex documents.

### Units

Configure display units for various values throughout the application:

- **Length**: Millimeters, inches, and other length units
- **Speed**: mm/min, mm/sec, inches/min, and other speed units
- **Acceleration**: mm/s² and other acceleration units

### Startup

Configure what happens when you start the application. Files specified on
the command line will always override these settings.

- **Startup behavior**:
  - Open nothing
  - Open last project
  - Open specific project
- **Project path**: Path to the specific project to open on startup
  (only visible when "Open specific project" is selected)

## Machines

The Machines page lets you manage your machine configurations.

- **Add Machine**: Create a new machine configuration
- **Remove Machine**: Delete a selected machine
- **Machine List**: Shows all configured machines with their connection status

For detailed machine configuration, see the [Machine Setup](../machine/general) section.

## Materials

The Materials page manages your material library.

- **Add Library**: Create a new material library
- **Edit**: Rename a selected library
- **Delete**: Remove a selected library

See [Materials](../application-settings/materials) for more details.

## Recipes

The Recipes page manages your operation recipes.

- **Add Recipe**: Create a new recipe
- **Edit**: Modify a selected recipe
- **Delete**: Remove a selected recipe

See [Recipes](../application-settings/recipes) for more details.

## Addons

The Addons page shows installed extension addons.

- **Addon List**: Shows all installed addons
- **Addon Details**: View information about each addon

## Privacy

The Privacy page controls anonymous usage reporting.

- **Report Anonymous Usage**: When enabled, anonymous usage data is sent to help improve Rayforge. No personal information is collected.

## Licenses

The Licenses page manages your license keys for premium features and addons.

- **Link Patreon Account**: Connect your Patreon account to access supporter benefits
- **Addon License Keys**: Enter license keys for premium addons

---

## Related Topics

- [Machine Setup](../machine/general) - Configure your laser cutter
- [Shortcuts](../reference/shortcuts) - Keyboard shortcuts
