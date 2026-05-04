---
description: "Configure general machine settings in Rayforge — set the machine name, select a driver, and configure speeds and acceleration."
---

# General Settings

The General page in Machine Settings contains the machine name, driver
selection and connection settings, and speed parameters.

![General Settings](/screenshots/machine-general.png)

## Machine Name

Give your machine a descriptive name. This helps identify the machine in
the machine selector dropdown when you have multiple machines configured.

## Driver

Select the driver that matches your machine's controller. The driver
handles communication between Rayforge and the hardware.

After selecting a driver, connection-specific settings appear below the
selector (e.g. serial port, baud rate). These vary depending on the
chosen driver.

:::tip
An error banner at the top of the page warns you if the driver is not
configured or encounters a problem.
:::

## Speeds & Acceleration

These settings control the maximum speeds and acceleration. They are used
for job time estimation and path optimization.

### Max Travel Speed

The maximum speed for rapid (non-cutting) movements when the laser is
off and the head is moving to a new position.

- **Typical range**: 2000-5000 mm/min
- **Note**: Actual speed is also limited by your firmware settings.
  This field is disabled if the selected G-code dialect does not
  support specifying a travel speed.

### Max Cut Speed

The maximum speed allowed during cutting or engraving operations.

- **Typical range**: 500-2000 mm/min
- **Note**: Individual operations may use lower speeds

### Acceleration

The rate at which the machine accelerates and decelerates, used for
time estimations and calculating the default overscan distance.

- **Typical range**: 500-2000 mm/s²
- **Note**: Must match or be lower than firmware acceleration settings

:::tip
Start with conservative speed values and increase gradually. Observe
your machine for belt skipping, motor stalling, or loss of positioning
accuracy.
:::

## Exporting a Machine Profile

Click the share icon in the header bar of the settings dialog to export
the current machine configuration. Choose a folder to save to. A zip file
is created containing the machine settings and its G-code dialect, which
can be shared with other users or imported on another system.

## See Also

- [Configuration Wizard](config-wizard) - Automatically detect and
  configure a connected device
- [Hardware Settings](hardware) - Work area dimensions and axis
  configuration
- [Device Settings](device) - Read and write firmware settings on the
  controller
