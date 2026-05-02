---
description: "Manage machines in Rayforge - add, configure, export, import, and switch between different laser cutters and engravers for your projects."
---

# Machines

![Machines Settings](/screenshots/application-machines.png)

The Machines page in Application Settings shows a list of all configured
machines. Each entry shows the machine name and has buttons to edit or
delete it. The currently active machine is marked with a check icon.

## Adding a Machine

1. Click the **Add Machine** button at the bottom of the list
2. Select a device profile from the list to use as a template — each
   profile pre-configures the machine settings and G-code dialect

![Add Machine Dialog](/screenshots/add-machine-dialog.png)

3. The [machine settings dialog](../machine/general) opens where you can
   adjust the configuration

Alternatively, click **Import from File...** in the profile selector to
add a machine from a previously exported profile.

## Editing a Machine

Click the edit icon next to a machine to open the
[machine settings dialog](../machine/general).

## Switching the Active Machine

Use the machine dropdown in the main window header to switch between
configured machines. The selection is remembered between sessions.

## Deleting a Machine

1. Click the delete icon next to the machine
2. Confirm the deletion

:::warning
Deleting a machine cannot be undone. Export the profile first
if you want to preserve the configuration.
:::
