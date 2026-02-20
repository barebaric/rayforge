# Machines

![Machines Settings](/screenshots/application-machines.png)

The Machines page in Application Settings allows you to manage machine
profiles. Each profile contains all the configuration for a specific
laser machine.

## Machine Profiles

Machine profiles store complete configuration for a laser cutter or
engraver, including:

- **General settings**: Name, speeds, acceleration
- **Hardware settings**: Work area dimensions, axis configuration
- **Laser settings**: Power range, PWM frequency
- **Device settings**: Serial port, baud rate, firmware type
- **G-code settings**: Custom G-code dialect options
- **Camera settings**: Camera calibration and alignment

## Managing Machines

### Adding a New Machine

1. Click the **Add New Machine** button
2. Enter a descriptive name for your machine
3. Configure the machine settings (see
   [Machine Setup](../machine/general) for details)
4. Click **Save** to create the profile

### Switching Between Machines

Use the machine selector dropdown in the main window to switch between
configured machines. All settings, including the selected machine, are
remembered between sessions.

### Duplicating a Machine

To create a similar machine profile:

1. Select the machine to duplicate
2. Click the **Duplicate** button
3. Rename the new machine and adjust settings as needed

### Deleting a Machine

1. Select the machine to delete
2. Click the **Delete** button
3. Confirm the deletion

:::warning
Deleting a machine profile cannot be undone. Make sure you have
noted any important settings before deleting.
:::

## Related Topics

- [Machine Setup](../machine/general) - Detailed machine configuration
- [First Time Setup](../getting-started/first-time-setup) - Initial
  configuration guide
