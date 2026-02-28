# First Time Setup

After installing Rayforge, you'll need to configure your laser cutter or engraver. This guide will walk you through creating your first machine and establishing a connection.

## Step 1: Launch Rayforge

Start Rayforge from your application menu or by running `rayforge` in a terminal. You should see the main interface with an empty canvas.

## Step 2: Create a Machine

Navigate to **Settings → Machines** or press <kbd>ctrl+comma</kbd> to open the settings dialog, then select the **Machines** page.

Click **Add Machine** to create a new machine. You can either:

1. **Choose a built-in profile** - Select from predefined machine templates
2. **Select "Custom"** - Start with a blank configuration

After selecting, the Machine Settings dialog opens for your new machine.

![Machine Settings](/screenshots/application-machines.png)

## Step 3: Configure General Settings

The **General** page contains basic machine information, driver selection, and connection settings.

![General Settings](/screenshots/machine-general.png)

### Machine Information

1. **Machine Name**: Give your machine a descriptive name (e.g., "K40 Laser", "Ortur LM2")

### Driver Selection

Select the appropriate driver for your device from the dropdown:

- **GRBL Serial** - For GRBL devices connected via USB/serial port
- **GRBL Network** - For GRBL devices with WiFi/Ethernet connectivity
- **Smoothie** - For Smoothieware-based devices

### Driver Settings

Depending on your selected driver, configure the connection parameters:

#### GRBL Serial (USB)

1. **Port**: Choose your device from the dropdown (e.g., `/dev/ttyUSB0` on Linux, `COM3` on Windows)
2. **Baud Rate**: Select `115200` (standard for most GRBL devices)

:::info
If your device doesn't appear in the list, check that it's connected and that you have the necessary permissions. On Linux, you may need to add your user to the `dialout` group.
:::

#### GRBL Network / Smoothie (WiFi/Ethernet)

1. **Host**: Enter the IP address of your device (e.g., `192.168.1.100`)
2. **Port**: Enter the port number (typically `23` or `8080`)

### Speeds & Acceleration

These settings are used for job time estimation and path optimization:

- **Max Travel Speed**: Maximum rapid movement speed
- **Max Cut Speed**: Maximum cutting speed
- **Acceleration**: Used for time estimations and overscan calculations

## Step 4: Configure Hardware Settings

Switch to the **Hardware** tab to set up your machine's physical dimensions.

![Hardware Settings](/screenshots/machine-hardware.png)

### Dimensions

- **Width**: Enter the maximum width of your working area in millimeters
- **Height**: Enter the maximum height of your working area in millimeters

### Axes

- **Coordinate Origin (0,0)**: Select where your machine's origin is located:
  - Bottom Left (most common for GRBL)
  - Top Left
  - Top Right
  - Bottom Right

### Axis Offsets (Optional)

Configure X and Y offsets if your machine requires them for precise positioning.

## Step 5: Automatic Connection

Rayforge automatically connects to your machine when the application starts (if the machine is powered on and connected). You don't need to manually click a connect button.

The connection status is displayed in the bottom-left corner of the main window with a status icon and label showing the current state (Connected, Connecting, Disconnected, Error, etc.).

:::success Connected!
If your machine shows "Connected" status, you're ready to start using Rayforge!
:::

## Optional: Advanced Configuration

### Multiple Lasers

If your machine has multiple laser modules (e.g., diode and CO2), you can configure them in the **Laser** page.

![Laser Settings](/screenshots/machine-laser.png)

See [Laser Configuration](../machine/laser) for details.

### Camera Setup

If you have a USB camera for alignment and positioning, configure it in the **Camera** page.

![Camera Settings](/screenshots/machine-camera.png)

See [Camera Integration](../machine/camera) for details.

### Device Settings

The **Device** page allows you to read and modify firmware settings directly on your connected device (such as GRBL parameters). This is an advanced feature and should be used with caution.

:::warning
Editing device settings can be dangerous and may render your machine inoperable if incorrect values are applied!
:::

---

## Troubleshooting Connection Issues

### Device Not Found

- **Linux (Serial)**: Add your user to the `dialout` group. This is required
  for **both Snap and non-Snap installations** on Debian-based distributions
  to avoid AppArmor DENIED messages:
  ```bash
  sudo usermod -a -G dialout $USER
  ```
  Log out and back in for changes to take effect.

- **Snap Package**: In addition to the `dialout` group above, ensure you've
  granted serial port permissions:
  ```bash
  sudo snap connect rayforge:serial-port
  ```

- **Windows**: Check Device Manager to confirm the device is recognized and
  note the COM port number.

### Connection Refused

- Verify the IP address and port number are correct
- Ensure your machine is powered on and connected to the network
- Check firewall settings if using network connection

### Machine Not Responding

- Try a different baud rate (some devices use `9600` or `57600`)
- Check for loose cables or poor connections
- Power cycle your laser cutter and try again

For more help, see [Connection Issues](../troubleshooting/connection).

---

**Next:** [Quick Start Guide →](quick-start)
