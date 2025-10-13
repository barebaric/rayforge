# Snap Permissions (Linux)

This page explains how to configure permissions for Rayforge when installed as a Snap package on Linux.

## What are Snap Permissions?

Snaps are containerized applications that run in a sandbox for security. By default, they have limited access to system resources. To use certain features (like serial ports for laser controllers), you must explicitly grant permissions.

## Required Permissions

Rayforge needs these Snap interfaces connected for full functionality:

| Interface | Purpose | Required? |
|-----------|---------|-----------|
| `serial-port` | Access to USB serial devices (laser controllers) | **Yes** (for machine control) |
| `home` | Read/write files in your home directory | Auto-connected |
| `removable-media` | Access external drives and USB storage | Optional |
| `network` | Network connectivity (for updates, etc.) | Auto-connected |

---

## Granting Serial Port Access

**This is the most important permission for Rayforge.**

### Check Current Permissions

```bash
# View all connections for Rayforge
snap connections rayforge
```

Look for the `serial-port` interface. If it shows "disconnected" or "-", you need to connect it.

### Connect Serial Port Interface

```bash
# Grant serial port access
sudo snap connect rayforge:serial-port
```

**You only need to do this once.** The permission persists across app updates and reboots.

### Verify Connection

```bash
# Check if serial-port is now connected
snap connections rayforge | grep serial-port
```

Expected output:
```
serial-port     rayforge:serial-port     :serial-port     -
```

If you see a plug/slot indicator, the connection is active.

---

## Granting Removable Media Access

If you want to import/export files from USB drives or external storage:

```bash
# Grant access to removable media
sudo snap connect rayforge:removable-media
```

Now you can access files in `/media` and `/mnt`.

---

## Troubleshooting Snap Permissions

### Serial Port Still Not Working

**After connecting the interface:**

1. **Replug the USB device:**
   - Unplug your laser controller
   - Wait 5 seconds
   - Plug it back in

2. **Restart Rayforge:**
   - Close Rayforge completely
   - Relaunch from the application menu or:
     ```bash
     snap run rayforge
     ```

3. **Check that the port appears:**
   - Open Rayforge  Settings  Machine
   - Look for serial ports in the dropdown
   - Should see `/dev/ttyUSB0`, `/dev/ttyACM0`, or similar

4. **Verify the device exists:**
   ```bash
   # List USB serial devices
   ls -l /dev/ttyUSB* /dev/ttyACM*
   ```

### "Permission Denied" Despite Connected Interface

This is rare but can happen if:

1. **The Snap installation is broken:**
   ```bash
   # Reinstall the snap
   sudo snap refresh rayforge --devmode
   # Or if that fails:
   sudo snap remove rayforge
   sudo snap install rayforge
   # Re-connect interfaces
   sudo snap connect rayforge:serial-port
   ```

2. **Conflicting udev rules:**
   - Check `/etc/udev/rules.d/` for custom serial port rules
   - They might conflict with Snap's device access

3. **AppArmor denials:**
   ```bash
   # Check for AppArmor denials
   sudo journalctl -xe | grep DENIED | grep rayforge
   ```

   If you see denials for serial ports, there may be an AppArmor profile conflict.

### Can't Access Files Outside Home Directory

**By design**, Snaps can't access files outside your home directory unless you grant `removable-media`.

**Workaround options:**

1. **Move files to your home directory:**
   ```bash
   # Copy SVG files to ~/Documents
   cp /some/other/location/*.svg ~/Documents/
   ```

2. **Grant removable-media access:**
   ```bash
   sudo snap connect rayforge:removable-media
   ```

3. **Use Snap's file picker:**
   - The built-in file chooser has broader access
   - Open files through File  Open rather than command-line arguments

---

## Manual Interface Management

### List All Available Interfaces

```bash
# See all Snap interfaces on your system
snap interface
```

### Disconnect an Interface

```bash
# Disconnect serial-port (if needed)
sudo snap disconnect rayforge:serial-port
```

### Reconnect After Disconnect

```bash
sudo snap connect rayforge:serial-port
```

---

## Alternative: Install from Source

If Snap permissions are too restrictive for your workflow:

**Option 1: Build from source**

```bash
# Clone the repository
git clone https://github.com/kylemartin57/rayforge.git
cd rayforge

# Install dependencies using pixi
pixi install

# Run Rayforge
pixi run rayforge
```

**Benefits:**
- No permission restrictions
- Full system access
- Easier debugging
- Latest development version

**Drawbacks:**
- Manual updates (git pull)
- More dependencies to manage
- No automatic updates

**Option 2: Use Flatpak (if available)**

Flatpak has similar sandboxing but sometimes with different permission models. Check if Rayforge offers a Flatpak package.

---

## Snap Permission Best Practices

### Only Connect What You Need

Don't connect interfaces you don't use:

-  Connect `serial-port` if you use a laser controller
-  Connect `removable-media` if you import from USB drives
- L Don't connect everything "just in case" - defeats security purpose

### Verify Snap Source

Always install from the official Snap Store:

```bash
# Check publisher
snap info rayforge
```

Look for:
- Verified publisher
- Official repository source
- Regular updates

---

## Understanding Snap Sandbox

### What Can Snaps Access by Default?

**Allowed:**
- Files in your home directory
- Network connections
- Display/audio

**Not allowed without explicit permission:**
- Serial ports (USB devices)
- Removable media
- System files
- Other users' home directories

### Why This Matters for Rayforge

Rayforge needs:

1. **Home directory access** (auto-granted)
   - To save project files
   - To read imported SVG/DXF files
   - To store preferences

2. **Serial port access** (must be granted)
   - To communicate with laser controllers
   - **This is the critical permission**

3. **Removable media** (optional)
   - To import files from USB drives
   - To export G-code to external storage

---

## Debugging Snap Issues

### Enable Verbose Snap Logging

```bash
# Run Snap with debug output
snap run --shell rayforge
# Inside the snap shell:
export RAYFORGE_LOG_LEVEL=DEBUG
exec rayforge
```

### Check Snap Logs

```bash
# View Rayforge logs
snap logs rayforge

# Follow logs in real-time
snap logs -f rayforge
```

### Check System Journal for Denials

```bash
# Look for AppArmor denials
sudo journalctl -xe | grep DENIED | grep rayforge

# Look for USB device events
sudo journalctl -f -u snapd
# Then plug in your laser controller
```

---

## Getting Help

If you're still having Snap-related issues:

1. **Check permissions first:**
   ```bash
   snap connections rayforge
   ```

2. **Try a serial port test:**
   ```bash
   # If you have screen or minicom installed
   sudo snap connect rayforge:serial-port
   # Then test in Rayforge
   ```

3. **Report the issue with:**
   - Output of `snap connections rayforge`
   - Output of `snap version`
   - Output of `snap info rayforge`
   - Your Ubuntu/Linux distribution version
   - Exact error messages

4. **Consider alternatives:**
   - Install from source (see above)
   - Use a different package format (AppImage, Flatpak)

---

## Quick Reference Commands

```bash
# Grant serial port access (most important)
sudo snap connect rayforge:serial-port

# Grant removable media access
sudo snap connect rayforge:removable-media

# Check current connections
snap connections rayforge

# View Rayforge logs
snap logs rayforge

# Refresh/update Rayforge
sudo snap refresh rayforge

# Remove and reinstall (last resort)
sudo snap remove rayforge
sudo snap install rayforge
sudo snap connect rayforge:serial-port
```

---

## Related Pages

- [Connection Issues](connection.md) - Serial connection troubleshooting
- [Common Problems](common.md) - General troubleshooting
- [Installation](../getting-started/installation.md) - Installation guide
- [Device Configuration](../machine/device-config.md) - Machine setup
