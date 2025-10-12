# Troubleshooting

Having issues with Rayforge? This section covers common problems and their solutions.

## Quick Fixes

Before diving into specific issues, try these general troubleshooting steps:

1. **Restart Rayforge**: Close and reopen the application
2. **Check connections**: Ensure cables are firmly connected
3. **Power cycle**: Turn your laser cutter off and on
4. **Update software**: Check for the latest Rayforge version
5. **Check permissions**: Verify you have the necessary hardware access permissions

## Common Issues

### Connection Problems

Can't connect to your laser cutter? See the [Connection Issues](connection.md) guide.

**Quick checklist:**

- [ ] Device is powered on
- [ ] USB cable is connected (or network is accessible)
- [ ] Correct port/IP address is selected
- [ ] Baud rate matches device settings (usually 115200)
- [ ] User has permission to access serial ports (Linux)

### Snap Package Permissions

Using the Snap version? You may need to grant hardware permissions. See [Snap Permissions](snap-permissions.md).

### Performance Issues

Experiencing slow performance? Check the [Performance](performance.md) guide.

**Common causes:**

- Large or complex files
- Insufficient system resources
- Too many operations or layers
- High-resolution raster images

### Common Problems

See [Common Problems](common.md) for solutions to frequently reported issues including:

- G-code generation errors
- Interface glitches
- File import problems
- Unexpected behavior

## Getting Help

If you can't find a solution here:

1. **Search existing issues**: Check [GitHub Issues](https://github.com/barebaric/rayforge/issues) to see if others have reported the same problem
2. **Report a new issue**: If your problem is not listed, [create a new issue](https://github.com/barebaric/rayforge/issues/new) with:
   - Detailed description of the problem
   - Steps to reproduce
   - Your system information (OS, Rayforge version)
   - Any error messages
   - Screenshots if applicable

## Diagnostic Information

When reporting issues, include this information:

### System Information

- **Operating System**: (e.g., Ubuntu 24.04, Windows 11)
- **Rayforge Version**: Check Help → About
- **Installation Method**: (PPA, Snap, pip, Windows installer)

### Machine Information

- **Device Type**: (GRBL, Smoothieware)
- **Connection Method**: (Serial, Network, Telnet)
- **Firmware Version**: Check Machine → GRBL Settings

### Log Files

Rayforge logs can help diagnose issues:

- **Linux**: `~/.local/share/rayforge/logs/`
- **Windows**: `%APPDATA%\rayforge\logs\`

Include relevant log entries when reporting issues.

---

## Issue Categories

- **[Connection Issues](connection.md)**: Can't connect to your machine
- **[Snap Permissions](snap-permissions.md)**: Permission problems with Snap package
- **[Performance](performance.md)**: Slow operation or lag
- **[Common Problems](common.md)**: Frequently reported issues and solutions
