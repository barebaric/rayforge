# Laser Settings

The Laser page in Machine Settings configures your laser head(s) and their properties.

![Laser Settings](/images/machine-laser.png)

## Laser Heads

Rayforge supports machines with multiple laser heads. Each laser head has its own configuration.

### Adding a Laser Head

Click the **Add Laser** button to create a new laser head configuration.

### Laser Head Properties

Each laser head has the following settings:

#### Name

A descriptive name for this laser head.

Examples:
- "10W Diode"
- "CO2 Tube"
- "Infrared Laser"

#### Tool Number

The tool index for this laser head. Used in G-code with the T command.

- Single-head machines: Use 0
- Multi-head machines: Assign unique numbers (0, 1, 2, etc.)

#### Maximum Power

The maximum power value for your laser.

- **GRBL typical**: 1000 (S0-S1000 range)
- **Some controllers**: 255 (S0-S255 range)
- **Percentage mode**: 100 (S0-S100 range)

This value should match your firmware's $30 setting.

#### Frame Power

The power level used for framing operations (outlining without cutting).

- Set to 0 to disable framing
- Typical values: 5-20 (just visible, won't mark material)
- Adjust based on your laser and material

#### Spot Size

The physical size of your focused laser beam in millimeters.

- Enter both X and Y dimensions
- Most lasers have a circular spot (e.g., 0.1 x 0.1)
- Affects engraving quality calculations

:::tip Measuring Spot Size
To measure your spot size:
1. Fire a short pulse at low power on a test material
2. Measure the resulting mark with calipers
3. Use the average of multiple measurements
:::

## See Also

- [Device Settings](device) - GRBL laser mode settings
