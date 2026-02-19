# General Settings

The General page in Machine Settings contains basic machine information and speed settings.

![General Settings](/images/machine-general.png)

## Machine Name

Give your machine a descriptive name. This helps identify the machine in the machine selector dropdown when you have multiple machines configured.

Examples:
- "Workshop K40"
- "Garage Diode Laser"
- "Ortur LM2 Pro"

## Speeds & Acceleration

These settings control the maximum speeds and acceleration for motion planning and time estimation.

### Max Travel Speed

The maximum speed for rapid (non-cutting) movements. This is used when the laser is off and the head is moving to a new position.

- **Typical range**: 2000-5000 mm/min
- **Purpose**: Motion planning and time estimation
- **Note**: Actual speed is also limited by your firmware settings

### Max Cut Speed

The maximum speed allowed during cutting or engraving operations.

- **Typical range**: 500-2000 mm/min
- **Purpose**: Limits operation speeds for safety
- **Note**: Individual operations may use lower speeds

### Acceleration

The rate at which the machine accelerates and decelerates.

- **Typical range**: 500-2000 mm/s²
- **Purpose**: Time estimation and motion planning
- **Note**: Must match or be lower than firmware acceleration settings

:::tip
Start with conservative speed values and increase gradually. Observe your machine for belt skipping, motor stalling, or loss of positioning accuracy.
:::

## See Also

- [Hardware Settings](hardware) - Machine dimensions and axis configuration
- [Device Settings](device) - Connection and GRBL settings
