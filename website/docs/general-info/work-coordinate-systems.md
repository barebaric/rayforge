# Work Coordinate Systems (WCS)

Work Coordinate Systems (WCS) allow you to define multiple reference points on
your machine's work area. This makes it easy to run the same job at
different positions without redesigning or repositioning your workpieces.

## Understanding WCS

Think of WCS as customizable "zero points" for your work. While your
machine has a fixed home position (determined by limit switches), WCS lets
you define where you want your work to start from.

### Why Use WCS?

- **Multiple fixtures**: Set up several work areas on your bed and switch
  between them
- **Repeatable positioning**: Run the same job in different locations
- **Quick alignment**: Set a reference point based on your material or
  workpiece
- **Production workflows**: Organize multiple jobs across your work area

## WCS Types

Rayforge supports the following coordinate systems:

| System  | Type    | Description                                             |
| ------- | ------- | ------------------------------------------------------- |
| **G53** | Machine | Absolute machine coordinates (fixed, cannot be changed) |
| **G54** | Work 1  | First work coordinate system (default)                  |
| **G55** | Work 2  | Second work coordinate system                           |
| **G56** | Work 3  | Third work coordinate system                            |
| **G57** | Work 4  | Fourth work coordinate system                           |
| **G58** | Work 5  | Fifth work coordinate system                            |
| **G59** | Work 6  | Sixth work coordinate system                            |

### Machine Coordinates (G53)

G53 represents the absolute position of your machine, with zero at the
machine's home position. This is fixed by your hardware and cannot be
changed.

**When to use:**

- Homing and calibration
- Absolute positioning relative to machine limits
- When you need to reference the physical machine position

### Work Coordinates (G54-G59)

These are offset coordinate systems that you can define. Each has its own
zero point that you can set anywhere on your work area.

**When to use:**

- Setting up multiple work fixtures
- Aligning to material positions
- Running the same job in different locations

## Visualizing WCS in the Interface

### 2D Canvas

The 2D canvas shows your WCS origin with a green marker:

- **Green lines**: Indicate the current WCS origin (0, 0) position
- **Grid alignment**: Grid lines are aligned to the WCS origin, not machine
  origin

The origin marker moves when you change the active WCS or its offset,
showing you exactly where your work will start.

### 3D Preview

In the 3D preview, WCS is displayed differently:

- **Grid and axes**: The entire grid appears as if the WCS origin is the
  world origin
- **Isolated view**: The WCS is shown "in isolation" - it looks like the
  grid is centered on the WCS, not the machine
- **Labels**: Coordinate labels are relative to the WCS origin

This makes it easy to visualize where your job will run relative to the
selected work coordinate system.

## Selecting and Changing WCS

### Via the Toolbar

1. Locate the WCS dropdown in the main toolbar (labeled "G53" by default)
2. Click to see the available coordinate systems
3. Select the WCS you want to use

### Via the Control Panel

1. Open the Control Panel (View → Control Panel or Ctrl+L)
2. Find the WCS dropdown in the machine status section
3. Select the desired WCS from the dropdown

## Setting WCS Offsets

You can define where each WCS origin is located on your machine.

### Setting Zero at Current Position

1. Connect to your machine
2. Select the WCS you want to configure (e.g., G54)
3. Jog the laser head to the position you want to be (0, 0)
4. In the Control Panel, click the zero buttons:
   - **Zero X**: Sets current X position as 0 for the active WCS
   - **Zero Y**: Sets current Y position as 0 for the active WCS
   - **Zero Z**: Sets current Z position as 0 for the active WCS

The offsets are stored in your machine's controller and persist between
sessions.

### Viewing Current Offsets

The Control Panel shows the current offsets for the active WCS:

- **Current Offsets**: Displays the (X, Y, Z) offset from machine origin
- **Current Position**: Shows the laser head's position in the active WCS

## WCS in Your Jobs

When you run a job, Rayforge uses the active WCS to position your work:

1. Design your job in the canvas
2. Select the WCS you want to use
3. Run the job - it will be positioned according to the WCS offset

The same job can be run at different positions simply by changing the
active WCS.

## Practical Workflows

### Workflow 1: Multiple Fixture Positions

You have a large bed and want to set up three work areas:

1. **Home your machine** to establish a reference
2. **Jog to first work area** and set G54 offset (Zero X, Zero Y)
3. **Jog to second work area** and set G55 offset
4. **Jog to third work area** and set G56 offset
5. Now you can switch between G54, G55, and G56 to run jobs in each area

### Workflow 2: Aligning to Material

You have a piece of material placed somewhere on your bed:

1. **Jog the laser head** to the corner of your material
2. **Select G54** (or your preferred WCS)
3. **Click Zero X and Zero Y** to set the material corner as (0, 0)
4. **Design your job** with (0, 0) as the origin
5. **Run the job** - it will start from the material corner

### Workflow 3: Production Grid

You need to cut the same part 10 times in different locations:

1. **Design one part** in Rayforge
2. **Set up G54-G59** offsets for your desired positions
3. **Run the job** with G54 active
4. **Switch to G55** and run again
5. **Repeat** for each WCS position

## Important Notes

### WCS Limitations

- **G53 cannot be changed**: Machine coordinates are fixed by hardware
- **Offsets persist**: WCS offsets are stored in your machine's controller
- **Connection required**: You must be connected to a machine to set WCS
  offsets

### WCS and Job Origin

WCS works independently of your job origin settings. The job origin determines
where on the canvas your job starts, while WCS determines where that
canvas position maps to on your machine.

### Machine Compatibility

Not all machines support all WCS features:

- **GRBL (v1.1+)**: Full support for G53-G59
- **Smoothieware**: Supports G54-G59 (offset reading may be limited)
- **Custom controllers**: Varies by implementation

---

**Related Pages:**

- [Coordinate Systems](coordinate-systems) - Understanding coordinate systems
- [Control Panel](../ui/control-panel) - Manual control and WCS management
- [Machine Setup](../machine/general) - Configure your machine
- [3D Preview](../ui/3d-preview) - Visualizing your jobs
