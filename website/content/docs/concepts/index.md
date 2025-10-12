# Concepts

This section explains fundamental concepts for understanding laser cutting and engraving with RayForge.

## Why Read This Section?

Understanding the underlying concepts helps you:

- Make better decisions about settings
- Troubleshoot problems independently
- Optimize quality and efficiency
- Understand why things work the way they do

---

## Core Concepts

### [Understanding Operations](understanding-operations.md)

Learn what operations are and when to use each type.

**Topics covered:**

- Contour vs Raster vs Depth operations
- When to use each operation type
- How operations process geometry
- Operation execution order

**Read this if:** You're confused about which operation to choose, or want to understand how RayForge generates toolpaths.

---

### [Coordinates and Origin](coordinates-and-origin.md)

Understand how RayForge handles coordinate systems and positioning.

**Topics covered:**

- Job origin vs machine origin vs workpiece origin
- Coordinate transformations
- Absolute vs relative positioning
- Common coordinate-related mistakes

**Read this if:** Your jobs appear in the wrong position, or you don't understand how positioning works.

---

### [Power vs Speed](power-vs-speed.md)

Learn the relationship between laser power, cutting speed, and material interaction.

**Topics covered:**

- How power and speed affect cutting
- Multi-pass strategies for thick materials
- Finding optimal settings
- Reading burn marks and adjusting

**Read this if:** You're struggling to find the right settings for your material, or cuts aren't going through.

---

### [G-code Basics](gcode-basics.md)

Understand what G-code is and how RayForge generates it.

**Topics covered:**

- What is G-code and how it controls machines
- Common G-code commands explained
- How RayForge generates G-code from operations
- When and how to edit G-code manually

**Read this if:** You want to understand the output, customize G-code, or troubleshoot machine behavior.

---

### [Laser Safety](laser-safety.md)

Essential safety practices for laser cutting and engraving.

**Topics covered:**

- Eye safety and laser classes
- Material hazards (never cut PVC!)
- Fire prevention and emergency procedures
- Proper ventilation and PPE
- Emergency stop procedures

**Read this if:** You're new to laser work, or want a safety refresher. **Everyone should read this.**

---

## Learning Paths

### For Beginners

1. Start with [Laser Safety](laser-safety.md) - **Mandatory**
2. Read [Understanding Operations](understanding-operations.md) - Understand what operations do
3. Read [Power vs Speed](power-vs-speed.md) - Learn to find settings
4. Skim [Coordinates and Origin](coordinates-and-origin.md) - Know it exists for troubleshooting
5. Skip [G-code Basics](gcode-basics.md) for now - Come back when curious

### For Intermediate Users

1. Review [Laser Safety](laser-safety.md) - Refresh your knowledge
2. Deep dive into [Power vs Speed](power-vs-speed.md) - Optimize your workflow
3. Read [Understanding Operations](understanding-operations.md) - Master operation selection
4. Study [G-code Basics](gcode-basics.md) - Understand the output
5. Read [Coordinates and Origin](coordinates-and-origin.md) as needed

### For Advanced Users

1. All articles - Deep understanding enables creative solutions
2. Focus on [G-code Basics](gcode-basics.md) - For customization
3. Master [Coordinates and Origin](coordinates-and-origin.md) - Complex positioning
4. Use [Power vs Speed](power-vs-speed.md) - Material science optimization

---

## Complementary Resources

### Practical Guides

Once you understand the concepts, apply them:

- [Quick Start](../getting-started/quick-start.md) - First job walkthrough
- [Troubleshooting](../troubleshooting/index.md) - Solving problems

### Reference Documentation

Look up specific details:

- [Keyboard Shortcuts](../reference/shortcuts.md) - Quick reference
- [G-code Dialects](../reference/gcode-dialects.md) - Dialect differences
- [Firmware Compatibility](../reference/firmware.md) - Controller support

### Feature Documentation

Learn about specific features:

- [Operations](../features/operations/index.md) - Detailed operation docs
- [Multi-Layer Workflow](../features/multi-layer.md) - Layer organization
- [Simulation Mode](../features/simulation-mode.md) - Preview execution

---

## How to Use This Section

### Quick Reference

**Jump directly to the concept** you need - each article is self-contained.

### Sequential Reading

**Read in order** for a structured learning experience:

1. Laser Safety (essential)
2. Understanding Operations
3. Power vs Speed
4. Coordinates and Origin
5. G-code Basics

### Problem-Driven Learning

**Use the index** to find concepts relevant to your current problem:

- **Job in wrong position?** → Coordinates and Origin
- **Not cutting through?** → Power vs Speed
- **Wrong operation type?** → Understanding Operations
- **Customize G-code?** → G-code Basics

---

## Contributing to Concepts

Found an explanation unclear? Have a better way to explain something?

**Help improve the documentation:**

1. Click "Edit this page" at the top
2. Make your improvements
3. Submit a pull request

Or open an issue on GitHub with suggestions.

---

## Glossary

Quick definitions of key terms:

**Operation** - A processing step that converts geometry to toolpaths (Contour, Raster, etc.)

**Toolpath** - The actual path the laser head follows

**G-code** - Machine control language that tells the controller what to do

**Origin** - The reference point (0,0) for positioning

**Power** - Laser intensity, typically 0-100%

**Speed** - How fast the laser head moves, in mm/min

**Feed rate** - Same as speed (CNC terminology)

**Workpiece** - An imported shape or image

**Layer** - A container for workpieces with a workflow

**Workflow** - The operation and transformers applied to a layer

For more terms, see the full glossary (if available) or search the documentation.

---

## Related Sections

- [Getting Started](../getting-started/index.md) - First steps with RayForge
- [Features](../features/index.md) - Feature documentation
- [Troubleshooting](../troubleshooting/index.md) - Problem solving
