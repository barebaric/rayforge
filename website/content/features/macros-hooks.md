# G-code Macros & Hooks

RayForge provides two powerful automation features for customizing your workflow: **Macros** and **Hooks**. Both allow you to inject custom G-code into your jobs, but they serve different purposes.

---

## Overview

| Feature | Purpose | Trigger | Use Case |
|---------|---------|---------|----------|
| **Macros** | Reusable G-code snippets | Manual execution | Quick commands, test patterns, custom routines |
| **Hooks** | Automatic G-code injection | Job lifecycle events | Startup sequences, layer changes, cleanup |

---

## Macros

Macros are **named, reusable G-code scripts** that you can execute manually at any time.

### What Are Macros For?

Common macro use cases:

- **Homing the machine** - Send `$H` quickly
- **Setting work offsets** - Store and recall G54/G55 positions
- **Air assist control** - Toggle air assist on/off
- **Focus testing** - Run a quick focus test pattern
- **Custom tool changes** - For multi-laser setups
- **Emergency routines** - Quick shutdown or alarm clear
- **Material probing** - Auto-focus or height measurement

### Creating a Macro

1. **Open Machine Settings:**
   - Navigate to **Settings  Machine  Macros**

2. **Add a new macro:**
   - Click the **"+"** button
   - Enter a descriptive name (e.g., "Home Machine", "Enable Air Assist")

3. **Write your G-code:**
   - Each line is a separate G-code command
   - Comments start with `;` or `(`
   - Variables can be used (see Variable Substitution below)

4. **Save the macro**

5. **Execute the macro:**
   - From the macros list, click the macro
   - Or assign a keyboard shortcut (if supported)

### Example Macros

#### Simple: Home the Machine

**Name:** Home Machine
**Code:**
```gcode
$H
; Waits for homing to complete
```

**Use:** Quickly home the machine before starting work.

---

#### Medium: Set Work Offset

**Name:** Set G54 to Current Position
**Code:**
```gcode
G10 L20 P1 X0 Y0
; Sets G54 work coordinate system origin to current position
```

**Use:** Mark the current laser position as the job origin.

---

#### Advanced: Focus Test Grid

**Name:** 9-Point Focus Test
**Code:**
```gcode
; 9-point grid for finding optimal focus
G21  ; Millimeters
G90  ; Absolute positioning
G0 X10 Y10
M3 S1000
G4 P0.1
M5
G0 X20 Y10
M3 S1000
G4 P0.1
M5
; ... (repeat for remaining points)
```

**Use:** Quickly test focus at different positions on the bed.

---

### Macro Best Practices

**Naming:**
- Use descriptive names: "Home Machine" not "Macro 1"
- Group related macros: "Air Assist ON", "Air Assist OFF"

**Safety:**
- **Always** include safety checks (position limits, feedrate limits)
- Add comments explaining what each command does
- Test macros in simulation/air mode first

**Organization:**
- Keep macros simple and focused on one task
- Create separate macros for on/off or start/stop operations
- Document required machine state (e.g., "Must be homed first")

**Error Handling:**
- Macros don't have built-in error handling
- If a command fails, subsequent commands may still execute
- Use conservative settings to avoid crashes

---

## Hooks

Hooks are **automatic G-code injections** triggered by specific events during job execution.

### Hook Triggers

RayForge supports these hook triggers:

| Trigger | When It Runs | Common Uses |
|---------|--------------|-------------|
| **Job Start** | Very beginning of the job | Homing, work offset, air assist on, preheat |
| **Job End** | Very end of the job | Return home, air assist off, beep, cooldown |
| **Layer Start** | Before processing each layer | Tool change, power adjust, comments |
| **Layer End** | After processing each layer | Progress notification, pause |
| **Workpiece Start** | Before processing each workpiece | Part numbering, alignment marks |
| **Workpiece End** | After processing each workpiece | Cooldown, inspection pause |

### Creating a Hook

1. **Open Machine Settings:**
   - Navigate to **Settings  Machine  Hooks**

2. **Select a trigger:**
   - Choose the event when this hook should run

3. **Write your G-code:**
   - Hook code is injected at the trigger point
   - Use variables for dynamic values (see below)

4. **Enable/disable:**
   - Toggle hooks on/off without deleting them

### Example Hooks

#### Job Start: Initialize Machine

**Trigger:** Job Start
**Code:**
```gcode
G21         ; Millimeters
G90         ; Absolute positioning
$H          ; Home the machine
G0 X0 Y0    ; Move to origin
M3 S0       ; Laser on but power 0 (some controllers need this)
M8          ; Air assist ON
```

**Purpose:** Ensures machine is in a known state before every job.

---

#### Job End: Return Home and Beep

**Trigger:** Job End
**Code:**
```gcode
M5          ; Laser OFF
M9          ; Air assist OFF
G0 X0 Y0    ; Return to origin
M300 S800 P200  ; Beep (if supported)
```

**Purpose:** Safely ends the job and signals completion.

---

#### Layer Start: Add Comment

**Trigger:** Layer Start
**Code:**
```gcode
; Starting layer: {layer_name}
; Layer index: {layer_index}
```

**Purpose:** Makes G-code more readable for debugging.

---

#### Workpiece Start: Part Numbering

**Trigger:** Workpiece Start
**Code:**
```gcode
; Part: {workpiece_name}
; Part {workpiece_index} of {total_workpieces}
```

**Purpose:** Track progress in multi-part jobs.

---

### Hook Execution Order

For a job with 2 layers, each with 2 workpieces:

```
[Job Start Hook]
  [Layer Start Hook] (Layer 1)
    [Workpiece Start Hook] (Workpiece 1)
      ... workpiece 1 G-code ...
    [Workpiece End Hook] (Workpiece 1)
    [Workpiece Start Hook] (Workpiece 2)
      ... workpiece 2 G-code ...
    [Workpiece End Hook] (Workpiece 2)
  [Layer End Hook] (Layer 1)
  [Layer Start Hook] (Layer 2)
    [Workpiece Start Hook] (Workpiece 3)
      ... workpiece 3 G-code ...
    [Workpiece End Hook] (Workpiece 3)
    [Workpiece Start Hook] (Workpiece 4)
      ... workpiece 4 G-code ...
    [Workpiece End Hook] (Workpiece 4)
  [Layer End Hook] (Layer 2)
[Job End Hook]
```

---

## Variable Substitution

Both macros and hooks support **variable substitution** to inject dynamic values.

### Available Variables

Variables use `{variable_name}` syntax and are replaced during G-code generation.

**Job-level variables:**

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `{job_name}` | Name of the current job/document | "test-job" |
| `{date}` | Current date | "2025-10-03" |
| `{time}` | Current time | "14:30:25" |

**Layer-level variables:**

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `{layer_name}` | Name of the current layer | "Cut Layer" |
| `{layer_index}` | Zero-based index of current layer | 0, 1, 2... |
| `{total_layers}` | Total number of layers in job | 3 |

**Workpiece-level variables:**

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `{workpiece_name}` | Name of the workpiece | "Circle 1" |
| `{workpiece_index}` | Zero-based index of current workpiece | 0, 1, 2... |
| `{total_workpieces}` | Total number of workpieces | 5 |

**Machine variables:**

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `{machine_name}` | Name of the machine profile | "My K40" |
| `{max_speed}` | Maximum cutting speed (mm/min) | 1000 |
| `{work_width}` | Work area width (mm) | 300 |
| `{work_height}` | Work area height (mm) | 200 |

### Example: Progress Notification

**Hook:** Layer Start
**Code:**
```gcode
; ========================================
; Layer {layer_index} of {total_layers}: {layer_name}
; Job: {job_name}
; Time: {time}
; ========================================
```

**Result in G-code:**
```gcode
; ========================================
; Layer 0 of 3: Cut Layer
; Job: test-project
; Time: 14:30:25
; ========================================
```

---

## Advanced Use Cases

### Multi-Tool Setup

For machines with multiple lasers or tools:

**Hook:** Workpiece Start
**Code:**
```gcode
; Select tool for workpiece {workpiece_name}
T{tool_number}  ; Tool change command (if supported)
G4 P1           ; Wait for tool change
```

### Conditional Pauses

Add optional pauses for inspection:

**Hook:** Layer End
**Code:**
```gcode
; M0  ; Uncomment to pause after each layer for inspection
```

### Air Assist Per Layer

Control air assist on a per-layer basis:

**Hook:** Layer Start (for cutting layers)
**Code:**
```gcode
M8  ; Air assist ON
```

**Hook:** Layer Start (for engraving layers)
**Code:**
```gcode
M9  ; Air assist OFF (prevents dust scattering for engraving)
```

!!! note "Layer-Specific Hooks"
    RayForge doesn't currently support per-layer hook customization. To achieve this, use conditional G-code or separate machine profiles.

---

## Troubleshooting

### Macro Doesn't Execute

**Problem:** Clicking a macro does nothing.

**Solutions:**

1. **Check connection** - Machine must be connected
2. **Check macro syntax** - Invalid G-code may be rejected
3. **Check machine state** - Some commands require specific states (not alarmed, homed, etc.)
4. **Check logs** - View debug logs for error messages

### Hook Not Appearing in G-code

**Problem:** Hook is enabled but doesn't show in generated G-code.

**Solutions:**

1. **Verify hook is enabled** - Check the toggle switch
2. **Regenerate G-code** - Make a small change to force regeneration
3. **Check trigger applies** - e.g., Workpiece Start requires at least one workpiece
4. **Look in the right place** - Search the G-code for hook content

### Variables Not Substituting

**Problem:** Variables appear as `{variable_name}` instead of values.

**Causes:**

1. **Typo in variable name** - Must match exactly (case-sensitive)
2. **Variable not available** - e.g., `{layer_name}` in Job Start hook (no layer yet)
3. **Escape characters** - Use `{{` to insert a literal `{`

**Solution:** Double-check variable names against the table above.

### G-code Syntax Errors

**Problem:** Generated G-code has errors due to hooks/macros.

**Diagnosis:**

- Export G-code and review the hook injection points
- Look for:
  - Missing newlines (each command should be on its own line)
  - Invalid command syntax
  - Commands not supported by your firmware

**Solutions:**

- Test hook G-code in a macro first
- Use comments liberally to identify problematic sections
- Validate against GRBL command reference

---

## Safety Considerations

!!! danger "Test Before Production"
    Always test macros and hooks in **simulation mode** or with the laser **disabled** before running on real jobs. Incorrectly configured G-code can:

    - Crash the machine into limits
    - Fire the laser unexpectedly
    - Damage materials or equipment

**Safety checklist:**

- [ ] Macros include feedrate limits (`F` parameter)
- [ ] Macros check position bounds
- [ ] Job Start hooks include `M5` or laser off command
- [ ] Job End hooks turn off laser (`M5`) and air assist (`M9`)
- [ ] No destructive commands without confirmation
- [ ] Tested in simulation or with laser disabled

---

## Best Practices Summary

### Macros

- **Keep simple** - One task per macro
- **Use comments** - Explain what each command does
- **Test thoroughly** - Verify on real hardware
- **Name clearly** - Descriptive names save time

### Hooks

- **Be conservative** - Hooks run on every job automatically
- **Add safety checks** - Ensure machine state is correct
- **Use variables** - Make hooks generic and reusable
- **Document triggers** - Note which hook runs when

### Variables

- **Verify availability** - Not all variables available at all triggers
- **Handle missing values** - Some variables may be empty
- **Test edge cases** - Single layer/workpiece jobs, empty names, etc.

---

## Related Pages

- [GRBL Settings](../machine/grbl-settings.md) - GRBL command reference
- [G-code Dialects](../reference/gcode-dialects.md) - G-code compatibility
- [Machine Setup](../machine/index.md) - Machine configuration
- [Multi-Layer Workflow](multi-layer.md) - Using hooks with layers
