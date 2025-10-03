# Material Test Generator - Design Specification

## Overview

The Material Test Generator creates parametric test grids for laser material testing. It generates test arrays with varying speed and power settings to help users find optimal parameters for different materials.

## Architecture

### Core Components

#### 1. MaterialTestGridProducer (OpsProducer)
- **Purpose**: Generates test grid operations directly from parameters
- **Key Feature**: Does not require pixel data or rendering
- **Parameters**:
  - Test type: Cut or Engrave (enum: `TestType.CUT`, `TestType.ENGRAVE`)
  - Speed range: (min, max) in mm/min
  - Power range: (min, max) in percentage (0-100)
  - Grid dimensions: (columns, rows)
  - Shape size: Size of each test square in mm
  - Spacing: Gap between squares in mm
  - Include labels: Boolean for axis labels

- **Execution Order**: Risk-optimized (highest speed first, then lowest power)
- **Coordinate System**: Millimeter space
- **Scalability**: Mathematically scalable

#### 2. MaterialTestRenderer (Renderer)
- **Purpose**: Renders visual preview for canvas display
- **Input**: JSON-encoded parameters from ImportSource data
- **Output**: Cairo ImageSurface with:
  - Grid cells with gradient shading (darker = more intense)
  - Speed and power labels (if enabled)
  - Axis labels

#### 3. ImportSource Integration
- **Source File**: Path("[material-test]") (virtual path)
- **Data**: JSON-encoded producer parameters
- **Metadata**: `{"type": "material_test"}`
- **Renderer**: MaterialTestRenderer instance

### Data Flow

```
User Parameters → MaterialTestGridProducer → Ops (with speed/power commands)
                ↓
          ImportSource (JSON params) → MaterialTestRenderer → Canvas Preview
                ↓
          WorkPiece (references source) → Layer (regular) → Document
```

## UI Components

### 1. Material Test Grid Settings Widget
- **Location**: Step settings dialog
- **Base Class**: `StepComponentSettingsWidget`
- **Class Property**: `show_general_settings = False` (hides power/speed settings)
- **Features**:
  - Preset selector with common configurations (Diode Cut/Engrave, CO2 Cut/Engrave)
  - Speed range controls (min/max)
  - Power range controls (min/max)
  - Grid dimensions (columns/rows)
  - Shape size and spacing sliders
  - Labels toggle
  - Real-time updates with undo/redo support

## Layer Integration

- **Layer Type**: Material test workpieces live in regular `Layer` objects
- **Mixing Operations**: Material test steps can be mixed with other steps in the workflow
- **Flexibility**: Users can combine material tests with cuts, engraves, and other operations

## Test Pattern Generation

### Grid Layout
- Cells arranged in columns (speed variations) × rows (power variations)
- Each cell represents a unique speed/power combination
- Labels extend into negative coordinate space (-15mm margin)

### Execution Order (Risk Optimization)
1. Sort cells by: highest speed (safest) first
2. Secondary sort: lowest power within same speed
3. Minimizes risk of material charring from low-speed, high-power cells

### Label Generation
- **Axis Labels**: "speed (mm/min)" and "power (%)"
- **Numeric Labels**: Speed values on X-axis, power values on Y-axis
- **Rendering**: Cairo text paths converted to vector ops
- **Settings**: 10% power, 1000 mm/min speed for label engraving

## Presets

Default configurations for common scenarios:
- **Diode Engrave**: 1000-10000 mm/min, 10-100% power
- **Diode Cut**: 100-5000 mm/min, 50-100% power
- **CO2 Engrave**: 3000-20000 mm/min, 10-50% power
- **CO2 Cut**: 1000-20000 mm/min, 30-100% power

Presets default to "(None)" - only apply when explicitly selected.

## Size Calculations

**Total Grid Size**:
- Width = columns × (shape_size + spacing) - spacing
- Height = rows × (shape_size + spacing) - spacing

**With Labels**:
- Add 15mm margin to left and top for label space

## Serialization

### Producer Dictionary
```
{
  "type": "MaterialTestGridProducer",
  "params": {
    "test_type": "Cut",  // or "Engrave" (string value from enum)
    "speed_range": [min, max],
    "power_range": [min, max],
    "grid_dimensions": [cols, rows],
    "shape_size": float,
    "spacing": float,
    "include_labels": bool
  }
}
```

### WorkPiece with Material Test Source
```
{
  "type": "workpiece",
  "uid": string,
  "name": string,
  "import_source_uid": string,  // References ImportSource with type="material_test"
  "matrix": [...],
  "size": [width, height]
}
```

**Note**: Material test workpieces are identified by their ImportSource metadata `{"type": "material_test"}`, not by layer type.

## Integration Points

### Pipeline Integration
- Registered in `rayforge/pipeline/producer/__init__.py`
- Step factory: `create_material_test_step()` in `rayforge/pipeline/steps.py`
- No modifiers or transformers needed (ops pre-generated)

### UI Registration
- Settings widget: `WIDGET_REGISTRY["MaterialTestGridProducer"]`
- Renderer: Automatic via ImportSource renderer property

### Document Integration
- Material test workpieces live in regular `Layer` objects
- No special layer type required
- Backward compatibility: Legacy `"materialtestlayer"` types are converted to regular `Layer` on load

## Design Decisions

### Why Not Subclass WorkPiece?
- ✅ Cleaner: Uses existing ImportSource mechanism
- ✅ Standard: No new core data model types
- ✅ Serialization: Leverages existing patterns

### Why NOT a Separate Layer Type?
- ✅ Better discoverability: Material tests found in normal workflow
- ✅ Flexibility: Can mix material tests with other operations (cut outline, then test grid)
- ✅ Simpler architecture: One less core object type to maintain
- ✅ Reduced complexity: No special cases in layer management UI



## Future Enhancements

- Advanced preset management UI (save load custom presets)
- Test result database integration
- QR code added to pattern encoding the machine id and the material id of that test run
- Automated image analysis of a picture of the laser cut/engraved test grid determines the best cut / engrave settings which are automatically linked to corresponding machine and material ids via the QR code added to the test grid pattern. 
- Automated wizard to help you quickly fine-tune settings for a particular machine and material using a bisection-search strategy that runs a series of smaller test-grids dynamically chosen based on the results of past test-grids to more efficiently identify the best settings.


