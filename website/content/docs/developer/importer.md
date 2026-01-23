# Importer Architecture

This document describes the architecture of Rayforge's file import system,
which handles converting various file formats (SVG, DXF, PNG, PDF, etc.) into
Rayforge's document model.

## Table of Contents

- [Overview](#overview)
- [Import Pipeline](#import-pipeline)
- [Scan Method](#scan-method)
- [Coordinate Systems](#coordinate-systems)
- [Key Classes](#key-classes)
- [Creating a New Importer](#creating-a-new-importer)

---

## Overview

The import system is built around a four-phase pipeline that transforms raw file
data into fully-positioned document objects. Each phase has a specific
responsibility and produces well-defined data structures.

```mermaid
flowchart TD
    raw[raw file data] --> scan
    raw --> parse
    scan[Scan<br/>Importer.scan<br/>for metadata] -->|ImportManifest| manifest
    manifest["ImportManifest"]
    
    parse[Phase 1: Parse<br/>Importer.parse] -->|ParsingResult| vectorize
    vectorize[Phase 2: Vectorize<br/>Importer.vectorize] -->|VectorizationResult| layout
    layout[Phase 3: Layout<br/>NormalizationEngine] -->|List of LayoutItem| assemble
    assemble[Phase 4: Assemble<br/>ItemAssembler] -->|ImportPayload| result
    result["ImportResult<br/>final output"]
    
    style scan fill:#f3e5f5
    style manifest fill:#f3e5f5
    style parse fill:#e1f5fe
    style vectorize fill:#e1f5fe
    style layout fill:#fff3e0
    style assemble fill:#e8f5e9
```

---

## Import Pipeline

### Phase 1: Parse

**Method:** [`Importer.parse()`](../../rayforge/image/base_importer.py:188)

Extracts geometric facts from the file including bounds, coordinate system
details, and layer information.

**Output:** [`ParsingResult`](../../rayforge/image/structures.py:105)

- `document_bounds`: Total canvas size in Native Coordinates
- `native_unit_to_mm`: Conversion factor to millimeters
- `is_y_down`: Y-axis orientation flag
- `layers`: List of [`LayerGeometry`](../../rayforge/image/structures.py:71)
- `world_frame_of_reference`: World Coordinates (mm, Y-Up)
- `background_world_transform`: Matrix for background positioning
- `untrimmed_document_bounds`: Reference for Y-inversion

**Coordinate System:**
- `document_bounds`: Native Coordinates (file-specific)
- `world_frame_of_reference`: World Coordinates (mm, Y-Up)

---

### Phase 2: Vectorize

**Method:** [`Importer.vectorize()`](../../rayforge/image/base_importer.py:225)

Converts parsed data into vector [`Geometry`](../../rayforge/core/geo/__init__.py)
objects according to the [`VectorizationSpec`](../../rayforge/core/vectorization_spec/__init__.py).

**Output:** [`VectorizationResult`](../../rayforge/image/structures.py:176)

- `geometries_by_layer`: Vector geometry per layer (Native Coordinates)
- `source_parse_result`: Reference to original ParsingResult
- `fills_by_layer`: Optional fill geometry (Sketch importer)

**Coordinate System:** Native Coordinates (file-specific)

---

### Phase 3: Layout

**Class:** [`NormalizationEngine`](../../rayforge/image/engine.py:19)

Calculates transformation matrices to map Native Coordinates to World
Coordinates based on user intent.

**Output:** `List[`[`LayoutItem`](../../rayforge/image/structures.py:217)`]`

Each `LayoutItem` contains:
- `world_matrix`: Normalized (0-1, Y-Up) → World (mm, Y-Up)
- `normalization_matrix`: Native → Normalized (0-1, Y-Up)
- `crop_window`: Subset of original file in Native Coordinates
- `layer_id`, `layer_name`: Layer identification

**Coordinate System:**
- Input: Native Coordinates
- Output: World Coordinates (mm, Y-Up) via intermediate Normalized space

---

### Phase 4: Assemble

**Class:** [`ItemAssembler`](../../rayforge/image/assembler.py:15)

Instantiates Rayforge domain objects ([`WorkPiece`](../../rayforge/core/workpiece/__init__.py),
[`Layer`](../../rayforge/core/layer/__init__.py)) based on the layout plan.

**Output:** [`ImportPayload`](../../rayforge/image/structures.py:265)

- `source`: The [`SourceAsset`](../../rayforge/core/source_asset/__init__.py)
- `items`: List of [`DocItem`](../../rayforge/core/item/__init__.py) ready for insertion
- `sketches`: Optional list of [`Sketch`](../../rayforge/core/sketcher/sketch/__init__.py) objects

**Coordinate System:** All DocItems in World Coordinates (mm, Y-Up)

---

## Scan Method

**Method:** [`Importer.scan()`](../../rayforge/image/base_importer.py:161)

A lightweight scan that extracts metadata without full processing. Used for
building the UI for an importer, including layer selection list.
This is NOT part of the main import pipeline executed by `get_doc_items()`.

**Output:** [`ImportManifest`](../../rayforge/image/structures.py:36)

- `layers`: List of [`LayerInfo`](../../rayforge/image/structures.py:13) objects
- `natural_size_mm`: Physical dimensions in millimeters (Y-Up)
- `title`: Optional document title
- `warnings`, `errors`: Non-critical issues discovered

**Coordinate System:** World Coordinates (mm, Y-Up) for `natural_size_mm`

---

## Coordinate Systems

The import pipeline handles multiple coordinate systems through careful
transformation:

### Native Coordinates (Input)

- File-specific coordinate system (SVG user units, DXF units, pixels)
- Y-axis orientation varies by format
- Bounds are absolute within the document's coordinate space
- Units converted to mm via `native_unit_to_mm` factor

### Normalized Coordinates (Intermediate)

- Unit square from (0,0) to (1,1)
- Y-axis points UP (Y-Up convention)
- Used as intermediate representation between native and world

### World Coordinates (Output)

- Physical world coordinates in millimeters (mm)
- Y-axis points UP (Y-Up convention)
- Origin (0,0) is at the bottom-left of the workpiece
- All positions are absolute in the world coordinate system

### Y-Axis Orientation

- **Y-Down formats** (SVG, images): Origin at top-left, Y increases downward
- **Y-Up formats** (DXF): Origin at bottom-left, Y increases upward
- Importers must set `is_y_down` flag correctly in `ParsingResult`
- `NormalizationEngine` handles Y-inversion for Y-Down sources

---

## Key Classes

### Importer (Base Class)

[`rayforge/image/base_importer.py`](../../rayforge/image/base_importer.py:34)

Abstract base class defining the interface for all importers. Subclasses must
implement the pipeline methods and declare their capabilities via the
`features` attribute.

**Features:**
- `BITMAP_TRACING`: Can trace raster images to vectors
- `DIRECT_VECTOR`: Can extract vector geometry directly
- `LAYER_SELECTION`: Supports layer-based imports
- `PROCEDURAL_GENERATION`: Generates content programmatically

### Data Structures

All data structures are defined in
[`rayforge/image/structures.py`](../../rayforge/image/structures.py):

| Class | Phase | Purpose |
|-------|-------|---------|
| [`LayerInfo`](../../rayforge/image/structures.py:13) | Scan | Lightweight layer metadata |
| [`ImportManifest`](../../rayforge/image/structures.py:36) | Scan | Scan phase result |
| [`LayerGeometry`](../../rayforge/image/structures.py:71) | Parse | Geometric layer info |
| [`ParsingResult`](../../rayforge/image/structures.py:105) | Parse | Geometric facts |
| [`VectorizationResult`](../../rayforge/image/structures.py:176) | Vectorize | Vector geometry |
| [`LayoutItem`](../../rayforge/image/structures.py:217) | Layout | Transformation config |
| [`ImportPayload`](../../rayforge/image/structures.py:265) | Assemble | Final output |
| [`ImportResult`](../../rayforge/image/structures.py:294) | Final | Complete result wrapper |

### Supporting Components

- [`NormalizationEngine`](../../rayforge/image/engine.py:19): Phase 3 layout
  calculations
- [`ItemAssembler`](../../rayforge/image/assembler.py:15): Phase 4 object
  creation

---

## Creating a New Importer

To add support for a new file format:

1. **Create a new importer class** that inherits from `Importer`
2. **Declare supported features** via the `features` class attribute
3. **Implement the required methods**:
   - `scan()`: Extract metadata quickly (for UI previews)
   - `parse()`: Extract geometric facts
   - `vectorize()`: Convert to vector geometry
   - `create_source_asset()`: Create the source asset
4. **Register the importer** in `rayforge/image/__init__.py`
5. **Add MIME type and extension mappings**

**Example:**

```python
from rayforge.image.base_importer import Importer, ImporterFeature
from rayforge.image.structures import (
    ImportManifest,
    ParsingResult,
    VectorizationResult,
)
from rayforge.core.source_asset import SourceAsset

class MyFormatImporter(Importer):
    label = "My Format"
    mime_types = ("application/x-myformat",)
    extensions = (".myf",)
    features = {ImporterFeature.DIRECT_VECTOR}

    def scan(self) -> ImportManifest:
        # Extract metadata without full processing
        return ImportManifest(
            layers=[],
            natural_size_mm=(100.0, 100.0),
        )

    def parse(self) -> Optional[ParsingResult]:
        # Extract geometric facts
        return ParsingResult(
            document_bounds=(0, 0, 100, 100),
            native_unit_to_mm=1.0,
            is_y_down=False,
            layers=[],
            world_frame_of_reference=(0, 0, 100, 100),
            background_world_transform=Matrix.identity(),
        )

    def vectorize(
        self, parse_result: ParsingResult, spec: VectorizationSpec
    ) -> VectorizationResult:
        # Convert to vector geometry
        return VectorizationResult(
            geometries_by_layer={None: Geometry()},
            source_parse_result=parse_result,
        )

    def create_source_asset(
        self, parse_result: ParsingResult
    ) -> SourceAsset:
        # Create the source asset
        return SourceAsset(
            original_data=self.raw_data,
            metadata={},
        )
```

**See also:** [`rayforge/image/dxf/importer.py`](../../rayforge/image/dxf/importer.py)
for a complete example of a vector importer, or
[`rayforge/image/jpg/importer.py`](../../rayforge/image/jpg/importer.py)
for a raster importer with bitmap tracing.
