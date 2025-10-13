# **Sequential Pipeline Architecture**

This document describes the pipeline in a linear, sequential flow, showing how
raw design data is progressively transformed into the final outputs used for
visualization and manufacturing.

```mermaid
graph TD
    subgraph "1. Input"
        Input("Input<br/>Document")
    end

    subgraph "2. Generator Pipeline"
        subgraph "2a. StepRunner"
            direction LR
            A(Single WorkPiece + Step Config);
            A --> B(Modifiers<br/><i>e.g., ToGrayscale</i>);
            B --> C(Producer<br/><i>e.g., EdgeTracer,<br/>Rasterizer</i>);
            C -- "Creates Artifact" --> D("Toolpaths<br/><i>Ops + Metadata</i>");
            D --> E(Transformers<br/><i>e.g., Tabs,<br/>Smooth</i>);
            E -- "Modifies Ops in-place" --> F(Vertex Encoder);
            F -- "Creates render data" --> G("Final Cached Artifact<br/><i>e.g. Ops, Vertices, Textures</i>");
        end

        subgraph "2b. JobRunner (Assembles the final job)"
            direction LR
            J("Multiple cached Artifacts<br/><i>(from StepRunner)</i>");
            J --> K(Assemble & Transform<br/><i>Applies position, scale, rotation</i>);
            K --> L(Per-Step Transformers<br/><i>e.g., Multi-Pass, Optimize</i>);
            L --> M(G-code Encoder);
            M --> N(<b>Final G-code</b>);
        end
    end

    subgraph "3. Consumers"
        Vis("Canvas Visualization (UI)")
        File("G-code File (for Machine)")
    end

    Input --> A;
    G -.-> J;
    G --> Vis;
    N --> File;

    classDef io fill:#f9f,stroke:#333,stroke-width:2px;
    classDef output fill:#bbf,stroke:#333,stroke-width:2px,max-width:500px;
    class Input,Vis,File io;
    class A,D,G,J,N output
```

# **Detailed Breakdown of the Sequence**

## **1. Input**

The process begins with the **Input**, which is the `Doc Model`. This is the
complete representation of the user's project, containing:

- **WorkPieces:** The individual design elements (like SVGs or images) that
  have been placed on the canvas.
- **Steps:** The specific instructions for how to process those WorkPieces
  (e.g., a "Contour" cut or a "Raster" engrave), including settings like
  power and speed.

## **2. The Generator**

The `Generator` is the core processing engine that runs in the background.
It takes the `Doc Model` as its input and orchestrates two distinct but
related processes: the `StepRunner` and the `JobRunner`.

### **2a. StepRunner: Per-Item Processing**

The `StepRunner` processes each `(WorkPiece, Step)` combination individually
to create a cached, reusable result. This allows for fast updates when only
a single item changes. Its internal sequence is:

1.  **Modifiers:** (Optional) If the input is a raster image, modifiers
    perform initial image conditioning, such as converting it to grayscale.
2.  **Producer:** This is the primary generation step. A `Producer` (like
    `EdgeTracer` or `Rasterizer`) analyzes the input and creates an
    `Artifact`. This initial artifact contains the raw toolpaths (`Ops`)
    and metadata describing the result (e.g., if it's scalable).
3.  **Per-Workpiece Transformers:** The newly generated `Ops` are passed
    through transformers that are specific to that individual workpiece,
    such as adding holding `Tabs` or `Smooth`ing the geometry.
4.  **Vertex Encoder:** Finally, the processed `Ops` are encoded into
    GPU-friendly formats for display. This generates `Vertex Data` (for
    lines and paths) and `Texture Data` (for raster fills).

The final output is a **Cached Artifact** stored in shared memory. This
artifact contains everything needed for both visualization and final
job assembly. The artifact system uses NumPy arrays for efficient
data transfer between processes and includes vertex data for GPU rendering
and command data for G-code generation.

### **2b. JobRunner: Final Assembly**

The `JobRunner` is invoked when the user wants to generate the final G-code
for the entire project. It consumes the artifacts created by the
`StepRunner`.

1.  **Assemble & Transform:** The `JobRunner` retrieves all the required
    `Artifacts` from the cache. It then applies the final world
    transformations to each one—placing them at their correct X/Y position,
    applying rotation, and scaling them to their final size. All these
    individual toolpaths are combined into a single, large `Ops` object.
2.  **Per-Step Transformers:** This unified `Ops` object is then processed
    by transformers that operate on the job as a whole. This is where final
    path `Optimize`ation (to reduce travel moves) and `Multi-Pass`
    operations are applied.
3.  **G-code Encoder:** The final, optimized `Ops` object is fed into the
    `G-code Encoder`, which translates the machine-agnostic commands
    into the specific G-code dialect required by the user's machine.
    The encoder handles machine-specific formatting, coordinate systems,
    and command syntax.

The final output is the complete **G-code** text.

## **3. Consumers**

The data generated by the pipeline is consumed by two primary clients:

1.  **Canvas Visualization (UI):** The UI directly and efficiently uses the
    `Vertex Data` and `Texture Data` from the **Cached Artifacts** to render
    the toolpaths on the screen. The rendering system uses OpenGL buffers
    populated directly from the NumPy arrays in the artifacts. Because it uses
    pre-computed data, the UI remains fast and responsive.
2.  **G-code File (for Machine):** The final **G-code** from the `JobRunner`
    is saved to a file, which can then be sent to the laser cutter or CNC
    machine for manufacturing. The G-code includes proper header/footer,
    coordinate setup, and machine-specific optimizations.
