from ..shared.ops_renderer import OPS_RENDERER

# The DXF importer produces vector operations (Ops), so it uses the shared
# OPS_RENDERER. We create this alias for consistency.
DXF_RENDERER = OPS_RENDERER
