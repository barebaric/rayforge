from ..shared.ops_renderer import OPS_RENDERER

# The RUIDA importer produces vector operations (Ops), so it uses the shared
# OPS_RENDERER. We create this alias for consistency.
RUIDA_RENDERER = OPS_RENDERER
