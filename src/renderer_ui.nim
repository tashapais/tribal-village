## renderer_ui.nim - UI overlay coordinator
##
## Thin coordinator that imports and re-exports focused UI sub-modules:
##   - renderer_building_ui: Building construction, overlays, placement ghost
##   - renderer_controls: Footer buttons, speed controls, HUD labels
##   - renderer_panels: Resource bar, unit info panel, minimap
##   - renderer_selection: Selection glow, rally points, trade routes

import renderer_building_ui
export renderer_building_ui

import renderer_controls
export renderer_controls

import renderer_panels
export renderer_panels

import renderer_selection
export renderer_selection
