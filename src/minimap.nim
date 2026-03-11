## Minimap rendering: bird's-eye terrain view with units, buildings, and viewport rect.
##
## All rendering uses boxy (dynamic texture for map content, drawRect for UI elements).

import
  boxy, pixie, vmath, windy, chroma,
  common, constants, environment, semantic

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

var
  minimapImageKey = "minimap_image"
  minimapCacheGeneration = -1
  minimapCacheStep = -1
  minimapFullRebuildCounter = 0
  minimapBaseImage: Image  # Cached terrain layer (static between map regens)

# ---------------------------------------------------------------------------
# Minimap Terrain Colors (ColorRGBX for direct pixel writes)
# ---------------------------------------------------------------------------

const
  MinimapWater        = rgbx(25, 50, 120, 255)   ## Deep water
  MinimapShallowWater = rgbx(60, 110, 160, 255)  ## Shallow water
  MinimapBridge       = rgbx(140, 110, 70, 255)  ## Bridge
  MinimapFertile      = rgbx(50, 100, 30, 255)   ## Fertile land
  MinimapRoad         = rgbx(140, 130, 110, 255) ## Road
  MinimapGrass        = rgbx(60, 120, 40, 255)   ## Grass
  MinimapDune         = rgbx(190, 170, 100, 255) ## Dune
  MinimapSand         = rgbx(180, 160, 90, 255)  ## Sand
  MinimapSnow         = rgbx(220, 230, 240, 255) ## Snow
  MinimapMud          = rgbx(90, 70, 50, 255)    ## Mud
  MinimapMountain     = rgbx(80, 75, 70, 255)    ## Mountain
  MinimapRamp         = rgbx(130, 120, 100, 255) ## Ramp
  MinimapEmpty        = rgbx(50, 80, 40, 255)    ## Empty/default

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

proc terrainColor(t: TerrainType): ColorRGBX =
  case t
  of Water:        MinimapWater
  of ShallowWater: MinimapShallowWater
  of Bridge:       MinimapBridge
  of Fertile:      MinimapFertile
  of Road:         MinimapRoad
  of Grass:        MinimapGrass
  of Dune:         MinimapDune
  of Sand:         MinimapSand
  of Snow:         MinimapSnow
  of Mud:          MinimapMud
  of Mountain:     MinimapMountain
  of RampUpN, RampUpS, RampUpW, RampUpE,
     RampDownN, RampDownS, RampDownW, RampDownE:
                   MinimapRamp
  of Empty:        MinimapEmpty

proc rebuildMinimapBase() =
  let mmW = MinimapSize
  let mmH = MinimapSize
  minimapBaseImage = newImage(mmW, mmH)
  let scaleX = MapWidth.float32 / mmW.float32
  let scaleY = MapHeight.float32 / mmH.float32
  for py in 0 ..< mmH:
    for px in 0 ..< mmW:
      let tx = clamp(int(px.float32 * scaleX), 0, MapWidth - 1)
      let ty = clamp(int(py.float32 * scaleY), 0, MapHeight - 1)
      minimapBaseImage.unsafe[px, py] = terrainColor(env.terrain[tx][ty])
  minimapCacheGeneration = env.mapGeneration

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

proc minimapRect*(panelRect: IRect): Rect =
  ## Screen rectangle for the minimap (bottom-left, above footer).
  ## Uses the layout system if available, falls back to calculated position.
  if uiLayout.minimapArea != nil and uiLayout.minimapArea.rect.w > 0:
    return uiLayout.minimapArea.rect

  let x = panelRect.x.float32 + MinimapMargin.float32
  let y = panelRect.y.float32 + panelRect.h.float32 -
          FooterHeight.float32 - MinimapMargin.float32 - MinimapSize.float32
  Rect(x: x, y: y, w: MinimapSize.float32, h: MinimapSize.float32)

proc isInMinimap*(panelRect: IRect, mousePosPx: Vec2): bool =
  ## Check if a pixel-space mouse position is inside the minimap.
  let mmRect = minimapRect(panelRect)
  mousePosPx.x >= mmRect.x and mousePosPx.x <= mmRect.x + mmRect.w and
    mousePosPx.y >= mmRect.y and mousePosPx.y <= mmRect.y + mmRect.h

proc minimapToWorld*(panelRect: IRect, mousePosPx: Vec2): Vec2 =
  ## Convert a pixel-space mouse position on the minimap to world coordinates.
  let mmRect = minimapRect(panelRect)
  let relX = (mousePosPx.x - mmRect.x) / mmRect.w
  let relY = (mousePosPx.y - mmRect.y) / mmRect.h
  vec2(relX * MapWidth.float32, relY * MapHeight.float32)

proc drawMinimap*(panelRect: IRect, cameraPos: Vec2, zoom: float32) =
  ## Draw the minimap overlay with units, buildings, and viewport rectangle.
  let mmW = MinimapSize
  let mmH = MinimapSize

  # Rebuild base terrain cache when map generation changes
  if minimapCacheGeneration != env.mapGeneration:
    rebuildMinimapBase()

  # Rebuild composite image when game state changes
  let needsRebuild = minimapCacheStep != env.currentStep or
                     (minimapFullRebuildCounter mod MinimapRebuildInterval == 0)
  inc minimapFullRebuildCounter

  if needsRebuild:
    minimapCacheStep = env.currentStep
    var img = minimapBaseImage.copy()
    let scaleX = MapWidth.float32 / mmW.float32
    let scaleY = MapHeight.float32 / mmH.float32

    # Buildings in team colors
    for kind in ThingKind:
      if not isBuildingKind(kind):
        continue
      for thing in env.thingsByKind[kind]:
        if not isValidPos(thing.pos):
          continue
        let px = clamp(int(thing.pos.x.float32 / scaleX), 0, mmW - 1)
        let py = clamp(int(thing.pos.y.float32 / scaleY), 0, mmH - 1)
        let teamId = thing.teamId
        let c = if teamId >= 0 and teamId < env.teamColors.len:
          env.teamColors[teamId]
        else:
          NeutralGrayMinimap
        img.unsafe[px, py] = rgbx(
          uint8(clamp(c.r * 255, 0, 255)),
          uint8(clamp(c.g * 255, 0, 255)),
          uint8(clamp(c.b * 255, 0, 255)), 255)

    # Agents as bright dots
    for agent in env.agents:
      if not isAgentAlive(env, agent):
        continue
      let apos = agent.pos
      if not isValidPos(apos):
        continue
      let px = clamp(int(apos.x.float32 / scaleX), 0, mmW - 1)
      let py = clamp(int(apos.y.float32 / scaleY), 0, mmH - 1)
      let c = env.agentColors[agent.agentId]
      img.unsafe[px, py] = rgbx(
        uint8(clamp(c.r * MinimapBrightnessMult.float32, MinimapBrightnessMin.float32, 255)),
        uint8(clamp(c.g * MinimapBrightnessMult.float32, MinimapBrightnessMin.float32, 255)),
        uint8(clamp(c.b * MinimapBrightnessMult.float32, MinimapBrightnessMin.float32, 255)), 255)

    # Walls as gray
    for wall in env.thingsByKind[Wall]:
      if not isValidPos(wall.pos):
        continue
      let px = clamp(int(wall.pos.x.float32 / scaleX), 0, mmW - 1)
      let py = clamp(int(wall.pos.y.float32 / scaleY), 0, mmH - 1)
      img.unsafe[px, py] = rgbx(MinimapWallGray.uint8, MinimapWallGray.uint8, MinimapWallGray.uint8, 255)

    bxy.addImage(minimapImageKey, img)

  # Draw minimap border and map image
  let mmRect = minimapRect(panelRect)
  let borderPos = vec2(mmRect.x - MinimapBorderWidth, mmRect.y - MinimapBorderWidth)
  let borderSize = vec2(mmRect.w + MinimapBorderExpand, mmRect.h + MinimapBorderExpand)

  bxy.drawRect(rect = Rect(x: borderPos.x, y: borderPos.y, w: borderSize.x, h: borderSize.y),
               color = UiMinimapBorderDark)

  # Map content rendered with boxy (dynamic texture)
  bxy.drawImage(minimapImageKey, vec2(mmRect.x, mmRect.y))

  # Semantic capture: minimap panel
  pushSemanticContext("Minimap")
  capturePanel("Minimap", vec2(mmRect.x, mmRect.y), vec2(mmRect.w, mmRect.h))

  # Viewport rectangle ---------------------------------------------------
  let scaleF = window.contentScale.float32
  let rectW = panelRect.w.float32 / scaleF
  let rectH = panelRect.h.float32 / scaleF
  let zoomScale = zoom * zoom

  # Camera center in world coords
  let cx = (rectW / 2.0'f32 - cameraPos.x) / zoomScale
  let cy = (rectH / 2.0'f32 - cameraPos.y) / zoomScale

  # Viewport half-size in world coords
  let viewHalfW = rectW / (2.0'f32 * zoomScale)
  let viewHalfH = (rectH - FooterHeight.float32 / scaleF) / (2.0'f32 * zoomScale)

  # World -> minimap pixel coords
  let mmScaleX = mmRect.w / MapWidth.float32
  let mmScaleY = mmRect.h / MapHeight.float32

  let vpLeft = mmRect.x + (cx - viewHalfW) * mmScaleX
  let vpTop = mmRect.y + (cy - viewHalfH) * mmScaleY
  let vpW = viewHalfW * 2.0'f32 * mmScaleX
  let vpH = viewHalfH * 2.0'f32 * mmScaleY

  # Clamp to minimap bounds
  let clLeft = max(vpLeft, mmRect.x)
  let clTop = max(vpTop, mmRect.y)
  let clRight = min(vpLeft + vpW, mmRect.x + mmRect.w)
  let clBottom = min(vpTop + vpH, mmRect.y + mmRect.h)

  if clRight > clLeft and clBottom > clTop:
    let lineW = MinimapViewportLineWidth

    # Draw viewport indicator lines
    let vpColor = withAlpha(UiViewportOutline, MinimapViewportAlpha)
    bxy.drawRect(rect = Rect(x: clLeft, y: clTop,
                 w: clRight - clLeft, h: lineW), color = vpColor)
    bxy.drawRect(rect = Rect(x: clLeft, y: clBottom - lineW,
                 w: clRight - clLeft, h: lineW), color = vpColor)
    bxy.drawRect(rect = Rect(x: clLeft, y: clTop,
                 w: lineW, h: clBottom - clTop), color = vpColor)
    bxy.drawRect(rect = Rect(x: clRight - lineW, y: clTop,
                 w: lineW, h: clBottom - clTop), color = vpColor)

  popSemanticContext()
