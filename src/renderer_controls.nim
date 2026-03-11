## renderer_controls.nim - Footer buttons, speed controls, and HUD labels
##
## Contains: footer button types and rendering, speed control buttons,
## selection labels, step labels, control mode labels.

import
  boxy, bumpy, pixie, vmath, tables, std/math,
  common, environment, semantic

import renderer_core, label_cache

# ─── Footer Button Types and Rendering ───────────────────────────────────────

type FooterButtonKind* = enum
  FooterPlayPause
  FooterStep
  FooterSlow
  FooterFast
  FooterFaster
  FooterSuper

type FooterButton* = object
  kind*: FooterButtonKind
  rect*: IRect
  labelKey*: string
  labelSize*: IVec2
  iconKey*: string
  iconSize*: IVec2
  isPressed*: bool

var
  footerIconSizes: Table[string, IVec2] = initTable[string, IVec2]()

proc buildFooterButtons*(panelRect: IRect): seq[FooterButton] =
  let footerTop = panelRect.y.int32 + panelRect.h.int32 - FooterHeight.int32
  let innerHeight = FooterHeight.float32 - FooterPadding * 2.0
  var buttons: seq[FooterButton] = @[]

  # Calculate button widths based on content
  let buttonDefs = [
    # Icon-first, but always provide a text fallback so missing atlas sprites don't crash UI.
    (FooterPlayPause, if paused: "icon_play" else: "icon_pause", if paused: "Play" else: "Pause"),
    (FooterStep, "icon_step", "Step"),
    (FooterSlow, "", "0.5x"),
    (FooterFast, "", "2x"),
    (FooterFaster, "icon_ffwd", "4x"),
  ]

  var totalWidth = 0.0'f32
  var buttonWidths: seq[float32] = @[]
  for (_, iconKey, labelText) in buttonDefs:
    var w = FooterButtonPaddingX * 2.0
    if iconKey.len > 0 and iconKey in bxy:
      if iconKey notin footerIconSizes:
        let size = bxy.getImageSize(iconKey)
        footerIconSizes[iconKey] = ivec2(size.x.int32, size.y.int32)
      let iconSize = footerIconSizes[iconKey]
      let sc = min(1.0'f32, innerHeight / iconSize.y.float32)
      w += iconSize.x.float32 * sc
    elif labelText.len > 0:
      let cached = ensureLabel("footer_btn", labelText, footerBtnLabelStyle)
      let labelSize = cached.size
      let sc = min(1.0'f32, innerHeight / labelSize.y.float32)
      w += labelSize.x.float32 * sc
    buttonWidths.add(w)
    totalWidth += w

  totalWidth += FooterButtonGap * (buttonWidths.len - 1).float32

  # Center buttons in footer
  var x = panelRect.x.float32 + (panelRect.w.float32 - totalWidth) / 2.0
  let y = footerTop.float32 + FooterPadding

  for i, (kind, iconKey, labelText) in buttonDefs:
    let w = buttonWidths[i]
    var btn = FooterButton(
      kind: kind,
      rect: IRect(x: x.int32, y: y.int32, w: w.int32, h: innerHeight.int32),
      isPressed: false
    )
    if iconKey.len > 0 and iconKey in bxy:
      btn.iconKey = iconKey
      btn.iconSize = footerIconSizes[iconKey]
    if labelText.len > 0:
      let cached = ensureLabel("footer_btn", labelText, footerBtnLabelStyle)
      btn.labelKey = cached.imageKey
      btn.labelSize = cached.size
    # Check pressed state
    btn.isPressed = case kind
      of FooterPlayPause: not paused
      of FooterStep: false
      of FooterSlow: speedMultiplier == SpeedSlow
      of FooterFast: speedMultiplier == SpeedFast
      of FooterFaster: speedMultiplier >= SpeedFaster and speedMultiplier < SpeedSuperMin
      of FooterSuper: speedMultiplier >= SpeedSuperMin
    buttons.add(btn)
    x += w + FooterButtonGap

  result = buttons

proc centerIn(rect: IRect, size: Vec2): Vec2 =
  vec2(
    rect.x.float32 + (rect.w.float32 - size.x) / 2.0,
    rect.y.float32 + (rect.h.float32 - size.y) / 2.0
  )

proc drawFooter*(panelRect: IRect, buttons: seq[FooterButton]) =
  pushSemanticContext("Footer")
  let footerTop = panelRect.y.float32 + panelRect.h.float32 - FooterHeight.float32

  # Draw footer background (Mettascope-style ribbon)
  let bgColor = UiBgHeader
  bxy.drawRect(rect = Rect(x: panelRect.x.float32, y: footerTop,
                          w: panelRect.w.float32, h: FooterHeight.float32),
               color = bgColor)
  bxy.drawRect(rect = Rect(x: panelRect.x.float32, y: footerTop,
                           w: panelRect.w.float32, h: FooterBorderHeight),
               color = UiBorderBright)

  # Draw buttons
  let innerHeight = FooterHeight.float32 - FooterPadding * 2.0
  for button in buttons:
    # Button background
    let btnBg = if button.isPressed:
      UiBgButtonActive
    else:
      UiBgButton
    bxy.drawRect(rect = Rect(x: button.rect.x.float32, y: button.rect.y.float32,
                            w: button.rect.w.float32, h: button.rect.h.float32),
                 color = btnBg)
    bxy.drawRect(rect = Rect(x: button.rect.x.float32, y: button.rect.y.float32,
                             w: button.rect.w.float32, h: FooterBorderHeight),
                 color = UiBorder)
    bxy.drawRect(rect = Rect(x: button.rect.x.float32, y: button.rect.y.float32 + button.rect.h.float32 - FooterBorderHeight,
                             w: button.rect.w.float32, h: FooterBorderHeight),
                 color = UiBorder)

    # Draw icon or label
    if button.iconKey.len > 0 and button.iconKey in bxy:
      let sc = min(1.0'f32, innerHeight / button.iconSize.y.float32)
      let iconSize = vec2(button.iconSize.x.float32 * sc, button.iconSize.y.float32 * sc)
      let iconPos = centerIn(button.rect, iconSize) + vec2(FooterIconCenterShiftX, FooterIconCenterShiftY) * sc
      drawUiImageScaled(button.iconKey, iconPos, iconSize)
    elif button.labelKey.len > 0 and button.labelKey in bxy:
      # Text labels use boxy
      let shift = if button.kind == FooterFaster: vec2(FooterIconCenterShiftX, FooterIconCenterShiftY) else: vec2(0.0, 0.0)
      let labelSize = vec2(button.labelSize.x.float32, button.labelSize.y.float32)
      let labelPos = centerIn(button.rect, labelSize) + shift
      drawUiImageScaled(button.labelKey, labelPos, labelSize)

  popSemanticContext()

# ─── HUD Labels ──────────────────────────────────────────────────────────────

var
  stepLabelKey = ""
  stepLabelLastValue = -1
  stepLabelSize = ivec2(0, 0)
  controlModeLabelKey = ""
  controlModeLabelLastValue = -2  # Start different from any valid value
  controlModeLabelSize = ivec2(0, 0)

proc drawFooterHudLabel(panelRect: IRect, key: string, labelSize: IVec2, xOffset: float32) =
  let footerTop = panelRect.y.float32 + panelRect.h.float32 - FooterHeight.float32
  let innerHeight = FooterHeight.float32 - FooterPadding * 2.0
  let scale = min(1.0'f32, innerHeight / labelSize.y.float32)
  let pos = vec2(
    panelRect.x.float32 + xOffset,
    footerTop + FooterPadding + (innerHeight - labelSize.y.float32 * scale) * 0.5 + FooterHudLabelYShift
  )
  let size = vec2(labelSize.x.float32 * scale, labelSize.y.float32 * scale)
  drawUiImageScaled(key, pos, size)

proc drawSelectionLabel*(panelRect: IRect) =
  if not isValidPos(selectedPos):
    return

  proc appendResourceCount(label: var string, thing: Thing) =
    var count = 0
    case thing.kind
    of Wheat, Stubble:
      count = getInv(thing, ItemWheat)
    of Fish:
      count = getInv(thing, ItemFish)
    of Relic:
      count = getInv(thing, ItemGold)
    of Tree, Stump:
      count = getInv(thing, ItemWood)
    of Stone, Stalagmite:
      count = getInv(thing, ItemStone)
    of Gold:
      count = getInv(thing, ItemGold)
    of Bush, Cactus:
      count = getInv(thing, ItemPlant)
    of Cow:
      count = getInv(thing, ItemMeat)
    of Corpse:
      for key, c in thing.inventory.pairs:
        if c > 0:
          label &= " (" & $key & " " & $c & ")"
          return
      return
    else:
      return
    label &= " (" & $count & ")"

  template displayNameFor(t: Thing): string =
    if t.kind == Agent:
      UnitClassLabels[t.unitClass]
    elif isBuildingKind(t.kind):
      BuildingRegistry[t.kind].displayName
    else:
      let name = ThingCatalog[t.kind].displayName
      if name.len > 0: name else: $t.kind

  var label = ""
  let selThing = env.grid[selectedPos.x][selectedPos.y]
  if not isNil(selThing):
    label = displayNameFor(selThing)
    appendResourceCount(label, selThing)
  else:
    let bgThing = env.backgroundGrid[selectedPos.x][selectedPos.y]
    if not isNil(bgThing):
      label = displayNameFor(bgThing)
      appendResourceCount(label, bgThing)

  if label.len == 0:
    return
  let cached = ensureLabel("info_label", label, infoLabelStyle)
  drawFooterHudLabel(panelRect, cached.imageKey, cached.size, FooterHudPadding)

proc drawStepLabel*(panelRect: IRect) =
  let currentStep = env.currentStep
  if currentStep != stepLabelLastValue or stepLabelKey.len == 0:
    stepLabelLastValue = currentStep
    invalidateLabel("hud_step")
    let text = "Step: " & $currentStep
    let cached = ensureLabelKeyed("hud_step", "hud_step", text, infoLabelStyle)
    stepLabelKey = cached.imageKey
    stepLabelSize = cached.size
  if stepLabelKey.len == 0:
    return
  let innerHeight = FooterHeight.float32 - FooterPadding * 2.0
  let scale = min(1.0'f32, innerHeight / stepLabelSize.y.float32)
  let labelW = stepLabelSize.x.float32 * scale
  let xOffset = panelRect.w.float32 - labelW - FooterHudPadding
  drawFooterHudLabel(panelRect, stepLabelKey, stepLabelSize, xOffset)

proc drawControlModeLabel*(panelRect: IRect) =
  let mode = playerTeam
  if mode != controlModeLabelLastValue or controlModeLabelKey.len == 0:
    controlModeLabelLastValue = mode
    invalidateLabel("hud_control_mode")
    let text = case mode
      of -1: "Observer"
      of 0..7: "Team " & $mode
      else: "Unknown"
    let cached = ensureLabelKeyed("hud_control_mode", "hud_control_mode", text, infoLabelStyle)
    controlModeLabelKey = cached.imageKey
    controlModeLabelSize = cached.size
  if controlModeLabelKey.len == 0:
    return
  let innerHeight = FooterHeight.float32 - FooterPadding * 2.0
  let scale = min(1.0'f32, innerHeight / controlModeLabelSize.y.float32)
  # Position to the left of the step label (center area of footer)
  let labelW = controlModeLabelSize.x.float32 * scale
  let stepLabelW = stepLabelSize.x.float32 * scale
  let xOffset = panelRect.w.float32 - labelW - stepLabelW - FooterHudPadding * 2.0 - ResourceBarXStart
  drawFooterHudLabel(panelRect, controlModeLabelKey, controlModeLabelSize, xOffset)
