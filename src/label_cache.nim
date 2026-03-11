## label_cache.nim - Unified text label rendering and caching
##
## Provides a single API for rendering text to images and caching them in the
## boxy atlas. Replaces the scattered per-module label caches (overlayLabelImages,
## footerLabelImages, tooltipLabelImages, infoLabelImages, resourceBarLabelImages,
## damageNumberImages) with one consistent system.

import
  boxy, pixie, vmath, tables,
  common, environment

# ─── Types ──────────────────────────────────────────────────────────────────

type
  LabelStyle* = object
    fontPath*: string
    fontSize*: float32
    padding*: float32
    bgAlpha*: float32           ## Background alpha (0 = no background)
    textColor*: Color           ## Text color (default: white)
    outline*: bool              ## Draw dark outline behind text for visibility
    outlineColor*: Color        ## Outline color (default: semi-transparent black)

  CachedLabel* = object
    imageKey*: string
    size*: IVec2

# ─── Cache Storage ──────────────────────────────────────────────────────────

var labelCache: Table[string, CachedLabel] = initTable[string, CachedLabel]()

# ─── Style Constructors ────────────────────────────────────────────────────

proc labelStyle*(fontPath: string, fontSize: float32, padding: float32,
                 bgAlpha: float32): LabelStyle =
  ## Standard label style with white text and optional background.
  LabelStyle(
    fontPath: fontPath,
    fontSize: fontSize,
    padding: padding,
    bgAlpha: bgAlpha,
    textColor: TintWhite,
    outline: false,
    outlineColor: TextOutlineColor,
  )

proc labelStyleColored*(fontPath: string, fontSize: float32, padding: float32,
                        textColor: Color): LabelStyle =
  ## Colored label style with no background (for tooltips, etc).
  LabelStyle(
    fontPath: fontPath,
    fontSize: fontSize,
    padding: padding,
    bgAlpha: 0.0,
    textColor: textColor,
    outline: false,
    outlineColor: TextOutlineColor,
  )

proc labelStyleOutlined*(fontPath: string, fontSize: float32, padding: float32,
                         textColor: Color): LabelStyle =
  ## Outlined label style with no background (for damage numbers, etc).
  LabelStyle(
    fontPath: fontPath,
    fontSize: fontSize,
    padding: padding,
    bgAlpha: 0.0,
    textColor: textColor,
    outline: true,
    outlineColor: TextOutlineColor,
  )

# ─── Cache Key Generation ──────────────────────────────────────────────────

proc makeCacheKey*(prefix: string, text: string, style: LabelStyle): string =
  ## Build a unique cache key from prefix, text, and style parameters.
  ## The key encodes style parameters so the same text with different styles
  ## gets separate cache entries.
  result = prefix & "/" & text
  # Include color in key when non-white (colored/outlined styles)
  if style.textColor != TintWhite:
    result &= "_c" & $int(style.textColor.r * 255) &
              "_" & $int(style.textColor.g * 255) &
              "_" & $int(style.textColor.b * 255)
  # Include font size when it varies (tooltip title vs text)
  if style.fontSize != 0:
    result &= "_s" & $int(style.fontSize)

# ─── Core Rendering ────────────────────────────────────────────────────────

proc renderLabel(text: string, style: LabelStyle): (Image, IVec2) =
  ## Render text to an image using the given style. Internal proc.
  var measureCtx = newContext(1, 1)
  measureCtx.font = style.fontPath
  measureCtx.fontSize = style.fontSize
  measureCtx.textBaseline = TopBaseline
  let w = max(1, (measureCtx.measureText(text).width + style.padding * 2).int)
  let h = max(1, (style.fontSize + style.padding * 2).int)
  var ctx = newContext(w, h)
  ctx.font = style.fontPath
  ctx.fontSize = style.fontSize
  ctx.textBaseline = TopBaseline
  # Background fill
  if style.bgAlpha > 0:
    ctx.fillStyle.color = withAlpha(LabelBgBlack, style.bgAlpha)
    ctx.fillRect(0, 0, w.float32, h.float32)
  # Outline (draw text in outline color at 8 offsets)
  if style.outline:
    ctx.fillStyle.color = style.outlineColor
    for dx in -1 .. 1:
      for dy in -1 .. 1:
        if dx != 0 or dy != 0:
          ctx.fillText(text, vec2(style.padding + dx.float32,
                                  style.padding + dy.float32))
  # Main text
  ctx.fillStyle.color = style.textColor
  ctx.fillText(text, vec2(style.padding, style.padding))
  result = (ctx.image, ivec2(w.int32, h.int32))

# ─── Public API ─────────────────────────────────────────────────────────────

proc ensureLabel*(prefix: string, text: string, style: LabelStyle): CachedLabel =
  ## Get or create a cached text label image. Returns the boxy image key and size.
  ##
  ## Parameters:
  ##   prefix: Cache namespace (e.g., "overlay", "footer_btn", "tooltip", "dmgnum")
  ##   text: The text to render
  ##   style: Font, size, color, and rendering options
  let cacheKey = makeCacheKey(prefix, text, style)
  if cacheKey in labelCache:
    return labelCache[cacheKey]
  let (image, size) = renderLabel(text, style)
  bxy.addImage(cacheKey, image)
  let cached = CachedLabel(imageKey: cacheKey, size: size)
  labelCache[cacheKey] = cached
  result = cached

proc ensureLabelKeyed*(cacheKey: string, imageKey: string, text: string,
                       style: LabelStyle): CachedLabel =
  ## Like ensureLabel but with explicit cache key and image key.
  ## Used for cases where the cache key doesn't follow the standard prefix/text pattern
  ## (e.g., HUD labels that use a fixed key like "hud_step").
  if cacheKey in labelCache:
    return labelCache[cacheKey]
  let (image, size) = renderLabel(text, style)
  bxy.addImage(imageKey, image)
  let cached = CachedLabel(imageKey: imageKey, size: size)
  labelCache[cacheKey] = cached
  result = cached

proc invalidateLabel*(cacheKey: string) =
  ## Remove a specific label from the cache (for labels that change, like step counter).
  labelCache.del(cacheKey)
