import std/[json, os, tables]
import zippy
import types, items, registry, replay_common

const
  DefaultReplayBaseName = "tribal_village"

type
  ReplaySeries = object
    hasLast: bool
    last: JsonNode
    changes: seq[tuple[step: int, value: JsonNode]]

  ReplayObject = ref object
    id: int
    constFields: Table[string, JsonNode]
    series: Table[string, ReplaySeries]
    active: bool

  ReplayWriter* = ref object
    enabled*: bool
    baseDir: string
    basePath: string
    baseName: string
    label: string
    episodeIndex: int
    outputPath: string
    fileName: string
    nextId: int
    thingIds: Table[pointer, int]
    objects: seq[ReplayObject]
    totalRewards: array[MapAgents, float32]
    lastInvalidCounts: array[MapAgents, int]
    maxStep: int
    active: bool

var replayWriter*: ReplayWriter = nil

proc maybeStartReplayEpisode*(env: Environment) =
  var writer = replayWriter
  if writer.isNil:
    let basePath = getEnv("TV_REPLAY_PATH", "")
    let baseDir = getEnv("TV_REPLAY_DIR", "")
    if basePath.len == 0 and baseDir.len == 0:
      return
    let baseName = getEnv("TV_REPLAY_NAME", DefaultReplayBaseName)
    let label = getEnv("TV_REPLAY_LABEL", "Tribal Village Replay")
    writer = ReplayWriter(
      enabled: true,
      baseDir: baseDir,
      basePath: basePath,
      baseName: baseName,
      label: label
    )
    replayWriter = writer
  inc writer.episodeIndex
  writer.thingIds.clear()
  writer.objects.setLen(0)
  writer.nextId = 1
  writer.totalRewards = default(array[MapAgents, float32])
  writer.lastInvalidCounts = default(array[MapAgents, int])
  writer.maxStep = -1
  writer.active = true
  if writer.basePath.len > 0:
    writer.outputPath = writer.basePath
  else:
    let suffix = "_" & $writer.episodeIndex
    let fileName = writer.baseName & suffix & ".json.z"
    if writer.baseDir.len > 0:
      writer.outputPath = writer.baseDir / fileName
    else:
      writer.outputPath = fileName
  writer.fileName = extractFilename(writer.outputPath)

proc addSeries(obj: ReplayObject, key: string, step: int, value: JsonNode) =
  var series = obj.series.mgetOrPut(key, ReplaySeries())
  if not series.hasLast:
    series.changes.add((step: step, value: value))
    series.last = value
    series.hasLast = true
  elif series.last != value:
    series.changes.add((step: step, value: value))
    series.last = value
  obj.series[key] = series

proc inventoryNode(thing: Thing): JsonNode =
  result = newJArray()
  for kind in ItemKind:
    if kind == ikNone:
      continue
    let count = getInv(thing, kind)
    if count <= 0:
      continue
    var pair = newJArray()
    pair.add(newJInt(kind.int))
    pair.add(newJInt(count))
    result.add(pair)

proc locationNode(pos: IVec2): JsonNode =
  result = newJArray()
  result.add(newJInt(pos.x))
  result.add(newJInt(pos.y))

proc ensureReplayObject(writer: ReplayWriter, thing: Thing): ReplayObject =
  let key = cast[pointer](thing)
  var objectId = writer.thingIds.getOrDefault(key, 0)
  if objectId == 0:
    objectId = writer.nextId
    inc writer.nextId
    writer.thingIds[key] = objectId
  if writer.objects.len < objectId:
    writer.objects.setLen(objectId)
  if writer.objects[objectId - 1].isNil:
    let replayObj = ReplayObject(id: objectId)
    replayObj.constFields = initTable[string, JsonNode]()
    replayObj.series = initTable[string, ReplaySeries]()
    replayObj.constFields["id"] = newJInt(objectId)
    replayObj.constFields["type_name"] = newJString(buildingSpriteKey(thing.kind))
    if thing.kind == Agent:
      replayObj.constFields["agent_id"] = newJInt(thing.agentId)
      replayObj.constFields["group_id"] = newJInt(getTeamId(thing))
      replayObj.constFields["inventory_max"] = newJInt(MapObjectAgentMaxInventory)
    elif thing.barrelCapacity > 0:
      replayObj.constFields["inventory_max"] = newJInt(thing.barrelCapacity)
    else:
      replayObj.constFields["inventory_max"] = newJInt(0)
    writer.objects[objectId - 1] = replayObj
  writer.objects[objectId - 1]

proc maybeLogReplayStep*(env: Environment, actions: ptr array[MapAgents, uint16]) =
  let writer = replayWriter
  if writer.isNil or not writer.active:
    return
  let stepIndex = env.currentStep - 1
  if stepIndex < 0:
    return
  writer.maxStep = max(writer.maxStep, stepIndex)

  var seen: seq[bool] = @[]

  for thing in env.things:
    if thing.isNil:
      continue
    let replayObj = writer.ensureReplayObject(thing)
    let objectIdx = replayObj.id - 1
    if objectIdx >= seen.len:
      seen.setLen(objectIdx + 1)
    seen[objectIdx] = true
    replayObj.active = true

    replayObj.addSeries("location", stepIndex, locationNode(thing.pos))
    replayObj.addSeries("orientation", stepIndex, newJInt(thing.orientation.int))
    replayObj.addSeries("inventory", stepIndex, inventoryNode(thing))
    var color = 0
    if thing.kind == Agent:
      color = getTeamId(thing)
    elif isBuildingKind(thing.kind) or thing.kind == Lantern:
      color = max(0, thing.teamId)
    replayObj.addSeries("color", stepIndex, newJInt(color))

    if thing.kind == Agent:
      let agentId = thing.agentId
      let actionValue = actions[][agentId]
      let verb = actionValue.int div ActionArgumentCount
      let arg = actionValue.int mod ActionArgumentCount
      var actionSuccess = false
      if agentId >= 0 and agentId < env.stats.len:
        let invalidNow = env.stats[agentId].actionInvalid
        actionSuccess = invalidNow == writer.lastInvalidCounts[agentId]
        writer.lastInvalidCounts[agentId] = invalidNow
      replayObj.addSeries("action_id", stepIndex, newJInt(verb))
      replayObj.addSeries("action_param", stepIndex, newJInt(arg))
      replayObj.addSeries("action_success", stepIndex, newJBool(actionSuccess))
      replayObj.addSeries("current_reward", stepIndex, newJFloat(env.rewards[agentId].float))
      writer.totalRewards[agentId] += env.rewards[agentId]
      replayObj.addSeries("total_reward", stepIndex, newJFloat(writer.totalRewards[agentId].float))
      replayObj.addSeries("is_frozen", stepIndex, newJBool(thing.frozen > 0))

  for idx, obj in writer.objects:
    if obj.isNil:
      continue
    if idx >= seen.len or not seen[idx]:
      if obj.active:
        obj.active = false
        obj.addSeries("location", stepIndex, locationNode(ivec2(-1, -1)))

proc buildReplayJson(writer: ReplayWriter): JsonNode =
  result = newJObject()
  result["version"] = newJInt(ReplayVersion)
  var actionNames = newJArray()
  for name in ActionNames:
    actionNames.add(newJString(name))
  result["action_names"] = actionNames
  var itemNames = newJArray()
  for kind in ItemKind:
    itemNames.add(newJString(ItemKindNames[kind]))
  result["item_names"] = itemNames
  var typeNames = newJArray()
  for kind in ThingKind:
    typeNames.add(newJString(buildingSpriteKey(kind)))
  result["type_names"] = typeNames
  result["num_agents"] = newJInt(MapAgents)
  let maxSteps = if writer.maxStep >= 0: writer.maxStep + 1 else: 0
  result["max_steps"] = newJInt(maxSteps)
  var mapSize = newJArray()
  mapSize.add(newJInt(MapWidth))
  mapSize.add(newJInt(MapHeight))
  result["map_size"] = mapSize
  result["file_name"] = newJString(writer.fileName)
  var mgConfig = newJObject()
  mgConfig["label"] = newJString(writer.label)
  result["mg_config"] = mgConfig

  var objectsArr = newJArray()
  for obj in writer.objects:
    if obj.isNil:
      continue
    var objNode = newJObject()
    for key, value in obj.constFields:
      objNode[key] = value
    for key, series in obj.series:
      objNode[key] = serializeChanges(series.changes)
    objectsArr.add(objNode)
  result["objects"] = objectsArr

proc maybeFinalizeReplay*(env: Environment) =
  let writer = replayWriter
  if writer.isNil or not writer.active:
    return
  let replayJson = buildReplayJson(writer)
  let jsonData = $replayJson
  let compressed = zippy.compress(jsonData, dataFormat = dfZlib)
  if writer.outputPath.len > 0:
    let dir = parentDir(writer.outputPath)
    if dir.len > 0:
      createDir(dir)
    writeFile(writer.outputPath, compressed)
  writer.active = false
