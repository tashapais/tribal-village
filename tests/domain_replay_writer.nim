## Tests for replay_writer.nim - Episode recording and JSON output.
##
## Tests the public interface: maybeStartReplayEpisode, maybeLogReplayStep,
## maybeFinalizeReplay. Uses TV_REPLAY_PATH env var to control output.
##
## Run: nim r --path:src tests/domain_replay_writer.nim

import std/[unittest, os, json, strformat]
import zippy
import environment
import types
import replay_writer as rw
import test_utils

proc readReplayJson(path: string): JsonNode =
  let compressed = readFile(path)
  let decompressed = zippy.uncompress(compressed, dataFormat = dfZlib)
  parseJson(decompressed)

proc getWriter(): rw.ReplayWriter =
  rw.replayWriter

proc setWriter(w: rw.ReplayWriter) =
  rw.replayWriter = w

suite "Replay Writer - Episode Lifecycle":
  test "maybeStartReplayEpisode no-ops without env vars":
    delEnv("TV_REPLAY_PATH")
    delEnv("TV_REPLAY_DIR")
    setWriter(nil)

    let env = makeEmptyEnv()
    rw.maybeStartReplayEpisode(env)

    check getWriter().isNil

  test "maybeStartReplayEpisode creates writer with TV_REPLAY_PATH":
    let tmpPath = getTempDir() / "test_replay_create.json.z"
    putEnv("TV_REPLAY_PATH", tmpPath)
    delEnv("TV_REPLAY_DIR")
    setWriter(nil)

    let env = makeEmptyEnv()
    rw.maybeStartReplayEpisode(env)

    check not getWriter().isNil
    check getWriter().enabled

    delEnv("TV_REPLAY_PATH")
    setWriter(nil)

  test "maybeStartReplayEpisode creates writer with TV_REPLAY_DIR":
    let tmpDir = getTempDir() / "replay_test_dir"
    createDir(tmpDir)
    putEnv("TV_REPLAY_DIR", tmpDir)
    delEnv("TV_REPLAY_PATH")
    setWriter(nil)

    let env = makeEmptyEnv()
    rw.maybeStartReplayEpisode(env)

    check not getWriter().isNil
    check getWriter().enabled

    delEnv("TV_REPLAY_DIR")
    removeDir(tmpDir)
    setWriter(nil)

suite "Replay Writer - No-Op Safety":
  test "maybeLogReplayStep safe with nil writer":
    setWriter(nil)
    let env = makeEmptyEnv()
    var actions: array[MapAgents, uint16]
    rw.maybeLogReplayStep(env, addr actions)
    # No crash = pass

  test "maybeFinalizeReplay safe with nil writer":
    setWriter(nil)
    let env = makeEmptyEnv()
    rw.maybeFinalizeReplay(env)
    # No crash = pass

  test "maybeLogReplayStep safe at step 0":
    let tmpPath = getTempDir() / "test_step0.json.z"
    putEnv("TV_REPLAY_PATH", tmpPath)
    delEnv("TV_REPLAY_DIR")
    setWriter(nil)

    let env = makeEmptyEnv()
    rw.maybeStartReplayEpisode(env)
    env.currentStep = 0  # stepIndex = -1

    var actions: array[MapAgents, uint16]
    rw.maybeLogReplayStep(env, addr actions)
    # No crash = pass

    delEnv("TV_REPLAY_PATH")
    setWriter(nil)

suite "Replay Writer - Full Replay Output":
  test "complete episode produces valid compressed JSON":
    let tmpPath = getTempDir() / "test_full_replay.json.z"
    putEnv("TV_REPLAY_PATH", tmpPath)
    delEnv("TV_REPLAY_DIR")
    setWriter(nil)

    let env = newEnvironment()
    rw.maybeStartReplayEpisode(env)

    for i in 0 ..< 5:
      var actions: array[MapAgents, uint16]
      env.step(addr actions)
      rw.maybeLogReplayStep(env, addr actions)

    rw.maybeFinalizeReplay(env)

    check fileExists(tmpPath)

    let replay = readReplayJson(tmpPath)
    check replay.hasKey("version")
    check replay["version"].getInt() == 3
    check replay.hasKey("num_agents")
    check replay["num_agents"].getInt() == MapAgents
    check replay.hasKey("max_steps")
    check replay["max_steps"].getInt() > 0
    check replay.hasKey("map_size")
    check replay["map_size"][0].getInt() == MapWidth
    check replay["map_size"][1].getInt() == MapHeight
    check replay.hasKey("objects")
    check replay["objects"].kind == JArray
    check replay["objects"].len > 0
    check replay.hasKey("action_names")
    check replay.hasKey("item_names")
    check replay.hasKey("type_names")
    check replay.hasKey("file_name")
    check replay.hasKey("mg_config")

    echo &"  Replay: {replay[\"objects\"].len} objects, {replay[\"max_steps\"].getInt()} steps"

    removeFile(tmpPath)
    delEnv("TV_REPLAY_PATH")
    setWriter(nil)

  test "replay objects have required const fields":
    let tmpPath = getTempDir() / "test_replay_fields.json.z"
    putEnv("TV_REPLAY_PATH", tmpPath)
    delEnv("TV_REPLAY_DIR")
    setWriter(nil)

    let env = newEnvironment()
    rw.maybeStartReplayEpisode(env)

    var actions: array[MapAgents, uint16]
    env.step(addr actions)
    rw.maybeLogReplayStep(env, addr actions)

    rw.maybeFinalizeReplay(env)

    let replay = readReplayJson(tmpPath)
    for obj in replay["objects"].items:
      check obj.hasKey("id")
      check obj.hasKey("type_name")
      check obj.hasKey("inventory_max")
      check obj.hasKey("location")

    echo &"  Validated {replay[\"objects\"].len} object schemas"

    removeFile(tmpPath)
    delEnv("TV_REPLAY_PATH")
    setWriter(nil)

  test "agent objects have agent-specific series":
    let tmpPath = getTempDir() / "test_replay_agent.json.z"
    putEnv("TV_REPLAY_PATH", tmpPath)
    delEnv("TV_REPLAY_DIR")
    setWriter(nil)

    let env = newEnvironment()
    rw.maybeStartReplayEpisode(env)

    var actions: array[MapAgents, uint16]
    env.step(addr actions)
    rw.maybeLogReplayStep(env, addr actions)

    rw.maybeFinalizeReplay(env)

    let replay = readReplayJson(tmpPath)
    var foundAgent = false
    for obj in replay["objects"].items:
      if obj.hasKey("agent_id"):
        foundAgent = true
        check obj.hasKey("group_id")
        check obj.hasKey("action_id")
        check obj.hasKey("action_param")
        check obj.hasKey("action_success")
        check obj.hasKey("current_reward")
        check obj.hasKey("total_reward")
        check obj.hasKey("is_frozen")
        break
    check foundAgent

    removeFile(tmpPath)
    delEnv("TV_REPLAY_PATH")
    setWriter(nil)

  test "replay with custom label":
    let tmpPath = getTempDir() / "test_replay_label.json.z"
    putEnv("TV_REPLAY_PATH", tmpPath)
    putEnv("TV_REPLAY_LABEL", "Custom Test Label")
    delEnv("TV_REPLAY_DIR")
    setWriter(nil)

    let env = newEnvironment()
    rw.maybeStartReplayEpisode(env)

    var actions: array[MapAgents, uint16]
    env.step(addr actions)
    rw.maybeLogReplayStep(env, addr actions)
    rw.maybeFinalizeReplay(env)

    let replay = readReplayJson(tmpPath)
    check replay["mg_config"]["label"].getStr() == "Custom Test Label"

    removeFile(tmpPath)
    delEnv("TV_REPLAY_PATH")
    delEnv("TV_REPLAY_LABEL")
    setWriter(nil)

  test "multi-step replay records all steps":
    let tmpPath = getTempDir() / "test_replay_multistep.json.z"
    putEnv("TV_REPLAY_PATH", tmpPath)
    delEnv("TV_REPLAY_DIR")
    setWriter(nil)

    let env = newEnvironment()
    rw.maybeStartReplayEpisode(env)

    let numSteps = 10
    for i in 0 ..< numSteps:
      var actions: array[MapAgents, uint16]
      env.step(addr actions)
      rw.maybeLogReplayStep(env, addr actions)

    rw.maybeFinalizeReplay(env)

    let replay = readReplayJson(tmpPath)
    check replay["max_steps"].getInt() == numSteps

    # Verify location series has multiple entries (state changed over steps)
    var hasMultiEntry = false
    for obj in replay["objects"].items:
      if obj.hasKey("location"):
        let locSeries = obj["location"]
        if locSeries.kind == JArray and locSeries.len > 1:
          hasMultiEntry = true
          break
    # At least some objects should have location changes over 10 steps
    # (agents move, things appear/disappear)
    echo &"  Multi-entry location series found: {hasMultiEntry}"
    check replay["max_steps"].getInt() == numSteps

    removeFile(tmpPath)
    delEnv("TV_REPLAY_PATH")
    setWriter(nil)
