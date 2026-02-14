#!/usr/bin/env bash
# Build Tribal Village shared libraries for different team counts.
# Team count is a compile-time constant in Nim, so we must recompile for each.
#
# Usage: ./build_team_configs.sh
# Produces: libtribal_village_teams{2,4,8}.so

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TYPES_FILE="$SCRIPT_DIR/src/types.nim"
REGISTRY_FILE="$SCRIPT_DIR/src/registry.nim"
NIM_BIN="$HOME/.nimby/nim/bin/nim"

# Backup original source files
cp "$TYPES_FILE" "$TYPES_FILE.bak"
cp "$REGISTRY_FILE" "$REGISTRY_FILE.bak"

restore_original() {
    cp "$TYPES_FILE.bak" "$TYPES_FILE"
    cp "$REGISTRY_FILE.bak" "$REGISTRY_FILE"
    rm -f "$TYPES_FILE.bak" "$REGISTRY_FILE.bak"
}
trap restore_original EXIT

build_for_teams() {
    local num_teams=$1
    echo "=== Building for $num_teams teams ==="

    # Restore originals first
    cp "$TYPES_FILE.bak" "$TYPES_FILE"
    cp "$REGISTRY_FILE.bak" "$REGISTRY_FILE"

    # Modify source for this team count
    python3 "$SCRIPT_DIR/modify_team_count.py" "$TYPES_FILE" "$num_teams"

    # Compile
    echo "  Compiling..."
    "$NIM_BIN" c --app:lib --mm:arc --opt:speed -d:danger \
        --out:"$SCRIPT_DIR/libtribal_village_teams${num_teams}.so" \
        "$SCRIPT_DIR/src/ffi.nim"

    local expected_agents=$(( num_teams * 125 + 6 ))
    echo "  Built libtribal_village_teams${num_teams}.so ($expected_agents agents)"
}

# Build for each team count
for teams in 2 4 8; do
    build_for_teams $teams
done

echo ""
echo "=== All builds complete ==="
ls -la "$SCRIPT_DIR"/libtribal_village_teams*.so
