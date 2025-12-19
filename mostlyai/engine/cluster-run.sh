#!/bin/bash

set -euo pipefail

# Default to kubectl proxy URL, but allow override via environment variable
export RAY_API_SERVER_ADDRESS="${RAY_API_SERVER_ADDRESS:-http://localhost:8001/api/v1/namespaces/subsalt/services/subsalt-raycluster-head-svc:8265/proxy}"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <python_script>"
    echo "Example: $0 slim-argn.py"
    exit 1
fi

SCRIPT_PATH="$1"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: File '$SCRIPT_PATH' not found"
    exit 1
fi

SCRIPT_NAME=$(basename "$SCRIPT_PATH")
# Get absolute path to script directory
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
# Full absolute path to the script
SCRIPT_FULL_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# Find the repository root (where pyproject.toml is)
REPO_ROOT="$SCRIPT_DIR"
while [ "$REPO_ROOT" != "/" ] && [ ! -f "$REPO_ROOT/pyproject.toml" ]; do
    REPO_ROOT=$(dirname "$REPO_ROOT")
done

if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
    echo "Error: Could not find pyproject.toml in parent directories"
    exit 1
fi

# Create temporary directory with full package structure
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Preparing package for upload..."

# Copy the entire mostlyai package
cp -r "$REPO_ROOT/mostlyai" "$TEMP_DIR/"

# Copy the script to the root of temp dir (if not already in mostlyai/)
if [[ "$SCRIPT_DIR" != *"mostlyai"* ]]; then
    cp "$SCRIPT_PATH" "$TEMP_DIR/"
fi

# Determine the relative script path for execution (use full path for matching)
if [[ "$SCRIPT_FULL_PATH" == *"mostlyai/engine/"* ]]; then
    EXEC_SCRIPT="mostlyai/engine/$SCRIPT_NAME"
elif [[ "$SCRIPT_FULL_PATH" == *"mostlyai/"* ]]; then
    EXEC_SCRIPT="mostlyai/$SCRIPT_NAME"
else
    EXEC_SCRIPT="$SCRIPT_NAME"
fi

# Count files being uploaded
PY_FILE_COUNT=$(find "$TEMP_DIR" -name "*.py" -type f | wc -l | tr -d ' ')

echo "Submitting Ray job..."
echo "  Script: $EXEC_SCRIPT"
echo "  Repository root: $REPO_ROOT"
echo "  Python files to upload: $PY_FILE_COUNT"
echo "  Ray API: $RAY_API_SERVER_ADDRESS"
echo ""

uvx --with "ray[default]" ray job submit --address "$RAY_API_SERVER_ADDRESS" --working-dir "$TEMP_DIR" -- python "$EXEC_SCRIPT"
