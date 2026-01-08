#!/bin/bash

set -euo pipefail

# Default to kubectl proxy URL, but allow override via environment variable
export RAY_API_SERVER_ADDRESS="${RAY_API_SERVER_ADDRESS:-http://localhost:8001/api/v1/namespaces/subsalt/services/subsalt-raycluster-head-svc:8265/proxy}"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <python_script>"
    echo "Example: $0 helloworld.py"
    exit 1
fi

SCRIPT_PATH="$1"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: File '$SCRIPT_PATH' not found"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
SCRIPT_NAME=$(basename "$SCRIPT_PATH")

# Find the project root (where mostlyai/ directory lives)
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)

# Create temporary directory for Python files only
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Preparing Python files for upload..."

# Copy all Python files from SCRIPT_DIR to temp directory, preserving structure
cd "$SCRIPT_DIR"
find . -name "*.py" -type f | while read -r pyfile; do
    target_dir="$TEMP_DIR/$(dirname "$pyfile")"
    mkdir -p "$target_dir"
    cp "$pyfile" "$target_dir/"
done

# Also copy the mostlyai library code from project root
if [ -d "$PROJECT_ROOT/mostlyai" ]; then
    echo "Including mostlyai library code from project root..."
    cd "$PROJECT_ROOT"
    find mostlyai -name "*.py" -type f | while read -r pyfile; do
        target_dir="$TEMP_DIR/$(dirname "$pyfile")"
        mkdir -p "$target_dir"
        cp "$pyfile" "$target_dir/"
    done
fi

# Count files being uploaded
PY_FILE_COUNT=$(find "$TEMP_DIR" -name "*.py" -type f | wc -l | tr -d ' ')

echo "Submitting Ray job..."
echo "  Script: $SCRIPT_NAME"
echo "  Source directory: $SCRIPT_DIR"
echo "  Python files to upload: $PY_FILE_COUNT"
echo "  Ray API: $RAY_API_SERVER_ADDRESS"
echo ""

uvx --with "ray[default]" ray job submit --address "$RAY_API_SERVER_ADDRESS" --working-dir "$TEMP_DIR" -- python "$SCRIPT_NAME"
