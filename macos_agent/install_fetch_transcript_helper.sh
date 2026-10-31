#!/bin/zsh
# Install the small upstream macOS helper that obtains an Apple API token.
# It runs locally and never opens the Podcasts application.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/bin"
TARGET="$TARGET_DIR/FetchTranscript"
SOURCE_REPOSITORY="https://github.com/dado3212/apple-podcast-transcript-downloader.git"
SOURCE_COMMIT="535e0ce8ddb609f27ee5b169d910b3517944e9f6"
BUILD_DIR="$(mktemp -d /private/tmp/apple-transcript-helper.XXXXXX)"

cleanup() {
  rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

mkdir -p "$TARGET_DIR"
git clone --quiet "$SOURCE_REPOSITORY" "$BUILD_DIR"
git -C "$BUILD_DIR" checkout --quiet "$SOURCE_COMMIT"
clang -Wno-objc-method-access -framework Foundation \
  -F/System/Library/PrivateFrameworks -framework AppleMediaServices \
  "$BUILD_DIR/FetchTranscript.m" -o "$TARGET"
chmod 700 "$TARGET"
printf 'Installed Apple credential helper: %s\n' "$TARGET"
