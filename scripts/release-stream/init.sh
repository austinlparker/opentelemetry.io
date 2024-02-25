#!/bin/bash

REPO="https://github.com/open-telemetry/opentelemetry.io"
CONTENT_FILES="content"

# Cleanup existing repo if it exists
if [ -d "./opentelemetry.io" ]; then
    rm -rf ./opentelemetry.io
fi

git clone --filter=blob:none --no-checkout "$REPO" opentelemetry.io
cd ./opentelemetry.io
git sparse-checkout init --cone
git sparse-checkout set "$CONTENT_FILES"

git checkout
