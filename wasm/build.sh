#!/bin/bash
set -e

echo "🔧 Building AntiGoogle WASM (pinned wasm-bindgen 0.2.84)"

# Install wasm-pack if needed
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

# Install matching wasm-bindgen-cli version
echo "Installing wasm-bindgen-cli 0.2.92..."
cargo install wasm-bindgen-cli --version 0.2.92 --force

# Add target
rustup target add wasm32-unknown-unknown

# Clean previous build
rm -rf pkg target

# Generate hyperplanes (must match server seeds!)
python3 generate_planes.py

# Build
wasm-pack build --target web --release
cp antigoogle_client.js pkg/.

# Check size
echo ""
echo "📦 Build complete:"
ls -lh pkg/*.wasm
