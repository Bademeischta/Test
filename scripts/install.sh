#!/usr/bin/env bash
set -e

# Install Python dependencies
if [ -f "requirements.txt" ]; then
  echo "Installing Python requirements..."
  pip install -r requirements.txt
fi

# Optional build of the C++ superengine
if [ -d "superengine" ]; then
  echo "Building superengine..."
  cd superengine
  mkdir -p build
  cd build
  cmake ..
  make -j$(nproc)
  cd ../..
fi

echo "Setup complete."
