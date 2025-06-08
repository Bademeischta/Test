#!/usr/bin/env bash
set -e

# Install Python dependencies
if [ -f "requirements.txt" ]; then
  echo "Installing Python requirements..."
  pip install -r requirements.txt
  pip install -e .
fi

# Optional build of the C++ superengine
if [ -d "superengine" ]; then
  echo "Building superengine..."
  cd superengine
  make -j$(nproc)
  cd ..
fi

echo "Setup complete."
