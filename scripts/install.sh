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
  mkdir -p build
  cd build

  if [ -n "$ONNXRuntime_DIR" ]; then
    echo "Using ONNXRuntime from $ONNXRuntime_DIR"
    cmake .. -DONNXRuntime_DIR="$ONNXRuntime_DIR"
  else
    cmake ..
  fi

  # Determine the number of parallel build jobs in a cross-platform way
  JOBS=1
  if command -v nproc >/dev/null 2>&1; then
    JOBS=$(nproc)
  elif command -v sysctl >/dev/null 2>&1; then
    JOBS=$(sysctl -n hw.ncpu)
  fi
  cmake --build . --parallel "$JOBS"



  cmake ..
  make -j$(nproc)

  cd ../..
fi

echo "Setup complete."
