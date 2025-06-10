#!/usr/bin/env bash
set -e

install_make() {
  if command -v make >/dev/null 2>&1; then
    return
  fi

  echo "make not found. Attempting to install build tools..."
  case "$(uname -s)" in
    Linux*)
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y build-essential make
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*)
      if ! command -v choco >/dev/null 2>&1; then
        echo "Chocolatey not found. Installing..."
        powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
        export PATH="$PATH:/c/ProgramData/chocolatey/bin"
      fi
      choco install -y make
      ;;
  esac
}

# Install Python dependencies
if [ -f "requirements.txt" ]; then
  echo "Installing Python requirements..."
  pip install -r requirements.txt
  pip install -e .
fi

install_make

# Optional build of the C++ superengine
if [ -d "superengine" ]; then
  echo "Building superengine..."
  cd superengine
  make -j"$(nproc)"
  cd ..
fi

echo "Setup complete."
