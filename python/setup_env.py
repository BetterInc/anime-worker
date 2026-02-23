#!/usr/bin/env python3
"""First-run setup: creates a virtual environment and installs requirements."""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
VENV_DIR = SCRIPT_DIR / "venv"
REQUIREMENTS = SCRIPT_DIR / "requirements.txt"


def main():
    print(f"Setting up Python environment in {VENV_DIR}")

    if not VENV_DIR.exists():
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])

    # Determine pip path
    if os.name == "nt":
        pip = str(VENV_DIR / "Scripts" / "pip")
        python = str(VENV_DIR / "Scripts" / "python")
    else:
        pip = str(VENV_DIR / "bin" / "pip")
        python = str(VENV_DIR / "bin" / "python")

    print("Upgrading pip...")
    subprocess.check_call([python, "-m", "pip", "install", "--upgrade", "pip"])

    print("Installing PyTorch (CUDA)...")
    subprocess.check_call([
        python, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ])

    if REQUIREMENTS.exists():
        print(f"Installing requirements from {REQUIREMENTS}...")
        subprocess.check_call([python, "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    print("\nSetup complete!")
    print(f"Python: {python}")
    print(f"Update your config.toml: python_path = \"{python}\"")


if __name__ == "__main__":
    main()
