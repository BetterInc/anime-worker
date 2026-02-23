#!/usr/bin/env python3
"""First-run setup: creates a virtual environment and installs requirements."""

import os
import platform
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

SCRIPT_DIR = Path(__file__).parent
VENV_DIR = SCRIPT_DIR / "venv"
PYTHON_DIR = SCRIPT_DIR / ".python"
REQUIREMENTS = SCRIPT_DIR / "requirements.txt"

# Standalone Python download URLs
PYTHON_URLS = {
    "Windows": "https://www.python.org/ftp/python/3.11.8/python-3.11.8-embed-amd64.zip",
    "Linux": "https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.11.8+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz",
    "Darwin": "https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.11.8+20240224-aarch64-apple-darwin-install_only.tar.gz",
}


def find_compatible_python():
    """Find or download a compatible Python version (3.10-3.12)."""
    # Check current Python first
    current_version = sys.version_info
    if (3, 10) <= current_version < (3, 13):
        print(f"✓ Using current Python {current_version.major}.{current_version.minor}")
        return sys.executable

    print(f"⚠ Current Python {current_version.major}.{current_version.minor} not compatible (need 3.10-3.12)")

    # Try to find installed Python versions
    print("Searching for compatible Python...")
    system = platform.system()

    candidates = []
    if system == "Windows":
        candidates = ["py", "python", "python3", "python3.12", "python3.11", "python3.10"]
    else:
        candidates = ["python3.12", "python3.11", "python3.10", "python3"]

    for cmd in candidates:
        try:
            if cmd == "py":
                # Windows py launcher - try specific versions
                for ver in ["3.12", "3.11", "3.10"]:
                    try:
                        result = subprocess.run(
                            [cmd, f"-{ver}", "--version"],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        print(f"✓ Found: {result.stdout.strip()}")
                        return [cmd, f"-{ver}"]
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
            else:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Check version
                version_str = result.stdout.strip()
                if "Python 3.1" in version_str:  # 3.10, 3.11, 3.12
                    major, minor = version_str.split()[1].split('.')[:2]
                    if (3, 10) <= (int(major), int(minor)) < (3, 13):
                        print(f"✓ Found: {version_str}")
                        return cmd
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    # No compatible Python found - download portable Python
    print("\n⚠ No compatible Python found on system")
    print("📦 Downloading portable Python 3.11 (one-time setup)...")
    return download_portable_python()


def download_portable_python():
    """Download and extract portable Python."""
    system = platform.system()

    if system not in PYTHON_URLS:
        print(f"ERROR: Unsupported platform: {system}")
        print("Please install Python 3.11 manually from https://www.python.org/downloads/")
        sys.exit(1)

    url = PYTHON_URLS[system]
    PYTHON_DIR.mkdir(exist_ok=True)

    filename = PYTHON_DIR / url.split("/")[-1]

    if not filename.exists():
        print(f"Downloading from {url}")
        urlretrieve(url, filename)
        print("✓ Download complete")

    print("Extracting...")
    if filename.suffix == ".zip":
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(PYTHON_DIR)
    else:
        with tarfile.open(filename, 'r:gz') as tar_ref:
            tar_ref.extractall(PYTHON_DIR)

    # Find python executable
    if system == "Windows":
        python_exe = PYTHON_DIR / "python.exe"
    else:
        python_exe = PYTHON_DIR / "python" / "bin" / "python3"
        if not python_exe.exists():
            python_exe = PYTHON_DIR / "bin" / "python3"

    if not python_exe.exists():
        print(f"ERROR: Could not find python executable in {PYTHON_DIR}")
        sys.exit(1)

    print(f"✓ Portable Python ready at {python_exe}")
    return str(python_exe)


def main():
    print("=" * 60)
    print("🐍 Python Environment Setup")
    print("=" * 60)

    # Find or download compatible Python
    python_exe = find_compatible_python()

    # Handle py launcher list format
    if isinstance(python_exe, list):
        create_cmd = python_exe + ["-m", "venv", str(VENV_DIR)]
    else:
        create_cmd = [python_exe, "-m", "venv", str(VENV_DIR)]

    if not VENV_DIR.exists():
        print(f"\n📦 Creating virtual environment...")
        subprocess.check_call(create_cmd)

    # Determine pip path
    if os.name == "nt":
        pip = str(VENV_DIR / "Scripts" / "pip")
        python = str(VENV_DIR / "Scripts" / "python")
    else:
        pip = str(VENV_DIR / "bin" / "pip")
        python = str(VENV_DIR / "bin" / "python")

    print("\n⬆️  Upgrading pip...")
    subprocess.check_call([python, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    print("\n🔥 Installing PyTorch (CUDA 12.4)...")
    subprocess.check_call([
        python, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ])

    if REQUIREMENTS.exists():
        print(f"\n📚 Installing dependencies from {REQUIREMENTS.name}...")
        subprocess.check_call([python, "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print(f"Python: {python}")
    print(f"\n💡 Config will auto-update. Just run: anime-worker run")
    print("=" * 60)


if __name__ == "__main__":
    main()
