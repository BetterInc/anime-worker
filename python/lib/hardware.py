"""GPU detection, loading strategy selection, and hardware reporting.

Copied from anime-video-gen/scripts/lib/hardware.py with minor adaptations
for standalone worker use (no database dependencies).
"""

import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_gpu_count():
    """Detect number of GPUs WITHOUT importing torch."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            return len(lines)
    except (FileNotFoundError, Exception):
        pass
    return 0


def estimate_model_memory_gb(model_path, dtype_bytes=2):
    """Estimate model memory usage in GB when loaded in the given dtype."""
    total_params = 0
    for index_file in model_path.rglob("*.safetensors.index.json"):
        try:
            with open(index_file) as f:
                index = json.load(f)
            if "metadata" in index and "total_size" in index["metadata"]:
                total_params += int(index["metadata"]["total_size"])
        except Exception:
            pass

    if total_params > 0:
        return (total_params / (1024 ** 3)) * (dtype_bytes / 2)

    total_bytes = 0
    for ext in ['*.safetensors', '*.bin']:
        for f in model_path.rglob(ext):
            total_bytes += f.stat().st_size
    return (total_bytes / (1024 ** 3)) * (dtype_bytes / 2)


def detect_vram():
    """Detect per-GPU VRAM. Returns list of (gpu_id, total_gb, free_gb)."""
    import torch
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        free_mem, _ = torch.cuda.mem_get_info(i)
        free_gb = free_mem / (1024 ** 3)
        gpus.append((i, total_gb, free_gb))
    return gpus


def detect_vram_lightweight():
    """Detect per-GPU VRAM via nvidia-smi. No torch import."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            gpus.append((
                int(parts[0]),
                int(parts[1]) / 1024,
                int(parts[2]) / 1024,
            ))
        return gpus
    except Exception:
        return []


def system_ram_gb():
    """Return (total_gb, used_gb)."""
    try:
        if os.name == 'nt':
            # Windows
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total = stat.ullTotalPhys / (1024 ** 3)
            avail = stat.ullAvailPhys / (1024 ** 3)
            return round(total, 1), round(total - avail, 1)
        else:
            # Linux/macOS
            with open('/proc/meminfo') as f:
                info = {}
                for line in f:
                    parts = line.split()
                    if parts[0] in ('MemTotal:', 'MemAvailable:'):
                        info[parts[0]] = int(parts[1]) / (1024 * 1024)
                total = info.get('MemTotal:', 0)
                avail = info.get('MemAvailable:', 0)
                return round(total, 1), round(total - avail, 1)
    except Exception:
        return 0, 0


def _compute_num_blocks_per_group(vram_gb, model_memory_gb):
    """Compute optimal num_blocks_per_group for group offload."""
    if model_memory_gb > 20:
        est_blocks = 40
    elif model_memory_gb > 8:
        est_blocks = 30
    else:
        est_blocks = 24
    block_gb = model_memory_gb / est_blocks

    inference_reserve_gb = min(vram_gb * 0.45, 12)
    available_for_blocks = vram_gb - inference_reserve_gb
    if available_for_blocks <= 0:
        return 1

    num_blocks = max(1, int(available_for_blocks / block_gb))
    num_blocks = min(num_blocks, 6)

    logger.info(
        f"  Group offload tuning: {block_gb:.1f}GB/block, "
        f"{available_for_blocks:.1f}GB available -> num_blocks_per_group={num_blocks}"
    )
    return num_blocks
