"""GPU detection, loading strategy selection, and hardware reporting.

Copied from anime-video-gen/scripts/lib/hardware.py with minor adaptations
for standalone worker use (no database dependencies).
"""

import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def detect_gpu_count():
    """Detect number of GPUs WITHOUT importing torch."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
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
        return (total_params / (1024**3)) * (dtype_bytes / 2)

    total_bytes = 0
    for ext in ["*.safetensors", "*.bin"]:
        for f in model_path.rglob(ext):
            total_bytes += f.stat().st_size
    return (total_bytes / (1024**3)) * (dtype_bytes / 2)


def detect_vram():
    """Detect per-GPU VRAM. Returns list of (gpu_id, total_gb, free_gb)."""
    import torch

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        free_mem, _ = torch.cuda.mem_get_info(i)
        free_gb = free_mem / (1024**3)
        gpus.append((i, total_gb, free_gb))
    return gpus


def detect_vram_lightweight():
    """Detect per-GPU VRAM via nvidia-smi. No torch import."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            gpus.append(
                (
                    int(parts[0]),
                    int(parts[1]) / 1024,
                    int(parts[2]) / 1024,
                )
            )
        return gpus
    except Exception:
        return []


def system_ram_gb():
    """Return (total_gb, used_gb)."""
    try:
        if os.name == "nt":
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
            total = stat.ullTotalPhys / (1024**3)
            avail = stat.ullAvailPhys / (1024**3)
            return round(total, 1), round(total - avail, 1)
        else:
            # Linux/macOS
            with open("/proc/meminfo") as f:
                info = {}
                for line in f:
                    parts = line.split()
                    if parts[0] in ("MemTotal:", "MemAvailable:"):
                        info[parts[0]] = int(parts[1]) / (1024 * 1024)
                total = info.get("MemTotal:", 0)
                avail = info.get("MemAvailable:", 0)
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


class VRAMManager:
    """Generic VRAM manager for any diffusion pipeline.

    Handles:
    - Tracking VRAM usage
    - Offloading model components to CPU when needed
    - Reloading components for next inference
    - Progress reporting
    """

    # Components that can be offloaded, in order of size (largest first)
    OFFLOADABLE_COMPONENTS = [
        "transformer",  # Wan, Flux
        "unet",  # SD, SDXL
        "text_encoder_2",
        "text_encoder",
        "text_encoder_3",
    ]

    def __init__(self, pipeline, device="cuda:0", min_free_vram_gb=6.0):
        self.pipeline = pipeline
        self.device = device
        self.min_free_vram_gb = min_free_vram_gb
        self.offloaded_components = {}  # name -> original device

    def get_free_vram_gb(self):
        """Get current free VRAM in GB."""
        import torch

        if not torch.cuda.is_available():
            return 0.0
        device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
        free, _ = torch.cuda.mem_get_info(device_idx)
        return free / (1024**3)

    def get_total_vram_gb(self):
        """Get total VRAM in GB."""
        import torch

        if not torch.cuda.is_available():
            return 0.0
        device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
        _, total = torch.cuda.mem_get_info(device_idx)
        return total / (1024**3)

    def log_vram(self, phase=""):
        """Log current VRAM status."""
        free = self.get_free_vram_gb()
        total = self.get_total_vram_gb()
        prefix = f"  [{phase}] " if phase else "  "
        logger.info(f"{prefix}VRAM: {free:.1f}GB free / {total:.1f}GB total")
        return free

    def offload_for_vae(self, progress_callback=None):
        """Offload model components to make room for VAE decode.

        Returns True if offloading was performed.
        """
        import torch

        free_gb = self.get_free_vram_gb()

        if free_gb >= self.min_free_vram_gb:
            logger.info(f"  VRAM OK ({free_gb:.1f}GB free), no offload needed")
            return False

        logger.info(
            f"  Low VRAM ({free_gb:.1f}GB free < {self.min_free_vram_gb}GB required)"
        )
        logger.info("  Offloading model components to CPU for VAE decode...")

        if progress_callback:
            progress_callback(0, "Offloading model for VAE...")

        offloaded_count = 0
        for component_name in self.OFFLOADABLE_COMPONENTS:
            component = getattr(self.pipeline, component_name, None)
            if component is None:
                continue

            # Check if it's actually on GPU
            try:
                param = next(component.parameters())
                if param.device.type != "cuda":
                    continue
                original_device = str(param.device)
            except StopIteration:
                continue

            # Offload to CPU
            logger.info(f"    Moving {component_name} to CPU...")
            component.to("cpu")
            self.offloaded_components[component_name] = original_device
            offloaded_count += 1

            # Clear cache and check if we have enough VRAM now
            torch.cuda.empty_cache()
            free_gb = self.get_free_vram_gb()

            if progress_callback:
                progress_callback(offloaded_count * 20, f"Offloaded {component_name}")

            if free_gb >= self.min_free_vram_gb:
                logger.info(f"  VRAM now OK ({free_gb:.1f}GB free)")
                break

        final_free = self.get_free_vram_gb()
        logger.info(
            f"  Offloaded {offloaded_count} components, VRAM now {final_free:.1f}GB free"
        )

        if progress_callback:
            progress_callback(100, "Model offloaded")

        return offloaded_count > 0

    def reload_offloaded(self, progress_callback=None):
        """Reload previously offloaded components back to GPU."""
        import torch

        if not self.offloaded_components:
            return False

        logger.info("  Reloading offloaded components to GPU...")

        if progress_callback:
            progress_callback(0, "Reloading model...")

        total = len(self.offloaded_components)
        for idx, (component_name, original_device) in enumerate(
            self.offloaded_components.items()
        ):
            component = getattr(self.pipeline, component_name, None)
            if component is not None:
                logger.info(f"    Moving {component_name} back to {original_device}...")
                component.to(original_device)

                if progress_callback:
                    progress_callback(
                        int((idx + 1) / total * 100), f"Reloaded {component_name}"
                    )

        self.offloaded_components.clear()
        torch.cuda.empty_cache()

        if progress_callback:
            progress_callback(100, "Model reloaded")

        return True


def get_optimal_vae_batch_size(free_vram_gb, frame_height=480, frame_width=832):
    """Calculate optimal batch size for VAE decode based on available VRAM.

    Args:
        free_vram_gb: Available VRAM in GB
        frame_height: Frame height in pixels
        frame_width: Frame width in pixels

    Returns:
        Optimal batch size for VAE decoding
    """
    # Rough estimate: VAE decode needs ~0.5GB per frame at 832x480
    # Scale with resolution
    pixels = frame_height * frame_width
    base_pixels = 480 * 832
    mem_per_frame_gb = 0.5 * (pixels / base_pixels)

    # Leave some headroom
    usable_vram = free_vram_gb * 0.8

    batch_size = max(1, int(usable_vram / mem_per_frame_gb))

    # Cap at reasonable values
    batch_size = min(batch_size, 32)

    logger.info(
        f"  VAE batch size: {batch_size} (based on {free_vram_gb:.1f}GB free VRAM)"
    )
    return batch_size
