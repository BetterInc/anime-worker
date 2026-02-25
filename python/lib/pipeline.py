"""Model pipeline loading, weight fixing, and offload configuration.

Copied from anime-video-gen/scripts/lib/pipeline.py for standalone worker use.
"""

import logging
from pathlib import Path

import diffusers
import torch

from .hardware import (
    _compute_num_blocks_per_group,
    detect_vram,
    estimate_model_memory_gb,
    system_ram_gb,
)

logger = logging.getLogger(__name__)


def find_model_path(base_path, local_dir=None):
    """Find a downloaded model in the base path."""
    if local_dir:
        candidate = base_path / local_dir
        if candidate.exists() and (candidate / "model_index.json").exists():
            logger.info(f"Found model: {candidate.name}")
            return candidate
        raise FileNotFoundError(
            f"Model directory not found: {candidate}. "
            f"The worker should have downloaded this model."
        )

    # Scan for any model
    for entry in base_path.iterdir():
        if entry.is_dir() and (entry / "model_index.json").exists():
            logger.info(f"Found model: {entry.name}")
            return entry

    raise FileNotFoundError(f"No model found in {base_path}.")


def fix_text_encoder_weight_tying(pipe):
    """Fix text encoder weight tying that some diffusers versions break on load."""
    for attr_name in dir(pipe):
        if not attr_name.startswith("text_encoder"):
            continue
        text_enc = getattr(pipe, attr_name, None)
        if text_enc is None or not hasattr(text_enc, "parameters"):
            continue
        shared = getattr(text_enc, "shared", None)
        if shared is None:
            continue
        for submodule_name in ["encoder", "decoder"]:
            submodule = getattr(text_enc, submodule_name, None)
            if submodule is None:
                continue
            embed = getattr(submodule, "embed_tokens", None)
            if embed is None:
                continue
            if shared.weight.data_ptr() != embed.weight.data_ptr():
                embed.weight = shared.weight
                logger.info(
                    f"  Fixed {attr_name} weight tying (shared -> {submodule_name}.embed_tokens)"
                )


def setup_pipeline(config, model_config=None):
    """Setup model pipeline. Auto-detects hardware and picks the best loading strategy."""
    # Configure PyTorch memory allocator to reduce fragmentation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Clear any existing CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache before model loading")

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    # Allow CPU for testing (will be slow!)
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - using CPU (will be very slow!)")
        gpus = []
    else:
        gpus = detect_vram()
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPUs detected: {len(gpus)}")
    for gpu_id, total_gb, free_gb in gpus:
        logger.info(
            f"  GPU {gpu_id}: {torch.cuda.get_device_properties(gpu_id).name} -- {total_gb:.1f} GB total, {free_gb:.1f} GB free"
        )

    # Use 'current' key (new) or fallback to 'wan22' (old) for backwards compat
    pipeline_model_config = config["model"].get(
        "current", config["model"].get("wan22", {})
    )
    if model_config:
        pipeline_model_config = {**pipeline_model_config, **model_config}
    model_config = pipeline_model_config

    base_path = Path(config["model"]["base_path"])
    local_dir = model_config.get("local_dir")
    model_path = find_model_path(base_path, local_dir=local_dir)

    dtype_str = model_config.get("model_dtype") or model_config.get(
        "convert_model_dtype", "bf16"
    )
    dtype_map = {
        "fp8": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    dtype_bytes = 2 if dtype_str in ("bf16", "bfloat16", "fp16", "float16") else 4
    model_memory_gb = estimate_model_memory_gb(model_path, dtype_bytes)
    logger.info(f"Estimated model memory: {model_memory_gb:.1f} GB (dtype={dtype_str})")
    logger.info(f"Loading {model_path.name} (dtype={dtype_str} -> {dtype})")

    pipeline_classes = {}
    for cls_name in [
        "WanPipeline",
        "MochiPipeline",
        "LTXPipeline",
        "HunyuanVideoPipeline",
        "TextToVideoSDPipeline",
    ]:
        cls = getattr(diffusers, cls_name, None)
        if cls:
            pipeline_classes[cls_name] = cls
    pipeline_class_name = model_config.get("model_class", "WanPipeline")
    logger.info(f"  Requested pipeline class: {pipeline_class_name}")
    logger.info(f"  model_config keys: {list(model_config.keys())}")
    PipelineClass = pipeline_classes.get(pipeline_class_name)
    if PipelineClass is None:
        raise RuntimeError(
            f"Pipeline class '{pipeline_class_name}' not found in diffusers {diffusers.__version__}. "
            f"Available: {list(pipeline_classes.keys())}. Try upgrading diffusers."
        )
    logger.info(f"  Using pipeline class: {PipelineClass.__name__}")

    load_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
    }

    if model_config.get("vae_class") == "AutoencoderKLWan":
        AutoencoderKLWan = getattr(diffusers, "AutoencoderKLWan", None)
        if AutoencoderKLWan is None:
            raise RuntimeError(
                f"AutoencoderKLWan not found in diffusers {diffusers.__version__}. "
                "Upgrade diffusers to use Wan models."
            )
        vae_dtype = (
            torch.float32 if model_config.get("vae_dtype") == "float32" else dtype
        )
        load_kwargs["vae"] = AutoencoderKLWan.from_pretrained(
            str(model_path), subfolder="vae", torch_dtype=vae_dtype
        )
        logger.info(f"  VAE loaded (dtype={vae_dtype})")

    # CPU-only mode if no GPUs
    if not gpus:
        logger.warning("No GPUs detected - loading model on CPU (will be VERY slow)")
        pipe = PipelineClass.from_pretrained(str(model_path), **load_kwargs)
        fix_text_encoder_weight_tying(pipe)
        # Keep on CPU
        logger.info("  Model loaded on CPU")
        return pipe

    total_free_vram = sum(free_gb for _, _, free_gb in gpus)
    best_gpu = max(gpus, key=lambda g: g[2])
    best_gpu_id, best_gpu_total, best_gpu_free = best_gpu
    loaded = False

    if best_gpu_free >= model_memory_gb + 8:
        pipe = PipelineClass.from_pretrained(str(model_path), **load_kwargs)
        fix_text_encoder_weight_tying(pipe)
        pipe = pipe.to(f"cuda:{best_gpu_id}")
        logger.info(
            f"  Loaded entirely on GPU {best_gpu_id} ({best_gpu_free:.0f}GB free)"
        )
        loaded = True

    if not loaded and len(gpus) > 1 and total_free_vram >= model_memory_gb + 8:
        try:
            pipe = PipelineClass.from_pretrained(
                str(model_path), device_map="balanced", **load_kwargs
            )
            fix_text_encoder_weight_tying(pipe)
            # Move VAE to GPU if it was loaded separately
            if hasattr(pipe, "vae") and pipe.vae is not None:
                vae_device = f"cuda:{best_gpu_id}"
                pipe.vae = pipe.vae.to(vae_device)
                logger.info(f"  VAE moved to {vae_device}")
            logger.info(
                f"  Model split across {len(gpus)} GPUs via device_map ({total_free_vram:.0f}GB total free)"
            )
            loaded = True
        except Exception as e:
            logger.warning(f"  device_map failed ({e}), falling back to group offload")

    if not loaded:
        ram_total, ram_used = system_ram_gb()
        ram_free = ram_total - ram_used
        if ram_free < model_memory_gb + 20:
            raise RuntimeError(
                f"Not enough RAM to load model: need ~{model_memory_gb + 20:.0f}GB free, "
                f"have {ram_free:.0f}GB. Free up memory or use a smaller model."
            )
        pipe = PipelineClass.from_pretrained(str(model_path), **load_kwargs)
        fix_text_encoder_weight_tying(pipe)
        try:
            num_blocks = _compute_num_blocks_per_group(best_gpu_free, model_memory_gb)
            use_stream = num_blocks == 1
            pipe.enable_group_offload(
                onload_device=torch.device(f"cuda:{best_gpu_id}"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=num_blocks,
                use_stream=use_stream,
            )
            logger.info(
                f"  Group offload on GPU {best_gpu_id} (num_blocks={num_blocks}, stream={use_stream})"
            )
            loaded = True
        except Exception as e:
            logger.warning(f"  Group offload failed ({e}), using CPU offload")
            pipe.enable_model_cpu_offload(gpu_id=best_gpu_id)
            logger.info(f"  CPU offload on GPU {best_gpu_id}")

    optimizations = [
        ("xformers", "enable_xformers_memory_efficient_attention"),
        ("enable_vae_tiling", "enable_vae_tiling"),
        ("enable_vae_slicing", "enable_vae_slicing"),
    ]
    for opt_key, method_name in optimizations:
        if opt_key == "xformers" or model_config.get(opt_key):
            try:
                getattr(pipe, method_name)()
                logger.info(f"  {opt_key} enabled")
            except (AttributeError, ImportError):
                logger.warning(f"  {opt_key} not supported, skipping")

    logger.info(f"{model_path.name} loaded successfully")
    return pipe
