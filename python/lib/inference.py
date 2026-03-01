"""Single scene preview generation (model inference).

Copied from anime-video-gen/scripts/lib/inference.py for standalone worker use.
"""

import inspect
import logging
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image

from .config import validate_generation_params
from .hardware import VRAMManager, get_optimal_vae_batch_size

logger = logging.getLogger(__name__)


def decode_latents_with_progress(pipeline, latents, num_frames, progress_callback=None):
    """Decode latents in batches with progress reporting.

    This avoids OOM on smaller GPUs by decoding frames in chunks.
    """
    vae = pipeline.vae

    # Determine batch size based on available VRAM
    if torch.cuda.is_available():
        free_vram_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
        # Rough estimate: each frame decode needs ~0.5GB for 832x480
        # Be conservative to avoid OOM
        if free_vram_gb > 12:
            batch_size = 16  # Plenty of VRAM
        elif free_vram_gb > 8:
            batch_size = 8
        elif free_vram_gb > 4:
            batch_size = 4
        else:
            batch_size = 2  # Very low VRAM
        logger.info(f"  VAE decode: {free_vram_gb:.1f}GB free, using batch_size={batch_size}")
    else:
        batch_size = 4

    # Check if VAE has temporal dimension handling
    # Wan VAE expects: (B, C, T, H, W)
    is_video_vae = hasattr(vae.config, 'temporal_compression_ratio') or 'Wan' in type(vae).__name__

    if is_video_vae:
        # Video VAE - decode all at once but with memory optimization
        logger.info(f"  VAE decode: Video VAE detected, decoding {num_frames} frames...")
        start_time = time.time()

        # Move VAE to GPU if not already
        vae_device = next(vae.parameters()).device
        latents = latents.to(vae_device, dtype=vae.dtype)

        with torch.no_grad():
            # Try to decode - if OOM, we'll need tiling
            try:
                if progress_callback:
                    progress_callback(0, "VAE decode starting...")

                decoded = vae.decode(latents).sample

                if progress_callback:
                    progress_callback(100, "VAE decode complete")

            except torch.cuda.OutOfMemoryError:
                logger.warning("  VAE OOM - trying with chunked temporal decode...")
                torch.cuda.empty_cache()

                # Decode in temporal chunks
                # latents shape: (B, C, T, H, W)
                T = latents.shape[2]
                chunk_size = max(1, T // 4)  # Quarter at a time
                decoded_chunks = []

                for i in range(0, T, chunk_size):
                    chunk_end = min(i + chunk_size, T)
                    chunk = latents[:, :, i:chunk_end, :, :]

                    if progress_callback:
                        pct = int((i / T) * 100)
                        progress_callback(pct, f"VAE decode chunk {i//chunk_size + 1}")

                    with torch.no_grad():
                        decoded_chunk = vae.decode(chunk).sample
                        decoded_chunks.append(decoded_chunk.cpu())

                    torch.cuda.empty_cache()

                decoded = torch.cat(decoded_chunks, dim=2).to(vae_device)

                if progress_callback:
                    progress_callback(100, "VAE decode complete")

        elapsed = time.time() - start_time
        logger.info(f"  VAE decode complete in {elapsed:.1f}s")

        # Convert to frames list
        # decoded shape: (B, C, T, H, W) -> list of (H, W, C) numpy arrays
        decoded = decoded.squeeze(0)  # Remove batch dim: (C, T, H, W)
        decoded = decoded.permute(1, 2, 3, 0)  # (T, H, W, C)
        decoded = ((decoded + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        frames = [decoded[i].cpu().numpy() for i in range(decoded.shape[0])]

        return frames

    else:
        # Image VAE - decode frame by frame
        logger.info(f"  VAE decode: Image VAE, decoding {num_frames} frames in batches of {batch_size}...")

        frames = []
        total_batches = (num_frames + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_frames)

            batch_latents = latents[:, :, start_idx:end_idx]

            with torch.no_grad():
                decoded_batch = vae.decode(batch_latents).sample

            # Convert to frames
            for i in range(decoded_batch.shape[2]):
                frame = decoded_batch[0, :, i].permute(1, 2, 0)
                frame = ((frame + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
                frames.append(frame.cpu().numpy())

            # Report progress
            pct = int(((batch_idx + 1) / total_batches) * 100)
            if progress_callback:
                progress_callback(pct, f"VAE decode {end_idx}/{num_frames} frames")
            logger.info(f"  VAE progress: {pct}% ({end_idx}/{num_frames} frames)")

            # Clear cache between batches
            torch.cuda.empty_cache()

        return frames


def generate_preview_frame(
    prompt,
    negative_prompt,
    config,
    output_path,
    scene_id,
    seed_manager,
    pipeline,
    previous_frame_path=None,
    output_base=None,
    model_config=None,
    progress_callback=None,  # Callback for VAE progress: (pct, message) -> None
):
    """Generate a single preview clip for a scene."""
    seed = seed_manager.get_or_create_seed(scene_id)
    logger.info(f"Generating preview for scene {scene_id} with seed {seed}")

    full_prompt = seed_manager.build_full_prompt(scene_id)
    logger.info(f"Full prompt: {full_prompt[:100]}...")

    preview_config = config["generation"]["preview"]

    logger.info("Preview settings:")
    logger.info(f"  Resolution: {preview_config['width']}x{preview_config['height']}")
    logger.info(f"  FPS: {preview_config.get('fps', 24)}")
    logger.info(f"  Frames: {preview_config['num_frames']}")
    logger.info(f"  Steps: {preview_config['num_inference_steps']}")
    logger.info(f"  Seed: {seed}")

    if previous_frame_path:
        logger.info(f"  Using last frame from previous scene: {previous_frame_path}")

    width, height, num_frames = validate_generation_params(
        preview_config["width"],
        preview_config["height"],
        preview_config["num_frames"],
        model_config,
    )
    if (
        width != preview_config["width"]
        or height != preview_config["height"]
        or num_frames != preview_config["num_frames"]
    ):
        logger.info(
            f"  Adjusted: {preview_config['width']}x{preview_config['height']}x{preview_config['num_frames']} -> {width}x{height}x{num_frames}"
        )

    torch.cuda.empty_cache()

    # Enable VAE tiling to reduce peak memory usage
    if hasattr(pipeline, 'enable_vae_tiling'):
        pipeline.enable_vae_tiling()
        logger.info("  VAE tiling enabled (reduces memory usage)")

    # Enable sliced attention for further memory savings
    if hasattr(pipeline, 'enable_vae_slicing'):
        pipeline.enable_vae_slicing()
        logger.info("  VAE slicing enabled")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Progress callback for real-time updates during diffusion
    total_steps = preview_config["num_inference_steps"]

    def diffusion_step_callback(step, timestep, latents):
        pct = int((step / total_steps) * 100)
        if step % max(1, total_steps // 10) == 0:  # Log every 10%
            logger.info(
                f"  Generation progress: {pct}% (step {step}/{total_steps})"
            )
        # Report to parent via progress_callback if provided
        if progress_callback:
            # Diffusion is 0-80% of the process, VAE decode is 80-100%
            scaled_pct = pct * 0.8
            progress_callback(scaled_pct, f"Diffusion step {step}/{total_steps}")

    pipeline_kwargs = {
        "prompt": full_prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "num_inference_steps": preview_config["num_inference_steps"],
        "guidance_scale": preview_config["guidance_scale"],
        "generator": generator,
        "callback": diffusion_step_callback,
        "callback_steps": 1,
    }

    if preview_config.get("guidance_scale_2") is not None:
        pipeline_kwargs["guidance_scale_2"] = preview_config["guidance_scale_2"]
    mc = model_config or {}
    if mc.get("flow_shift") is not None:
        pipeline_kwargs["flow_shift"] = mc["flow_shift"]
    if mc.get("flow_reverse"):
        pipeline_kwargs["flow_reverse"] = True

    if previous_frame_path and Path(previous_frame_path).exists():
        try:
            input_image = Image.open(previous_frame_path)
            pipeline_kwargs["image"] = input_image
            logger.info("  Image conditioning enabled (last-frame continuity)")
        except Exception as e:
            logger.warning(f"Could not load previous frame, using text-only: {e}")

    sig = inspect.signature(pipeline.__call__)
    accepted = set(sig.parameters.keys())
    if accepted and "**" not in str(sig):
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_keyword:
            unsupported = [k for k in list(pipeline_kwargs.keys()) if k not in accepted]
            for k in unsupported:
                logger.warning(f"  Dropping unsupported pipeline kwarg: {k}")
                del pipeline_kwargs[k]

    logger.info("  Starting diffusion inference...")
    start_time = time.time()

    # Log VRAM before
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        logger.info(f"  VRAM before inference: {free/(1024**3):.1f}GB free / {total/(1024**3):.1f}GB total")

    # Try to get latents separately so we can decode with progress
    # This gives us control over VAE decode for progress reporting
    use_custom_vae_decode = True

    if use_custom_vae_decode:
        # Request latent output instead of decoded frames
        pipeline_kwargs["output_type"] = "latent"

    # Run the diffusion pipeline
    output = pipeline(**pipeline_kwargs)

    diffusion_time = time.time() - start_time
    logger.info(f"  Diffusion complete in {diffusion_time:.1f}s")

    # Log VRAM after diffusion
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        logger.info(f"  VRAM after diffusion: {free/(1024**3):.1f}GB free / {total/(1024**3):.1f}GB total")

    # CRITICAL: Offload model to CPU before VAE decode to free VRAM
    # This is essential for 16GB cards where model + VAE don't fit together
    vram_manager = VRAMManager(pipeline, min_free_vram_gb=6.0)
    vram_manager.offload_for_vae(progress_callback=progress_callback)

    # Decode VAE with progress
    if use_custom_vae_decode and hasattr(output, "frames") and output.frames is not None:
        latents = output.frames[0] if isinstance(output.frames, list) else output.frames
        logger.info(f"  Got latents shape: {latents.shape if hasattr(latents, 'shape') else 'unknown'}")

        # Progress callback that logs and reports to parent
        def vae_progress(pct, msg):
            logger.info(f"  VAE: {pct}% - {msg}")
            if progress_callback:
                # Scale VAE progress to 80-95% of overall progress
                scaled_pct = 80 + (pct * 0.15)
                progress_callback(scaled_pct, f"VAE: {msg}")

        frames = decode_latents_with_progress(
            pipeline,
            latents,
            num_frames,
            progress_callback=vae_progress
        )
        logger.info(f"  Got {len(frames)} decoded frames")
    else:
        # Fallback: pipeline already decoded
        logger.info("  Extracting frames from pipeline output...")
        frames = output.frames[0] if hasattr(output, "frames") else output.images
        logger.info(f"  Got {len(frames) if hasattr(frames, '__len__') else 'unknown'} frames")

    # Log VRAM after VAE
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        logger.info(f"  VRAM after VAE decode: {free/(1024**3):.1f}GB free / {total/(1024**3):.1f}GB total")

    video_path = output_path.replace(".png", ".mp4")
    export_to_video(frames, video_path, fps=preview_config["fps"])
    logger.info(f"Preview video saved to: {video_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("  GPU cache cleared")

    if output_base:
        lastframe_path = Path(output_base) / f"scene_{scene_id:03d}_lastframe.png"
        try:
            last_frame = frames[-1]
            if not isinstance(last_frame, np.ndarray):
                last_frame = np.array(last_frame)
            if last_frame.dtype != np.uint8:
                if last_frame.max() <= 1.0:
                    last_frame = (last_frame * 255).astype(np.uint8)
                else:
                    last_frame = last_frame.astype(np.uint8)
            last_frame_img = Image.fromarray(last_frame)
            last_frame_img.save(str(lastframe_path))
            logger.info(f"  Last frame saved for Scene {scene_id + 1}")
            return True, str(lastframe_path)
        except Exception as e:
            logger.warning(f"  Could not save last frame for continuity: {e}")

    return True, None
