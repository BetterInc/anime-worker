"""Single scene preview generation (model inference).

Copied from anime-video-gen/scripts/lib/inference.py for standalone worker use.
"""

import inspect
import logging
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image

from .config import validate_generation_params

logger = logging.getLogger(__name__)


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

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Progress callback for real-time updates
    total_steps = preview_config["num_inference_steps"]

    def progress_callback(step, timestep, latents):
        progress = int((step / total_steps) * 100)
        if step % max(1, total_steps // 10) == 0:  # Log every 10%
            logger.info(
                f"  Generation progress: {progress}% (step {step}/{total_steps})"
            )

    pipeline_kwargs = {
        "prompt": full_prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "num_inference_steps": preview_config["num_inference_steps"],
        "guidance_scale": preview_config["guidance_scale"],
        "generator": generator,
        "callback": progress_callback,
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

    output = pipeline(**pipeline_kwargs)

    frames = output.frames[0] if hasattr(output, "frames") else output.images

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
