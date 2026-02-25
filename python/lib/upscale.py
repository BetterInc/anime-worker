"""Video upscaling using Real-ESRGAN.

Upscales videos from 720p to 1080p (or custom resolutions) using the
Real-ESRGAN AnimeVideo-v3 model via frame extraction and reassembly.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)


def check_dependencies() -> Tuple[bool, str]:
    """Check if required dependencies (ffmpeg, realesrgan-ncnn-vulkan) are available."""
    missing = []

    # Check ffmpeg
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")

    # Check ffprobe
    if not shutil.which("ffprobe"):
        missing.append("ffprobe")

    # Check Real-ESRGAN (optional - we can fall back to Python package)
    has_realesrgan_ncnn = shutil.which("realesrgan-ncnn-vulkan") is not None

    if missing:
        return False, f"Missing dependencies: {', '.join(missing)}"

    return True, "realesrgan-ncnn-vulkan" if has_realesrgan_ncnn else "python-fallback"


def upscale_video(
    input_path: Path,
    output_path: Path,
    scale: int = 2,
    model_name: str = "realesr-animevideov3",
    fps: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """Upscale a video using Real-ESRGAN.

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        scale: Upscale factor (default: 2 for 720p->1080p)
        model_name: Real-ESRGAN model to use
        fps: Output FPS (None = same as input)
        progress_callback: Optional callback(progress_pct, message)

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Upscaling video: {input_path.name}")
    logger.info(f"  Scale: {scale}x")
    logger.info(f"  Model: {model_name}")

    # Check dependencies
    deps_ok, deps_info = check_dependencies()
    if not deps_ok:
        logger.error(f"Dependency check failed: {deps_info}")
        return False

    logger.info(f"  Method: {deps_info}")

    # Create temp directory for frames
    temp_dir = Path(tempfile.mkdtemp(prefix="upscale_"))
    frames_input = temp_dir / "input"
    frames_output = temp_dir / "output"
    frames_input.mkdir(exist_ok=True)
    frames_output.mkdir(exist_ok=True)

    try:
        if progress_callback:
            progress_callback(5, "Extracting frames...")

        # Step 1: Extract frames from input video
        logger.info("Step 1/3: Extracting frames...")
        extract_cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-qscale:v",
            "1",  # High quality
            "-vsync",
            "0",  # Preserve frame timing
            str(frames_input / "frame_%05d.png"),
        ]

        result = subprocess.run(
            extract_cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error(f"Frame extraction failed: {result.stderr}")
            return False

        # Count extracted frames
        frame_files = sorted(frames_input.glob("frame_*.png"))
        total_frames = len(frame_files)
        logger.info(f"  Extracted {total_frames} frames")

        if total_frames == 0:
            logger.error("No frames extracted")
            return False

        if progress_callback:
            progress_callback(15, f"Upscaling {total_frames} frames...")

        # Step 2: Upscale frames with Real-ESRGAN
        logger.info("Step 2/3: Upscaling frames...")

        if deps_info == "realesrgan-ncnn-vulkan":
            # Use NCNN version (faster, multi-GPU)
            upscale_cmd = [
                "realesrgan-ncnn-vulkan",
                "-i",
                str(frames_input),
                "-o",
                str(frames_output),
                "-n",
                model_name,
                "-s",
                str(scale),
                "-f",
                "png",
            ]

            result = subprocess.run(
                upscale_cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
            )

            if result.returncode != 0:
                logger.error(f"Upscaling failed: {result.stderr}")
                return False
        else:
            # Fallback: Use Python Real-ESRGAN (if available)
            logger.warning("realesrgan-ncnn-vulkan not found, trying Python fallback")
            success = _upscale_with_python(
                frames_input,
                frames_output,
                scale,
                model_name,
                total_frames,
                progress_callback,
            )
            if not success:
                return False

        # Verify upscaled frames
        upscaled_files = sorted(frames_output.glob("frame_*.png"))
        if len(upscaled_files) != total_frames:
            logger.error(
                f"Frame count mismatch: expected {total_frames}, got {len(upscaled_files)}"
            )
            return False

        logger.info(f"  Upscaled {len(upscaled_files)} frames")

        if progress_callback:
            progress_callback(85, "Reassembling video...")

        # Step 3: Reassemble video from upscaled frames
        logger.info("Step 3/3: Reassembling video...")

        # Get FPS from input video if not specified
        if fps is None:
            fps = _get_video_fps(input_path)

        reassemble_cmd = [
            "ffmpeg",
            "-framerate",
            str(fps),
            "-i",
            str(frames_output / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",  # High quality
            "-preset",
            "slow",
            "-y",  # Overwrite output
            str(output_path),
        ]

        result = subprocess.run(
            reassemble_cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            logger.error(f"Video reassembly failed: {result.stderr}")
            return False

        logger.info(f"Upscaled video saved to: {output_path}")

        if progress_callback:
            progress_callback(100, "Upscaling complete")

        return True

    except subprocess.TimeoutExpired:
        logger.error("Upscaling timed out")
        return False
    except Exception as e:
        logger.exception(f"Upscaling failed: {e}")
        return False
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary frames")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir: {e}")


def _upscale_with_python(
    frames_input: Path,
    frames_output: Path,
    scale: int,
    model_name: str,
    total_frames: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """Fallback upscaling using Python Real-ESRGAN package."""
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        logger.info("Using Python Real-ESRGAN package")

        # Initialize model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=scale,
        )
        upsampler = RealESRGANer(
            scale=scale,
            model_path=f"weights/{model_name}.pth",
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,  # Use FP16 for speed
        )

        # Process frames
        frame_files = sorted(frames_input.glob("frame_*.png"))
        for i, frame_file in enumerate(frame_files, 1):
            output_file = frames_output / frame_file.name

            from PIL import Image
            import numpy as np

            # Read, upscale, save
            img = Image.open(frame_file)
            img_array = np.array(img)

            output_array, _ = upsampler.enhance(img_array, outscale=scale)

            output_img = Image.fromarray(output_array)
            output_img.save(output_file)

            # Progress update
            if progress_callback and i % 10 == 0:
                pct = 15 + (70 * i / total_frames)
                progress_callback(pct, f"Upscaling frame {i}/{total_frames}")

        return True

    except ImportError:
        logger.error(
            "Python Real-ESRGAN package not installed. Install with: pip install realesrgan"
        )
        return False
    except Exception as e:
        logger.exception(f"Python upscaling failed: {e}")
        return False


def _get_video_fps(video_path: Path) -> int:
    """Get video FPS using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and result.stdout.strip():
            # Parse fraction (e.g., "24/1" or "30000/1001")
            parts = result.stdout.strip().split("/")
            if len(parts) == 2:
                fps = int(float(parts[0]) / float(parts[1]))
                logger.info(f"Detected FPS: {fps}")
                return fps

    except Exception as e:
        logger.warning(f"Could not detect FPS: {e}")

    # Default fallback
    logger.info("Using default FPS: 24")
    return 24
