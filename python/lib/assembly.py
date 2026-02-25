"""Video assembly using FFmpeg concat.

Concatenates multiple video clips into a single output file.
"""

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def check_dependencies() -> Tuple[bool, str]:
    """Check if required dependencies (ffmpeg, ffprobe) are available."""
    missing = []

    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")

    if not shutil.which("ffprobe"):
        missing.append("ffprobe")

    if missing:
        return False, f"Missing dependencies: {', '.join(missing)}"

    return True, "ok"


def assemble_videos(
    input_files: List[Path],
    output_path: Path,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "slow",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """Assemble multiple video clips into a single file.

    Args:
        input_files: List of video files to concatenate (in order)
        output_path: Path to output video file
        codec: Video codec (default: libx264)
        crf: Constant Rate Factor for quality (default: 18, high quality)
        preset: Encoding preset (default: slow)
        progress_callback: Optional callback(progress_pct, message)

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Assembling {len(input_files)} video clips")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Codec: {codec}, CRF: {crf}, Preset: {preset}")

    # Check dependencies
    deps_ok, deps_info = check_dependencies()
    if not deps_ok:
        logger.error(f"Dependency check failed: {deps_info}")
        return False

    # Validate inputs
    if not input_files:
        logger.error("No input files provided")
        return False

    for i, video_file in enumerate(input_files):
        if not video_file.exists():
            logger.error(f"Input file not found: {video_file}")
            return False
        logger.info(f"  [{i + 1}] {video_file.name}")

    if progress_callback:
        progress_callback(5, "Analyzing input videos...")

    # Get metadata for progress estimation
    total_duration = 0.0
    for video_file in input_files:
        duration = _get_video_duration(video_file)
        if duration:
            total_duration += duration

    logger.info(
        f"Total duration: {total_duration:.1f}s ({total_duration / 60:.1f} minutes)"
    )

    # Create temporary concat file
    concat_file = Path(tempfile.mktemp(suffix=".txt", prefix="concat_"))

    try:
        if progress_callback:
            progress_callback(10, "Creating concat list...")

        # Create FFmpeg concat demuxer file
        _create_concat_file(input_files, concat_file)

        if progress_callback:
            progress_callback(15, "Concatenating videos...")

        # Build FFmpeg command using concat demuxer (lossless concatenation)
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-y",  # Overwrite output
            str(output_path),
        ]

        logger.info("Running FFmpeg concat...")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Run FFmpeg with progress monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Monitor progress via FFmpeg output
        last_progress = 15
        for line in process.stdout:
            # FFmpeg progress line format: "frame=  123 fps= 45 q=28.0 size=    1024kB time=00:00:05.12 ..."
            if "time=" in line and progress_callback:
                try:
                    time_str = line.split("time=")[1].split()[0]
                    # Parse time (HH:MM:SS.MS)
                    h, m, s = time_str.split(":")
                    current_seconds = int(h) * 3600 + int(m) * 60 + float(s)

                    if total_duration > 0:
                        progress = 15 + (80 * current_seconds / total_duration)
                        progress = min(95, max(last_progress, progress))

                        if progress > last_progress + 5:  # Update every 5%
                            progress_callback(
                                progress,
                                f"Encoding... {current_seconds:.0f}/{total_duration:.0f}s",
                            )
                            last_progress = progress
                except Exception:
                    pass  # Ignore parsing errors

        process.wait()

        if process.returncode != 0:
            logger.error("FFmpeg concat failed")
            return False

        logger.info(f"Assembled video saved to: {output_path}")

        if progress_callback:
            progress_callback(100, "Assembly complete")

        return True

    except subprocess.TimeoutExpired:
        logger.error("Assembly timed out")
        return False
    except Exception as e:
        logger.exception(f"Assembly failed: {e}")
        return False
    finally:
        # Cleanup concat file
        try:
            if concat_file.exists():
                concat_file.unlink()
                logger.debug("Cleaned up concat file")
        except Exception as e:
            logger.warning(f"Failed to cleanup concat file: {e}")


def _create_concat_file(video_files: List[Path], concat_path: Path) -> None:
    """Create FFmpeg concat demuxer file.

    Format:
        file '/absolute/path/to/video1.mp4'
        file '/absolute/path/to/video2.mp4'
    """
    with open(concat_path, "w") as f:
        for video in video_files:
            # FFmpeg concat format requires absolute paths
            abs_path = video.absolute()
            # Escape single quotes in path
            escaped_path = str(abs_path).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")

    logger.debug(f"Created concat file: {concat_path}")


def _get_video_duration(video_path: Path) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data.get("format", {}).get("duration", 0))
            return duration

    except Exception as e:
        logger.warning(f"Could not get duration for {video_path.name}: {e}")

    return None


def get_video_info(video_path: Path) -> Optional[dict]:
    """Get comprehensive video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(video_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            return json.loads(result.stdout)

    except Exception as e:
        logger.warning(f"Could not get video info for {video_path.name}: {e}")

    return None
