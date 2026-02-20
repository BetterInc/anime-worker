#!/usr/bin/env python3
"""
Entry point for GPU inference, launched by the Rust worker.

Receives job config as a single JSON line on stdin.
Outputs JSON progress/completion lines on stdout.
Logs to stderr.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Setup logging to stderr (stdout is reserved for JSON protocol)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("inference_runner")


def emit(msg: dict):
    """Write a JSON message to stdout (protocol with Rust)."""
    print(json.dumps(msg), flush=True)


def main():
    # Read job config from stdin
    raw = sys.stdin.readline()
    if not raw.strip():
        emit({"type": "error", "message": "No input received on stdin"})
        sys.exit(1)

    try:
        job = json.loads(raw)
    except json.JSONDecodeError as e:
        emit({"type": "error", "message": f"Invalid JSON on stdin: {e}"})
        sys.exit(1)

    task_id = job["task_id"]
    task_type = job["task_type"]
    scene = job["scene"]
    project = job["project"]
    model_path = job["model_path"]
    model_config = job.get("model_config", {})
    pipeline_config = job.get("pipeline_config", {})
    output_dir = Path(job["output_dir"])
    last_frame_path = job.get("last_frame_path")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Task {task_id[:8]}: {task_type} for scene {scene.get('id', '?')}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_dir}")

    emit({"type": "progress", "pct": 0, "message": "Loading model..."})

    try:
        from lib.pipeline import setup_pipeline
        from lib.inference import generate_preview_frame
        from lib.config import validate_generation_params

        # Build config dict that setup_pipeline expects
        config = {
            "model": {
                "base_path": str(Path(model_path).parent),
                "wan22": model_config,
            },
            "generation": {
                "preview": pipeline_config,
                "final": pipeline_config,
            },
        }

        # Override local_dir to point at our model
        mc = dict(model_config)
        mc["local_dir"] = Path(model_path).name

        emit({"type": "progress", "pct": 5, "message": "Loading pipeline..."})
        pipe = setup_pipeline(config, model_config=mc)

        emit({"type": "progress", "pct": 15, "message": "Starting generation..."})

        # Build a simple seed manager that just returns the scene's seed
        scene_id = scene.get("id", 0)
        seed = scene.get("seed")

        class SimpleSeedManager:
            def get_or_create_seed(self, sid):
                import random
                return seed if seed else random.randint(0, 2**32 - 1)

            def build_full_prompt(self, sid):
                parts = []
                style = project.get("style", "")
                if style:
                    parts.append(style)
                story = project.get("story_context", "")
                if story:
                    parts.append(story)
                sc = scene.get("scene_context", "")
                if sc:
                    parts.append(sc)
                parts.append(scene.get("prompt", ""))
                return ", ".join(p for p in parts if p)

        seed_mgr = SimpleSeedManager()

        output_filename = f"scene_{scene_id:03d}_{task_type}.mp4"
        output_path = str(output_dir / output_filename)

        # Hook into tqdm to report progress
        _original_tqdm = None
        try:
            import tqdm
            _original_tqdm = tqdm.tqdm

            class ProgressTqdm(tqdm.tqdm):
                def update(self, n=1):
                    super().update(n)
                    if self.total and self.total > 0:
                        pct = (self.n / self.total) * 100
                        emit({
                            "type": "progress",
                            "pct": pct,
                            "message": f"Step {self.n}/{self.total}",
                        })

            tqdm.tqdm = ProgressTqdm
            # Also patch tqdm.auto
            if hasattr(tqdm, 'auto'):
                tqdm.auto.tqdm = ProgressTqdm
        except ImportError:
            pass

        success, lastframe_path = generate_preview_frame(
            prompt=scene.get("prompt", ""),
            negative_prompt=scene.get("negative_prompt", ""),
            config=config,
            output_path=output_path,
            scene_id=scene_id,
            seed_manager=seed_mgr,
            pipeline=pipe,
            previous_frame_path=last_frame_path,
            output_base=str(output_dir),
            model_config=mc,
        )

        # Restore tqdm
        if _original_tqdm:
            import tqdm
            tqdm.tqdm = _original_tqdm

        if not success:
            emit({"type": "error", "message": "Generation returned failure"})
            sys.exit(1)

        # Collect metadata
        actual_seed = seed_mgr.get_or_create_seed(scene_id)
        files = [output_filename]
        metadata = {
            "scene_id": scene_id,
            "seed": actual_seed,
            "width": pipeline_config.get("width"),
            "height": pipeline_config.get("height"),
            "fps": pipeline_config.get("fps", 24),
        }

        # Get file size
        output_file = output_dir / output_filename
        if output_file.exists():
            metadata["file_size"] = output_file.stat().st_size

        emit({
            "type": "complete",
            "files": files,
            "metadata": metadata,
        })

    except Exception as e:
        logger.exception("Inference failed")
        emit({"type": "error", "message": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
