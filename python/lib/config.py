"""Simplified config for worker: receives config from JSON, no disk loading."""

import logging

logger = logging.getLogger(__name__)


def validate_generation_params(width, height, num_frames, model_config):
    """Validate and snap width/height/num_frames to valid values per model."""
    model_class = (model_config or {}).get('model_class', 'WanPipeline')

    if 'LTX' in model_class:
        alignment = 32
    else:
        alignment = 16

    width = max(alignment, (width // alignment) * alignment)
    height = max(alignment, (height // alignment) * alignment)

    if 'Wan' in model_class:
        k = max(1, round((num_frames - 1) / 4))
        num_frames = 4 * k + 1
    elif 'LTX' in model_class:
        k = max(1, round((num_frames - 1) / 8))
        num_frames = 8 * k + 1
    elif 'Hunyuan' in model_class:
        k = max(1, round((num_frames - 1) / 4))
        num_frames = 4 * k + 1

    return width, height, num_frames
