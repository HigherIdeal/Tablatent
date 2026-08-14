"""Stage 1 public API.

The previous deterministic autoencoder pipeline has been replaced by the VAE
pipeline. Keep this module as the stable import path used by scripts.
"""

from .pipeline_vae import evaluate_stage1, train_stage1

__all__ = ["train_stage1", "evaluate_stage1"]
