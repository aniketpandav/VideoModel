"""Serving layer for the video platform.

Dual-track by design (see BLUEPRINT.md):
  - ToyBackbone  -> serves the from-scratch 3D U-Net checkpoint; runs on the local
                    GTX 1650. This is the milestone/demo engine.
  - LTXBackbone  -> wraps a pretrained LTX-Video model for production-quality video;
                    requires a cloud GPU (8 GB+). This is the engine the website ships.

Both expose the same `Backbone.generate()` interface so the API and website never
change when you swap one for the other.
"""
from .backbones import Backbone, ToyBackbone, LTXBackbone, get_backbone

__all__ = ["Backbone", "ToyBackbone", "LTXBackbone", "get_backbone"]
