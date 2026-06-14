"""Serving layer for the video platform.

Dual-track by design (see BLUEPRINT.md):
  ToyBackbone    → serves the from-scratch 3D U-Net checkpoint; runs on the local
                   GTX 1650. This is the milestone/demo engine.
  LTXVideoBackbone → wraps a pretrained LTX-Video model for production-quality T2V
                   and I2V; requires a cloud GPU (8 GB+).
  CogVideoXBackbone → wraps CogVideoX-5b-I2V, specialist for Mode 2 (image→video).

Both Track A backbones expose the same Backbone.generate() / generate_i2v()
interface so the API and pipeline never change when you swap one for the other.

New production pipeline layers (all injectable / swappable):
  PromptEngine   → intent detection, safety, enhancement, narrative expansion
  VideoPipeline  → orchestrates Mode 1 / Mode 2, routes to chunker for long video
  LongVideoChunker → chunk-and-stitch for videos beyond 30 seconds
  QualityValidator → validates output frames before delivery
  StorageManager → job metadata (SQLite) + file storage (local / S3)
  AsyncJobQueue  → single-worker async queue (dev) / Celery (prod)
"""
from .backbones import (
    Backbone,
    BackboneFactory,
    CogVideoXBackbone,
    LTXVideoBackbone,
    ToyBackbone,
    get_backbone,
)
from .chunker import LongVideoChunker
from .pipeline import VideoPipeline
from .prompt_engine import EnhancedRequest, PromptEngine, PromptValidationError, SafetyError
from .quality import QualityValidator, ValidationResult
from .queue import AsyncJobQueue
from .render import encode_video, finish_clip, make_preview_gif, save_clip, to_uint8_clip
from .storage import FileStorage, JobRecord, JobStore, StorageManager

__all__ = [
    # Backbones
    "Backbone", "BackboneFactory", "ToyBackbone", "LTXVideoBackbone",
    "CogVideoXBackbone", "get_backbone",
    # Pipeline
    "VideoPipeline", "LongVideoChunker",
    # Prompt
    "PromptEngine", "EnhancedRequest", "PromptValidationError", "SafetyError",
    # Quality
    "QualityValidator", "ValidationResult",
    # Queue
    "AsyncJobQueue",
    # Render
    "encode_video", "finish_clip", "make_preview_gif", "save_clip", "to_uint8_clip",
    # Storage
    "StorageManager", "JobStore", "FileStorage", "JobRecord",
]
