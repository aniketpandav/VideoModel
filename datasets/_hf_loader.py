"""Utility to import the real HuggingFace `datasets` library.

Our local package folder is also called `datasets/`, which shadows the
installed `huggingface/datasets` library on sys.path. This module resolves
the real one from site-packages and exposes it under its canonical package
name once per process so lazy streaming imports like `datasets.features`
continue to work.

Usage inside any file in this project:

    from datasets._hf_loader import load_hf_dataset
    ds = load_hf_dataset("zengxianyu/open-sora-pexels-subset", split="train", streaming=True)
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def _find_hf_datasets_path() -> Path:
    """Locate the installed HuggingFace datasets package directory."""
    project_root = Path(__file__).parent.parent.resolve()

    for p in sys.path:
        try:
            rel = Path(p or ".").resolve()
        except Exception:
            continue

        if rel == project_root:
            continue

        candidate = rel / "datasets" / "__init__.py"
        if candidate.exists():
            return rel

    raise ImportError(
        "HuggingFace 'datasets' library not found in site-packages.\n"
        "Install it with:  pip install datasets"
    )


_HF_DATASETS_PATH = _find_hf_datasets_path()
_HF_DATASETS: ModuleType | None = None


def _patch_multiprocess_resource_tracker() -> None:
    """Silence a Python 3.12 incompatibility in multiprocess ResourceTracker.

    multiprocess 0.70.19 assumes RLock has a private `_recursion_count()`
    method. Some Windows Python 3.12 builds expose an RLock without it, which
    causes an ignored exception during interpreter shutdown after importing
    HuggingFace datasets.
    """
    try:
        import multiprocess.resource_tracker as resource_tracker
    except Exception:
        return

    tracker_cls = getattr(resource_tracker, "ResourceTracker", None)
    if tracker_cls is None or getattr(tracker_cls, "_videogen_py312_patch", False):
        return

    def _lock_recursion_count(lock) -> int:
        recursion_count = getattr(lock, "_recursion_count", None)
        if recursion_count is None:
            return 0
        try:
            return recursion_count()
        except Exception:
            return 0

    def _stop_locked(self, close=os.close, waitpid=getattr(os, "waitpid", None)):
        if _lock_recursion_count(self._lock) > 1:
            return self._reentrant_call_error()
        if self._fd is None:
            return
        if self._pid is None:
            return

        close(self._fd)
        self._fd = None

        if waitpid is not None:
            waitpid(self._pid, 0)
        self._pid = None

    tracker_cls._stop_locked = _stop_locked
    tracker_cls._videogen_py312_patch = True


def _hf_sys_path() -> list[str]:
    """Put site-packages first and remove the project root during HF import."""
    hf_path = str(_HF_DATASETS_PATH)
    project_root = Path(__file__).parent.parent.resolve()
    return [hf_path] + [
        p for p in sys.path
        if p != hf_path and Path(p or ".").resolve() != project_root
    ]


def _is_hf_datasets_module(module: ModuleType | None) -> bool:
    if module is None or not hasattr(module, "load_dataset"):
        return False
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        return Path(module_file).resolve().is_relative_to(_HF_DATASETS_PATH)
    except Exception:
        return False


def _ensure_hf_datasets() -> ModuleType:
    """Import HuggingFace datasets once as the canonical `datasets` package."""
    global _HF_DATASETS

    current = sys.modules.get("datasets")
    if _is_hf_datasets_module(current):
        _HF_DATASETS = current
        _patch_multiprocess_resource_tracker()
        return current

    if _is_hf_datasets_module(_HF_DATASETS):
        sys.modules["datasets"] = _HF_DATASETS
        _patch_multiprocess_resource_tracker()
        return _HF_DATASETS

    old_path = list(sys.path)
    for name in [
        name for name in list(sys.modules)
        if name == "datasets" or name.startswith("datasets.")
    ]:
        sys.modules.pop(name, None)

    try:
        sys.path = _hf_sys_path()
        module = importlib.import_module("datasets")
    finally:
        sys.path = old_path

    if not hasattr(module, "load_dataset"):
        raise ImportError(
            "Installed package named 'datasets' does not expose load_dataset()."
        )

    _HF_DATASETS = module
    _patch_multiprocess_resource_tracker()
    return module


def get_hf_datasets() -> ModuleType:
    """Return the real HuggingFace datasets module."""
    return _ensure_hf_datasets()


def load_hf_dataset(*args, **kwargs):
    """Thin wrapper around huggingface datasets.load_dataset().

    Recent Hugging Face Datasets releases no longer execute Hub dataset
    loading scripts via ``trust_remote_code``. Keep this wrapper explicit so
    callers do not accidentally rely on remote Python code, and turn the
    resulting error into a message that points at standard-format datasets.
    """
    kwargs.pop("trust_remote_code", None)
    hf = _ensure_hf_datasets()
    try:
        return hf.load_dataset(*args, **kwargs)
    except Exception as exc:
        message = str(exc)
        if "trust_remote_code" in message and "not supported" in message:
            dataset_name = args[0] if args else kwargs.get("path", "<unknown>")
            raise RuntimeError(
                f"Hugging Face dataset {dataset_name!r} requires a dataset "
                "loading script, but this Datasets version no longer supports "
                "trust_remote_code. Use a standard dataset source such as "
                "Parquet/CSV/JSON/WebDataset/video files, or remove this "
                "source from the online training config."
            ) from exc
        raise
