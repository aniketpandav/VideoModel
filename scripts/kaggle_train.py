"""Kaggle / cloud training entry point with HuggingFace Hub checkpoint sync.

Wraps the regular training pipeline so a 12-hour Kaggle session can:

  1. Pull existing checkpoints from an HF Hub repo (resume from previous session).
  2. Run training normally. Trainer auto-resumes from the latest local checkpoint.
  3. Push checkpoints back to HF Hub every N minutes AND on exit.

If anything kills the session (timeout, OOM, crash, manual stop), the most
recent periodic push is preserved on the Hub. Next session, pull → resume.

Usage (inside a Kaggle notebook cell):

    !python scripts/kaggle_train.py --mode dit \\
        --hf_repo your-username/video-model-ckpts \\
        --hf_token $HF_TOKEN \\
        --sync_every_minutes 10

Tokens are read from --hf_token, then $HF_TOKEN, then ~/.huggingface/token.
On Kaggle, put your token in "Add-ons -> Secrets" as HF_TOKEN.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hf_sync import (  # noqa: E402
    PeriodicSync, pull_checkpoints, push_checkpoints, verify_writable,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kaggle-friendly training wrapper with HF Hub checkpoint sync.")
    p.add_argument("--mode", choices=["vae", "dit", "lora"], default="dit")
    p.add_argument("--model_config", default="configs/model/dit_micro.yaml")
    p.add_argument("--train_config", default=None,
                   help="Defaults to configs/training/train_<mode>.yaml")
    p.add_argument("--online", action="store_true", default=True,
                   help="Use online streaming dataset (default ON for cloud).")
    p.add_argument("--no_online", dest="online", action="store_false")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overrides", nargs="*", default=[],
                   help="Config overrides as key=value pairs, e.g. "
                        "training.checkpoint_every=500")

    # HF sync options
    p.add_argument("--hf_repo", required=True,
                   help="HF Hub repo for checkpoints, e.g. 'username/video-model-ckpts'.")
    p.add_argument("--hf_token", default=None,
                   help="HF token. Falls back to $HF_TOKEN, then huggingface-cli login.")
    p.add_argument("--sync_every_minutes", type=int, default=10,
                   help="Periodic checkpoint push interval (min 1).")
    p.add_argument("--skip_initial_pull", action="store_true",
                   help="Don't try to pull existing checkpoints before training.")
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="Local checkpoint directory (also used as the HF repo root).")

    # LoRA
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_targets", nargs="+", default=None)

    # DiT needs the trained VAE. Leave unset to auto-detect (checkpoints/vae_latest.pt,
    # checkpoints/vae/vae_latest.pt, etc.); pass explicitly if it lives elsewhere.
    p.add_argument("--vae_checkpoint", default=None)

    return p.parse_args()


def _default_train_config(mode: str) -> str:
    return {
        "vae":  "configs/training/train_vae.yaml",
        "dit":  "configs/training/train_dit.yaml",
        "lora": "configs/training/train_dit.yaml",
    }[mode]


def _ensure_checkpoint_dir_in_overrides(overrides: list[str], checkpoint_dir: str) -> list[str]:
    """Force training.checkpoint_dir to our sync target, unless user already set it."""
    if any(o.startswith("training.checkpoint_dir=") for o in overrides):
        return overrides
    return overrides + [f"training.checkpoint_dir={checkpoint_dir}"]


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    os.chdir(PROJECT_ROOT)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: FAIL FAST if we can't actually save to HF Hub. Without this, a wrong
    # repo_id or read-only token means training runs for hours and saves nothing
    # (checkpoints die with the Kaggle instance). Abort now with a clear message.
    verify_writable(args.hf_repo, token=args.hf_token)

    # Step 1: pull any prior state. First-time runs return False — that's fine.
    if not args.skip_initial_pull:
        pull_checkpoints(args.hf_repo, checkpoint_dir, token=args.hf_token)

    # Step 2: start background sync. Stops + does final push via atexit/signal.
    sync = PeriodicSync(
        local_dir=checkpoint_dir,
        repo_id=args.hf_repo,
        interval_seconds=max(60, args.sync_every_minutes * 60),
        token=args.hf_token,
    ).start()

    # Make sure we ALWAYS push on exit, including SIGTERM (Kaggle's "session ending")
    # and Ctrl+C. Without this, a kill -15 from Kaggle would skip the final upload.
    def _shutdown(*_args):
        logger.info("Shutdown signal received — flushing checkpoints to HF Hub...")
        sync.stop(final_push=True)

    atexit.register(_shutdown)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda s, f: (_shutdown(), sys.exit(130)))
        except (ValueError, OSError):
            # Some platforms (e.g. notebooks on Windows) restrict signal handlers.
            pass

    train_config = args.train_config or _default_train_config(args.mode)
    overrides = _ensure_checkpoint_dir_in_overrides(args.overrides, str(checkpoint_dir))

    logger.info("Training mode=%s, model_config=%s, train_config=%s",
                args.mode, args.model_config, train_config)
    logger.info("Checkpoint dir: %s (synced every %d min to %s)",
                checkpoint_dir, args.sync_every_minutes, args.hf_repo)

    # Step 3: run training. Trainer's own find_latest_checkpoint() will resume
    # from whatever pull_checkpoints just dropped into checkpoint_dir.
    try:
        if args.mode == "vae":
            from training.train_vae import train_vae
            train_vae(
                config_path=train_config,
                model_config_path=args.model_config,
                use_online=args.online,
                online_limit=args.limit,
                overrides=overrides,
            )
        else:
            from training.train_dit import train_dit
            train_dit(
                model_config=args.model_config,
                train_config=train_config,
                use_online=args.online,
                online_limit=args.limit,
                use_lora=(args.mode == "lora"),
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_target_modules=args.lora_targets,
                vae_checkpoint=args.vae_checkpoint,
                overrides=overrides,
            )
    finally:
        # Belt-and-braces: stop the sync thread + push, even if atexit didn't fire
        # in the expected order (e.g. exception during interpreter shutdown).
        sync.stop(final_push=True)


if __name__ == "__main__":
    main()
