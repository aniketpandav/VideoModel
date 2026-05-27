"""CLI training entry point."""

from __future__ import annotations
import argparse, os, subprocess, sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def _prefer_project_venv() -> None:
    """Re-run with the project venv before importing training dependencies."""
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if os.environ.get("VIDEOGEN_SKIP_VENV_REEXEC") == "1" or not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    if current_python != venv_python.resolve():
        print(f"Re-running with project venv: {venv_python}")
        raise SystemExit(subprocess.call([str(venv_python), *sys.argv]))


_prefer_project_venv()
sys.path.insert(0, str(PROJECT_ROOT))


def _normalize_args(argv: list[str]) -> list[str]:
    """Accept `-- limit 1000` as a forgiving alias for `--limit 1000`."""
    if "--" not in argv:
        return argv

    idx = argv.index("--")
    if len(argv) > idx + 2 and argv[idx + 1] == "limit":
        return argv[:idx] + ["--limit", argv[idx + 2]] + argv[idx + 3:]
    return argv


def main():
    p = argparse.ArgumentParser(
        description="VideoGen Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train VAE using online streaming data (no local videos needed):
  python scripts/train.py --mode vae --online

  # Train DiT using online streaming data:
  python scripts/train.py --mode dit --online

  # Train DiT with at most 1000 online clips per epoch:
  python scripts/train.py --mode dit --online --limit 1000

  # Train DiT on local videos:
  python scripts/train.py --mode dit

  # Train with a custom config:
  python scripts/train.py --mode vae --online --model_config configs/model/dit_small.yaml

Online sources:
  pexels    - Nature / landscape clips (Open-Sora curated)
  internvid - Web video filtered to nature keywords
  synthetic - Procedural colour gradients (offline / testing)
  kabr      - Legacy metadata source; skipped with current HF Datasets
""")
    p.add_argument("--mode", choices=["vae", "dit", "lora"], default="dit",
                   help="Training mode")
    p.add_argument("--model_config", type=str, default="configs/model/dit_small.yaml",
                   help="Model architecture config YAML")
    p.add_argument("--train_config", type=str, default=None,
                   help="Training hyperparameter config YAML (auto-selected by mode if omitted)")
    p.add_argument("--online", action="store_true",
                   help="Stream training data from HuggingFace Hub — no local downloads needed")
    p.add_argument("--limit", type=int, default=None,
                   help="With --online, limit the number of online video clips per epoch")
    p.add_argument("--sources", nargs="+",
                   choices=["kabr", "mpala", "wilds_drones", "deepsea",
                            "pexels", "internvid", "synthetic"],
                   default=None,
                   help="Override online sources (e.g. --sources kabr synthetic)")
    p.add_argument("--overrides", nargs="*", default=[],
                   help="Config overrides as key=value pairs")
    # LoRA-specific args
    p.add_argument("--lora_rank", type=int, default=8,
                   help="LoRA rank (only used with --mode lora)")
    p.add_argument("--lora_alpha", type=float, default=1.0,
                   help="LoRA alpha scaling (only used with --mode lora)")
    p.add_argument("--lora_targets", nargs="+", default=None,
                   help="LoRA target modules (default: to_q to_v)")
    args = p.parse_args(_normalize_args(sys.argv[1:]))

    if args.limit is not None:
        if args.limit <= 0:
            p.error("--limit must be a positive integer")
        if not args.online:
            p.error("--limit can only be used with --online")

    # Auto-select training config based on mode
    if args.train_config is None:
        args.train_config = {
            "vae":  "configs/training/train_vae.yaml",
            "dit":  "configs/training/train_dit.yaml",
            "lora": "configs/training/train_dit.yaml",
        }[args.mode]

    # If --sources provided, patch it into the config on-the-fly via env var
    if args.sources and args.online:
        os.environ["VIDEOGEN_ONLINE_SOURCES"] = json.dumps(args.sources)
    if args.limit and args.online:
        os.environ["VIDEOGEN_ONLINE_LIMIT"] = str(args.limit)

    if args.mode == "vae":
        from training.train_vae import train_vae
        train_vae(config_path=args.train_config,
                  model_config_path=args.model_config,
                  use_online=args.online,
                  online_limit=args.limit,
                  overrides=args.overrides)

    elif args.mode in ("dit", "lora"):
        from training.train_dit import train_dit
        train_dit(model_config=args.model_config,
                  train_config=args.train_config,
                  use_online=args.online,
                  online_limit=args.limit,
                  use_lora=(args.mode == "lora"),
                  lora_rank=args.lora_rank,
                  lora_alpha=args.lora_alpha,
                  lora_target_modules=args.lora_targets,
                  overrides=args.overrides)


if __name__ == "__main__":
    main()
