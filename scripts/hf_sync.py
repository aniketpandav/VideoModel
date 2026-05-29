"""HuggingFace Hub sync helpers for cloud training.

Used by the Kaggle / cloud training wrappers to:
  * pull existing checkpoints down before training starts (resume across sessions)
  * push new checkpoints up periodically and on exit (so a killed session doesn't
    lose progress beyond the last sync interval)

Why HF Hub?
  - Free private model repos up to 50 GB
  - upload_folder() diffs against the remote and only uploads changed files,
    so a periodic full-folder sync is cheap (no per-step bookkeeping needed)
  - Same API works from Kaggle, Colab, RunPod, your laptop — no S3 setup

The token can come from three places, in priority order:
  1. Explicit `token` argument
  2. HF_TOKEN environment variable
  3. ~/.huggingface/token (set by `huggingface-cli login`)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# These files change every save but aren't worth syncing in real time.
# (They are still synced on the final push at exit.)
_TRANSIENT_PATTERNS = (
    "*.tmp",
    "*.lock",
    "*.partial",
)


def _resolve_token(token: Optional[str]) -> Optional[str]:
    if token:
        return token
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    return None


def pull_checkpoints(
    repo_id: str,
    local_dir: str | Path,
    token: Optional[str] = None,
    allow_patterns: Optional[list[str]] = None,
    repo_type: str = "model",
) -> bool:
    """Download all (or matching) files from an HF Hub repo into local_dir.

    Returns True if files were pulled, False if the repo doesn't exist yet
    (e.g. first training session — perfectly normal).
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=_resolve_token(token),
            repo_type=repo_type,
            allow_patterns=allow_patterns,
        )
        logger.info("Pulled checkpoints from %s -> %s", repo_id, local_dir)
        return True
    except RepositoryNotFoundError:
        logger.info("Repo %s does not exist yet — starting fresh.", repo_id)
        return False
    except Exception as e:
        # Network blip, auth issue, etc. Don't kill training — log and continue.
        logger.warning("Pull from %s failed (%s) — continuing without remote state.",
                       repo_id, e)
        return False


def verify_writable(
    repo_id: str,
    token: Optional[str] = None,
    repo_type: str = "model",
) -> None:
    """Fail fast if we cannot create the repo and upload to it.

    Call this at startup BEFORE training. The whole point of cloud training is
    that checkpoints persist off-instance; if the token is read-only or the
    repo_id is wrong, we must abort now rather than train for hours into a void
    (which is exactly what happened when pushes failed silently).

    Raises RuntimeError with an actionable message on any failure.
    """
    from huggingface_hub import HfApi, create_repo

    resolved = _resolve_token(token)
    if not resolved:
        raise RuntimeError(
            "No HuggingFace token found. Set the HF_TOKEN secret (with WRITE access) "
            "or pass --hf_token. Create one at https://huggingface.co/settings/tokens"
        )

    api = HfApi(token=resolved)

    # 1. Confirm the token authenticates and has write capability.
    try:
        me = api.whoami()
    except Exception as e:
        raise RuntimeError(
            f"HuggingFace token is invalid or expired: {e}. "
            "Create a new WRITE token at https://huggingface.co/settings/tokens "
            "and update the HF_TOKEN secret."
        ) from e

    role = (me.get("auth", {}).get("accessToken", {}) or {}).get("role")
    if role == "read":
        raise RuntimeError(
            f"HuggingFace token for user '{me.get('name')}' is READ-ONLY (role='read'). "
            "Checkpoints cannot be uploaded with a read token. Create a WRITE token at "
            "https://huggingface.co/settings/tokens and update the HF_TOKEN secret."
        )

    # 2. Confirm we can create / access the target repo.
    try:
        create_repo(repo_id=repo_id, token=resolved, repo_type=repo_type,
                    private=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(
            f"Cannot create or access HF repo '{repo_id}': {e}. "
            f"Check that HF_REPO is '<your-username>/<repo-name>' (you are '{me.get('name')}') "
            "and that the token has write access to it."
        ) from e

    # 3. Round-trip a tiny file so we KNOW uploads actually work.
    try:
        api.upload_file(
            path_or_fileobj=b"ok",
            path_in_repo=".sync_check",
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="sync write-check",
        )
    except Exception as e:
        raise RuntimeError(
            f"Write-check upload to '{repo_id}' failed: {e}. "
            "The repo exists but the token can't upload to it."
        ) from e

    logger.info("HF Hub write-check passed: %s (user=%s, role=%s)",
                repo_id, me.get("name"), role)


def push_checkpoints(
    local_dir: str | Path,
    repo_id: str,
    token: Optional[str] = None,
    repo_type: str = "model",
    ignore_patterns: Optional[list[str]] = None,
    commit_message: str = "Sync checkpoints",
    create_if_missing: bool = True,
) -> bool:
    """Upload local_dir to an HF Hub repo, creating it if needed.

    Uses upload_folder which diffs against remote and only sends changed files.
    Returns True on success, False on failure (logged, not raised — we don't
    want a transient upload failure to kill an otherwise-healthy training run).
    """
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import HfHubHTTPError

    local_dir = Path(local_dir)
    if not local_dir.exists() or not any(local_dir.iterdir()):
        logger.debug("Skipping push: %s is empty or missing.", local_dir)
        return False

    resolved_token = _resolve_token(token)
    api = HfApi(token=resolved_token)

    if create_if_missing:
        try:
            create_repo(repo_id=repo_id, token=resolved_token,
                        repo_type=repo_type, private=True, exist_ok=True)
        except Exception as e:
            logger.warning("create_repo(%s) failed: %s", repo_id, e)

    patterns = list(_TRANSIENT_PATTERNS)
    if ignore_patterns:
        patterns.extend(ignore_patterns)

    try:
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
            ignore_patterns=patterns,
        )
        logger.info("Pushed %s -> %s", local_dir, repo_id)
        return True
    except HfHubHTTPError as e:
        logger.warning("Push to %s failed (HTTP): %s", repo_id, e)
        return False
    except Exception as e:
        logger.warning("Push to %s failed: %s", repo_id, e)
        return False


class PeriodicSync:
    """Background thread that periodically pushes a folder to HF Hub.

    Use as a context manager OR call start()/stop() explicitly. Stopping
    triggers a final push so anything written since the last interval lands.

    Example:
        with PeriodicSync(local_dir="checkpoints", repo_id="me/ckpts",
                          interval_seconds=600) as sync:
            run_training()
        # final push happens automatically on exit
    """

    def __init__(
        self,
        local_dir: str | Path,
        repo_id: str,
        interval_seconds: int = 600,
        token: Optional[str] = None,
        repo_type: str = "model",
        on_push: Optional[Callable[[bool], None]] = None,
    ):
        self.local_dir = Path(local_dir)
        self.repo_id = repo_id
        self.interval_seconds = max(60, int(interval_seconds))
        self.token = token
        self.repo_type = repo_type
        self.on_push = on_push

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            # Wait first, then push — so we don't double-push right after a manual
            # pull at startup. The final push on stop() catches anything new.
            woken = self._stop_event.wait(self.interval_seconds)
            if woken:
                break
            ok = push_checkpoints(
                self.local_dir, self.repo_id,
                token=self.token, repo_type=self.repo_type,
                commit_message=f"Periodic sync ({time.strftime('%Y-%m-%d %H:%M:%S')})",
            )
            if self.on_push:
                try:
                    self.on_push(ok)
                except Exception:
                    pass

    def start(self) -> "PeriodicSync":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                         name="HFPeriodicSync")
        self._thread.start()
        logger.info("Started periodic HF sync: %s every %ds -> %s",
                    self.local_dir, self.interval_seconds, self.repo_id)
        return self

    def stop(self, final_push: bool = True) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if final_push:
            push_checkpoints(
                self.local_dir, self.repo_id,
                token=self.token, repo_type=self.repo_type,
                commit_message=f"Final sync ({time.strftime('%Y-%m-%d %H:%M:%S')})",
            )

    def __enter__(self) -> "PeriodicSync":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(final_push=True)


if __name__ == "__main__":
    # Tiny CLI for manual one-shot pull/push from a shell.
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="One-shot HF Hub sync helper.")
    p.add_argument("action", choices=["pull", "push"])
    p.add_argument("--local_dir", required=True)
    p.add_argument("--repo_id", required=True)
    p.add_argument("--token", default=None)
    p.add_argument("--repo_type", default="model")
    args = p.parse_args()

    if args.action == "pull":
        pull_checkpoints(args.repo_id, args.local_dir,
                         token=args.token, repo_type=args.repo_type)
    else:
        push_checkpoints(args.local_dir, args.repo_id,
                         token=args.token, repo_type=args.repo_type)
