"""Caption cleaning and processing utilities."""

from __future__ import annotations
import re, unicodedata


def clean_caption(caption: str) -> str:
    """Clean and normalize a caption string."""
    caption = unicodedata.normalize("NFKC", caption)
    caption = re.sub(r"<[^>]+>", "", caption)  # Remove HTML
    caption = re.sub(r"https?://\S+", "", caption)  # Remove URLs
    caption = re.sub(r"[^\w\s.,!?;:'\"-]", " ", caption)  # Keep basic punctuation
    caption = re.sub(r"\s+", " ", caption).strip()
    if len(caption) < 3:
        caption = "a video clip"
    return caption


def truncate_caption(caption: str, max_tokens: int = 128) -> str:
    """Truncate caption to approximate token count (words ≈ 1.3 tokens)."""
    words = caption.split()
    max_words = int(max_tokens / 1.3)
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return caption


def augment_caption(caption: str) -> str:
    """Simple caption augmentation by varying structure."""
    prefixes = ["", "a video of ", "a clip showing ", "video: "]
    import random
    prefix = random.choice(prefixes)
    if prefix and not caption.lower().startswith(prefix.strip()):
        return prefix + caption.lower()
    return caption
