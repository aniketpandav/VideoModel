"""ONNX model export script."""

from __future__ import annotations
import argparse, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_config
from models.dit.dit import VideoDiT
from utils.checkpoint import load_checkpoint


def export_onnx(checkpoint: str, config_path: str = "configs/model/dit_small.yaml",
                output: str = "exports/model.onnx", opset: int = 17):
    """Export DiT model to ONNX format."""
    config = load_config(config_path)
    dit_cfg = config.model.dit
    vae_cfg = config.model.vae

    dit = VideoDiT(in_channels=vae_cfg.latent_channels, hidden_size=dit_cfg.hidden_size,
                   num_layers=dit_cfg.num_layers, num_heads=dit_cfg.num_heads,
                   mlp_ratio=dit_cfg.mlp_ratio, patch_size=tuple(dit_cfg.patch_size),
                   cross_attention_dim=config.model.text_encoder.hidden_size)

    if Path(checkpoint).exists():
        load_checkpoint(checkpoint, dit)

    dit.eval()
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # Dummy inputs
    B, Z, T, H, W = 1, vae_cfg.latent_channels, 4, 32, 32
    dummy_latent = torch.randn(B, Z, T, H, W)
    dummy_t = torch.tensor([500])
    dummy_ctx = torch.randn(B, 64, config.model.text_encoder.hidden_size)

    torch.onnx.export(dit, (dummy_latent, dummy_t, dummy_ctx), output,
                      opset_version=opset, input_names=["latent", "timestep", "context"],
                      output_names=["noise_pred"],
                      dynamic_axes={"latent": {0: "batch", 2: "T", 3: "H", 4: "W"},
                                    "context": {0: "batch", 1: "seq_len"}})
    print(f"ONNX model exported to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/model/dit_small.yaml")
    p.add_argument("--output", type=str, default="exports/model.onnx")
    args = p.parse_args()
    export_onnx(args.checkpoint, args.config, args.output)
