"""CLI generation entry point."""

from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    p = argparse.ArgumentParser(description="VideoGen - Generate Videos")
    p.add_argument("--mode", choices=["t2v", "i2v"], default="t2v",
                   help="Generation mode: t2v (text-to-video) or i2v (image-to-video)")
    p.add_argument("--prompt", type=str, default="a beautiful sunset over the ocean")
    p.add_argument("--image", type=str, default=None, help="Reference image for i2v mode")
    p.add_argument("--output", type=str, default="output/generated.mp4")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--duration", type=float, default=None,
                   help="Optional duration in seconds (4 to 3600)")
    p.add_argument("--chunk_frames", type=int, default=None,
                   help="Frames per generation chunk for long videos")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt")
    p.add_argument("--config", type=str, default="configs/model/dit_small.yaml")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "t2v":
        from inference.text_to_video import text_to_video
        text_to_video(args.prompt, model_config=args.config, checkpoint=args.checkpoint,
                     output_path=args.output, num_frames=args.frames,
                     height=args.height, width=args.width,
                     guidance_scale=args.cfg, steps=args.steps,
                     seed=args.seed, fps=args.fps,
                     duration_seconds=args.duration,
                     chunk_frames=args.chunk_frames)
    elif args.mode == "i2v":
        if not args.image:
            print("Error: --image required for i2v mode")
            sys.exit(1)
        from inference.image_to_video import image_to_video
        image_to_video(args.image, args.prompt, model_config=args.config,
                      checkpoint=args.checkpoint, output_path=args.output,
                      num_frames=args.frames, height=args.height, width=args.width,
                      guidance_scale=args.cfg, steps=args.steps, seed=args.seed,
                      fps=args.fps, duration_seconds=args.duration,
                      chunk_frames=args.chunk_frames)


if __name__ == "__main__":
    main()
