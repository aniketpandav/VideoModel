# Serving layer

Turns a trained model into an HTTP video-generation service + a minimal demo website.
Built dual-track (see [BLUEPRINT.md](../BLUEPRINT.md)):

| Backbone | What it serves | Where it runs | Quality |
|---|---|---|---|
| `toy` (default) | the from-scratch 3D U-Net checkpoint | local GTX 1650 (or CPU) | demo: moving shapes |
| `ltx` | pretrained LTX-Video (Apache-2.0) | **cloud GPU, 8 GB+** | production T2V |

The HTTP contract is identical for both — swapping is one env var, no caller changes.

## Run it locally (toy backbone)

```powershell
# 1) one-time deps (into the project venv)
./env/Scripts/python.exe -m pip install -r serve/requirements.txt

# 2) train a checkpoint first (writes runs/local/last.pt)
#    expandable_segments avoids fragmentation OOM on the marginal 4 GB card
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
./env/Scripts/python.exe scripts/train.py --config configs/train_local.yaml

# 3) start the API + demo site
./env/Scripts/python.exe -m uvicorn serve.api:app --port 8000
#   then open http://localhost:8000
```

> **One process at a time on this box.** The 4 GB GTX 1650 + a small Windows paging
> file cannot host two torch processes at once. Run *either* training *or* the server
> locally. In production they live on separate hosts, so this is a dev-only constraint.
> To serve while training runs, pin the API to CPU: `set VDM_DEVICE=cpu`.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `VDM_BACKBONE` | `toy` | `toy` or `ltx` |
| `VDM_CKPT` | `runs/local/last.pt` | toy checkpoint path |
| `VDM_DEVICE` | auto | force `cpu` or `cuda` |
| `VDM_OUT` | `outputs` | where rendered clips are written |

## API

| Method | Path | Body / result |
|---|---|---|
| `GET` | `/health` | `{status, backbone, trained_steps}` |
| `POST` | `/v1/videos` | `{prompt, steps?, seed?, fmt?}` → `{job_id, status}` |
| `GET` | `/v1/videos/{job_id}` | `{status, progress, url?, error?}` |
| `GET` | `/v1/videos/{job_id}/file` | the rendered mp4/gif |
| `GET` | `/` | demo web page |

The toy model is class-conditional on motion **direction** — put `left`/`right`/`up`/`down`
in the prompt; other prompts hash to a deterministic direction.

## Go to production (Track A)

1. On a cloud GPU host: `pip install diffusers transformers accelerate sentencepiece`.
2. Set `VDM_BACKBONE=ltx` and start the same server.
3. Replace the in-process `ThreadPoolExecutor` in [api.py](api.py) with Redis/RQ workers
   (BLUEPRINT sec 13) and write artifacts to S3/MinIO instead of local disk.
4. Add the render tail (RIFE → Real-ESRGAN → FFmpeg grade) inside `serve/render.py:finish_clip`.
