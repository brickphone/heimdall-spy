# Building gallery v2 on a rented GPU (runbook — not executed)

This runbook moves the slow part of the gallery v2 build (MegaLoc encoding) to a
rented NVIDIA GPU. Nothing here has been rented or run yet. The build is the same
code and frozen plan as on the Mac (`swespy/eval/gallery_build.py`), so the result is
identical apart from GPU floating-point rounding, which step 7 checks.

## What runs where

| Step | Where | Data moved |
| --- | --- | --- |
| Metadata + frozen plan (already done) | Mac | — |
| Download, crops, quality filter, leakage guards, MegaLoc encoding | Cloud GPU | images come straight from Mapillary/Commons/KartaView to the cloud machine |
| Copy the gallery back | Cloud → Mac | ~5 GB images + ~3.3 GB descriptors + JSON |
| Verify descriptors, coverage map, evaluation | Mac | — |

## 1. Choose a machine

Any recent NVIDIA GPU works; the build is limited by downloading and JPEG decoding
more than by the GPU. Recommended:

| Provider | GPU | Typical on-demand price* | Notes |
| --- | --- | --- | --- |
| RunPod (Community/Secure Cloud) | RTX 4090 24 GB | ~$0.35–0.70/h | Pick ≥ 8 vCPU, ≥ 32 GB RAM, ≥ 40 GB disk |
| Vast.ai | RTX 4090 24 GB | ~$0.30–0.60/h | Filter: ≥ 8 vCPU, ≥ 500 Mbit/s download, ≥ 40 GB disk, "verified" host |
| RunPod / Vast.ai | A100 40/80 GB | ~$1.00–1.90/h | Not needed; only faster at encoding, which is not the bottleneck |

\*Prices change daily; check the provider's page before renting. Choose an image with
CUDA 12.x and Ubuntu 22.04 (e.g. RunPod "PyTorch 2.x" template). Prefer a host with
several CPU cores and fast download bandwidth over a bigger GPU.

## 2. Copy code and the frozen plan to the machine (from the Mac)

The Mapillary token and `.env` are **never** copied.

```bash
# On the Mac, from the repository root. HOST/PORT come from the provider's SSH panel.
HOST=root@<ip>; PORT=<port>
cd ~/Documents/GitHub/heimdall-spy/swespy
rsync -avP -e "ssh -p $PORT" \
  --exclude .env --exclude '.venv*' --exclude data --exclude qdrant_storage \
  --exclude '__pycache__' ./ $HOST:/workspace/swespy/
# Frozen plan + the two eval manifests the leakage guards need (metadata only;
# the test manifest is read by the sealed guard and is deleted again in step 8).
rsync -avP -e "ssh -p $PORT" -R \
  data/gallery_v2/plan.json data/gallery_v2/plan-report.json \
  data/public_eval/recent_linkoping_streets_v1/manifest.json \
  data/paired_views/recent_pilot_v2/manifest.json \
  $HOST:/workspace/swespy/
```

## 3. Set up Python (on the machine)

```bash
ssh -p $PORT $HOST
cd /workspace/swespy
pip install uv==0.9.5
uv sync --frozen            # torch 2.5.1 Linux wheels include CUDA 12.4
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

MegaLoc source and weights download on first use from GitHub/Hugging Face at the
pinned revisions and are checksum-verified (`swespy/model/megaloc.py`).

## 4. Provide the Mapillary token for this session only

```bash
read -rs MAPILLARY_ACCESS_TOKEN && export MAPILLARY_ACCESS_TOKEN   # paste, Enter; not echoed
```

`read -s` keeps it out of shell history; it lives only in this shell's environment
and is read by `swespy.config.Settings`. Do not put it in a provider "environment
variables" template (stored in their dashboard) or in any file. Run the build inside
`tmux` so it survives a dropped SSH connection — start `tmux` *before* the `read`
line, or repeat the `read` inside tmux.

## 5. Run the build

```bash
tmux new -s build
# (repeat step 4 here if you started tmux after it)
uv run python -u -m eval.gallery_build --stage recent   --device cuda \
    --dtype float32 --batch-size 64 2>&1 | tee -a data/gallery_v2/build.log
uv run python -u -m eval.gallery_build --stage historic --device cuda \
    --dtype float32 --batch-size 64 2>&1 | tee -a data/gallery_v2/build.log
```

Keep `float32`: fp16 changed the top-1 match for 2 of the 66 dev photos and the top-5
set for 8 of them in our check (`docs/experiments/05_build_speed.md`). It would also
not match the Mac's fp32 query encoder. Both commands resume where they stopped; rerun the same line after any interruption.
Detach with `Ctrl-b d`, reattach with `tmux attach -t build`.

## 6. Copy the result back to the Mac

```bash
# On the Mac:
cd ~/Documents/GitHub/heimdall-spy/swespy
rsync -avP -e "ssh -p $PORT" $HOST:/workspace/swespy/data/gallery_v2/ data/gallery_v2/ \
  --exclude 'mapillary/' --exclude 'commons/' --exclude 'kartaview/'
rsync -avP -e "ssh -p $PORT" $HOST:/workspace/swespy/data/megaloc/descriptors/ \
  data/megaloc/descriptors/
```

This brings `gallery-index-recent.json`, `gallery-index-historic.json`,
`encoder-identity.json`, `chunks/` (per-chunk filter and guard reports), the stored
640 px images (needed for Part 5 re-ranking) and the descriptor cache. Expect
~8.5 GB; at 100 Mbit/s home bandwidth ≈ 12 min.

## 7. Verify on the Mac before using the index

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-benchmark/bin/python -m eval.gallery_build \
    --stage historic --verify 200 --device mps
```

Re-encodes 200 random references with the Mac's encoder and compares them with the
cloud descriptors; `"ok": true` requires cosine ≥ 0.999 for every one. If it fails, do
not mix the cloud gallery with Mac-encoded queries; re-encode locally instead.

## 8. Clean up (on the machine, then destroy it)

```bash
unset MAPILLARY_ACCESS_TOKEN
rm -rf /workspace/swespy        # removes images, eval manifests and the plan
```

Then **terminate/destroy** the instance in the provider console (stopping alone may
keep billing for storage).

## Expected time and cost

Measured on the M1 (MPS, fp32, batch 16): **8.9 images/s** to encode. The cloud
figures below are **estimates, not measurements**. No cloud run has been done, and
download throughput from Mapillary has not been measured on any machine yet.

| | M1 Mac (pipelined) | RTX 4090 (estimate) |
| --- | --- | --- |
| Encoding speed | 8.9 img/s (measured) | ~150–300 img/s fp32 (DINOv2-B at 322 px; unmeasured) |
| Stage `recent` (36,588 refs incl. 2,056 panorama crops) | ~1.2–1.4 h (encode-bound) | ~15–25 min (download-bound) |
| Stage `historic` (60,738 refs) | ~2.0–2.3 h | ~20–35 min |
| Setup + copy back (8.5 GB at 100 Mbit/s) | — | ~15 min + ~12 min |
| **Total** | **~3.5 h** | **~1–1.5 h wall-clock** |
| Cost | electricity | RTX 4090 at ~$0.35–0.70/h → **~$0.50–1.00**; A100 at ~$1.0–1.9/h → ~$1.5–3 |

Download assumptions: about 6 s per 100-ID Graph API request with four in flight, and
twelve parallel image downloads. Once encoding is fast, download plus quality
filtering is the bottleneck on both machines. Since the Mac now finishes in about one
evening, the cloud mainly saves about 2 hours. It adds setup, a copy-back and an
extra place that holds the test manifest, so the Mac remains the default.
