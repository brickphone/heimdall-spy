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

## 1. Choose a machine (preference: RTX 3090, A40, RTX 4000 Ada; EU; ≤ $1/h)

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

## 2. Create the pod (RunPod API)

- Image: `runpod/pytorch:1.6.0-cu1281-torch280-ubuntu2404`. Its default `python` is
  Python 3.12 (deadsnakes; verified from the image's build history) and it ships rsync,
  tmux and sshd, so setup needs no GitHub download and no apt installs.
- Env: `PUBLIC_KEY` = the dedicated `~/.ssh/heimdall_runpod.pub`;
  `MAPILLARY_ACCESS_TOKEN` = `{{ RUNPOD_SECRET_mapillary_token }}`. RunPod fills it in
  from the account secret, so the token is never read, printed or copied by us.
- Ports `22/tcp` with a public IP (rsync needs real SSH, not the SSH proxy).
- Time limit: if setup (creation → "GPU visible, MegaLoc loads") takes > 15 min,
  terminate.

## 3. Copy code and data (from the Mac)

```bash
rsync -a --exclude-from=<excludes: .env*, data/, .venv*, .uv-python/, caches> ./ pod:/root/work/swespy/
rsync -aR data/gallery_v2/plan.json data/gallery_v2/plan-report.json \
  data/public_eval/recent_linkoping_streets_v1/manifest.json \
  data/megaloc/assets/<pinned revision>/ pod:/root/work/swespy/
```

The **test-set manifest is not copied**: the build runs with `--defer-sealed-guard` and
the sealed guard is applied on the Mac (step 6). The pinned MegaLoc source and licence
are copied (the checksum check still runs); the weights come from Hugging Face.

## 4. Set up on the pod

```bash
ssh pod 'bash /root/work/swespy/scripts/cloud_setup.sh'
```

`uv sync --frozen` with the image's `/usr/bin/python3.12` and `UV_PYTHON_DOWNLOADS=never`
(PyPI only), then prints `token_present: True` (never the value), the GPU name and
`megaloc_loaded: device cuda`.

## 5. Run the build in tmux

```bash
ssh pod 'tmux new -d -s build "bash /root/work/swespy/scripts/cloud_build.sh 2>&1 | tee -a /root/work/swespy/data/gallery_v2/build.log"'
```

Watch read-only: `ssh -t pod tmux attach -r -t build` (detach `Ctrl-b d`). Resumable:
rerun the same command after any interruption.

## 6. Copy back, finalize and verify (on the Mac)

```bash
rsync -a pod:/root/work/swespy/data/gallery_v2/ data/gallery_v2/ \
  --exclude plan.json --exclude plan-report.json
rsync -a pod:/root/work/swespy/data/megaloc/descriptors/ data/megaloc/descriptors/
for s in recent historic; do
  .venv-benchmark/bin/python -m eval.gallery_build --stage $s --finalize-sealed
done
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-benchmark/bin/python -m eval.gallery_build \
  --stage historic --verify 200 --device mps
.venv-benchmark/bin/python -m eval.densify coverage --index-name gallery-index-recent.json
.venv-benchmark/bin/python -m eval.densify coverage --index-name gallery-index-historic.json
```

`--finalize-sealed` applies the sealed test-set guard locally and reports counts only;
`swespy.model.locator.Gallery` refuses an index whose sealed guard is still pending.
`--verify` must report `"ok": true` (cosine ≥ 0.999 for 200 random references).

## 7. Terminate

Terminate the pod via the API (`podTerminate`) — not just stop it — and confirm
`myself { pods }` is empty. The container disk (code, images, dev manifest) is
deleted with it.

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
