# Gallery build speed: encoding benchmark, pipelining, staged build

Date: 2026-11-07. Nothing was downloaded or built; the benchmark used existing images.

## Encoding benchmark (M1, MPS)

Setup: `swespy/eval/encode_benchmark.py`. 160 random existing gallery images,
re-encoded as gallery v2 stores them (640 px JPEG). Each (dtype, batch) runs in a
fresh process with a warm-up at its own batch size. The first attempt ran all
settings in one process: the batch-32/64 runs left ~7.5 GB of MPS memory cached,
the Mac (16 GB, with ~8.3 GB of 9.2 GB swap already in use) started paging, and
every later setting slowed down. Those numbers were discarded.

| Batch | fp32 model-only | fp32 end-to-end | GPU memory | fp16 (autocast) end-to-end | GPU memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.0 | 8.0 | 1.9 GB | 7.0 | 2.0 GB |
| 8 | 8.8 | 8.9 | 2.0 GB | 7.5 | 3.0 GB |
| **16** | **9.0** | **8.9** | **2.1 GB** | 7.7 | 4.1 GB |
| 32 | 9.1 | 9.0 | 3.3 GB | 8.0 | 4.3 GB |
| 64 | 8.7 | 8.5 | 7.5 GB | 7.5 | 6.5 GB |

Images per second. End-to-end includes JPEG decoding and preprocessing.

**fp16.** `model.half()` crashes because the pinned MegaLoc mixes in fp32 constants
(`mps.add` with fp32 and fp16 operands), so fp16 was run through `torch.autocast`.
On MPS it is **slower** than fp32 at every batch size. It also changes results on the
66 dev photos against the 2,489-reference polygon gallery:

| | top-1 unchanged | top-5 set unchanged |
| --- | ---: | ---: |
| Gallery and query in fp16 | 64/66 | 58/66 |
| fp16 gallery, fp32 query | 65/66 | 58/66 |

Descriptor cosine fp16 vs fp32: minimum 0.9970, mean 0.9998. That isn't "the same
top-5 within a small tolerance", so **fp32 stays** (also recommended for the cloud).

**Positional-encoding cache.** MegaLoc's DINOv2 backbone interpolates positional
embeddings with a bicubic operator that MPS lacks, so each forward pass falls back to
the CPU. `swespy/model/megaloc.py:cache_pos_encoding` memoises the result for a fixed
input size. Descriptors are bit-identical (maximum difference 0.0 on 32 images). The
speed-up was negligible (batch 16: 8.77 → 8.82 img/s), so it was not the bottleneck.
It is kept because it is harmless.

**Default: fp32, batch 16.** Batch 32 is 0.6% faster end to end but uses 1.2 GB more
unified memory on a machine that was already swapping. Batch 64 is slower and caused
the paging seen in the first run.

## Pipelined, staged build (`swespy/eval/gallery_build.py`)

- The frozen plan is processed in 1,000-reference chunks. An asyncio producer runs
  download → panorama crops → quality rules (parallel process pool) → near-duplicate
  check against everything kept so far → pixel-hash leakage guards (dev visible, test
  sealed). It then writes a chunk checkpoint.
- A GPU thread encodes finished chunks while the producer continues. Image decoding is
  prefetched in a small thread pool, at most four batches ahead. A bounded queue (3
  chunks) limits how far downloads run ahead.
- Resume: finished chunks are skipped, filtered-but-unencoded chunks are queued first,
  and descriptors already in the hash-keyed cache are not recomputed. Duplicate state
  is rebuilt from the checkpoints.
- Stages:

| Stage | Contents | References to encode |
| --- | --- | ---: |
| `recent` | Mapillary 2018+, all 257 panoramas (×8 crops), 1,736 Commons, 334 KartaView | 36,588 |
| `historic` | adds pre-2018 Mapillary to the same index | 60,738 |

  `historic` refuses to start until `recent` is complete. Part 6 evaluates after each
  stage, replacing the pre-2018 ablation.
- Tests: `tests/unit/test_gallery_build.py` covers both stages, resume and stage
  ordering, with a mocked network and a fake encoder.

## Time estimates (encoding at 8.9 img/s, measured)

| | Before (sequential) | Now (pipelined) |
| --- | --- | --- |
| `recent` | — | ~1.2–1.4 h |
| `historic` | — | ~2.0–2.3 h |
| Total | ~10 h estimated | **~3.5 h** |

The old ~10 h estimate assumed 3.5 img/s from an earlier full-pipeline run plus a
separate download stage. Most of the improvement comes from measuring encoding
properly (8.9 img/s). Pipelining hides roughly 1 h of download and filtering, and
staging gives first results after about 1.3 h. Download throughput hasn't been
measured yet; if it falls below ~9 img/s, the build becomes download-bound.

Cloud alternative: [`docs/CLOUD_BUILD.md`](../CLOUD_BUILD.md). An RTX 4090 is
estimated (not measured) at ~1–1.5 h wall-clock and ~$0.50–1.

## Commands (not run)

```bash
cd swespy
caffeinate -i scripts/build_gallery_v2.sh recent   2>&1 | tee -a data/gallery_v2/build.log
caffeinate -i scripts/build_gallery_v2.sh historic 2>&1 | tee -a data/gallery_v2/build.log
```

Quit Docker Desktop and other memory-heavy apps first; the benchmark showed that
memory pressure, not compute, is what slows the M1 down.
