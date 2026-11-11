# Approximate gallery search (replacing the exact scan)

Date: 2026-11-11. **Dev set only (66); test set untouched; no field photos.**
Motivation: with automatic crops, the exact search (one pass over all 91,714 × 8,448
descriptors per variant) seemed to dominate Mac latency (experiment 14: ~9 s of a 14 s
query) and keeps 3.1 GB hot in memory. (Result: the 9 s was a cold page cache; see below.)

## Pre-registration (written before any index was built or measured)

**Index family.** FAISS `IndexIVFPQ` (L2 distance, like the exact search) built
once from the memory-mapped descriptor matrix and stored next to it, plus **exact
re-scoring**: the IVF-PQ index proposes `refine × depth` candidates, whose exact squared
L2 distances are recomputed from the memory-mapped vectors; the best `depth` (50) by exact
distance are returned. Returned similarities are therefore exact; only the candidate set
can differ from the exact search.

Fixed build settings: `nlist = 1024` coarse centroids, `PQ m = 96` sub-quantizers of
8 bits (8,448 / 96 = 88 dimensions each), trained on all gallery vectors, seed 1234,
PQ on the raw vectors (`by_residual = False`: training on residuals would copy the
3.1 GB matrix in RAM). Candidate distances are recomputed with `faiss.knn` on the
candidate rows, the same per-pair arithmetic as the exact search.

**Grid** (search-time only, nothing rebuilt): `nprobe ∈ {8, 16, 32, 64, 128}` ×
`refine ∈ {4, 10}` (200 or 500 candidates re-scored exactly).

**Pass criteria, on the 66 dev photos, both for the whole photo and for the default
pipeline with automatic crops (`full`):**

1. The merged retrieval top 5 (IDs and order) is identical to the exact search for
   **≥ 95% of dev photos** (≥ 63 of 66). Since re-ranking verifies exactly these 5 and the
   similarities are exact, an identical top 5 gives an identical answer.
2. **Dev within 500 m is unchanged** against the exact search on the same machine (the
   Mac; experiment 14 on the pod gave whole 31/66, full 33/66). Photos whose top 5 is
   identical keep their result by construction; the others are re-run through the full
   pipeline.

**Choice among passing settings:** the lowest median search time on the Mac for the `full`
pipeline over the 66 dev photos; ties go to the smaller `nprobe`, then the smaller
`refine`. If no setting passes, the exact search stays and this is reported.

**Then:** the chosen setting becomes the default search, the exact search stays available
(`GALLERY_SEARCH=exact`), and Mac end-to-end latency is reported for `whole` and `full`.

## Result 1: IVF-PQ fails the rule

No setting passed criterion 1. Best: `nprobe 128, refine 10` with 24/66 (whole) and 27/66
(full) identical top 5, against ≥ 63 required. Not a bug: on shared IDs the re-scored
similarities equal the exact ones to the last digit, and photos with a clear best match
(e.g. cosine 0.40) are identical. The misses are genuine. Dev queries sit far from every
reference (top cosine typically 0.11–0.19, gaps in the third decimal), finer than
8-bit PQ on 88-dimensional slices resolves, so the true neighbours often do not reach the
500 re-scored candidates (top-50 recall 10–33/50 on 6 photos checked). Per the rule, the
exact search stays for IVF-PQ.

| nprobe | refine | identical whole | identical full | search ms whole | search ms full |
| --- | --- | --- | --- | --- | --- |
| 8 | 4 | 13 | 14 | 2.1 | 8.1 |
| 8 | 10 | 13 | 14 | 2.6 | 11.8 |
| 16 | 4 | 15 | 15 | 1.8 | 8.0 |
| 16 | 10 | 15 | 15 | 2.5 | 11.7 |
| 32 | 4 | 15 | 18 | 1.9 | 8.6 |
| 32 | 10 | 18 | 21 | 2.8 | 12.2 |
| 64 | 4 | 19 | 20 | 2.0 | 9.3 |
| 64 | 10 | 23 | 24 | 2.8 | 13.1 |
| 128 | 4 | 19 | 23 | 2.3 | 10.6 |
| 128 | 10 | 24 | 27 | 3.1 | 14.5 |

Exact search, same queries: 64.2 ms (whole), 321.3 ms (full), median per photo. Rows: `data/auto_regions/ann/select_ivfpq.json`.

## Second family (registered after result 1, before it was built or measured)

**IVF with exact scoring (`ivf`)**: the same 1,024 coarse centroids and list assignments
as the IVF-PQ index (no new training); a query probes its `nprobe` nearest lists and
**every** vector in them is scored exactly from the memory-mapped matrix (`faiss.knn`).
Recall depends only on probing the right lists; no quantization. Grid:
`nprobe ∈ {16, 32, 64, 128, 256}`. Same pass criteria (≥ 63/66 identical top 5 for whole
and full; dev within 500 m unchanged), same choice rule (lowest median full-pipeline search
time; ties to the smaller `nprobe`). If none passes, the exact search stays.

## Result 2: IVF with exact scoring also fails

| nprobe (of 1,024) | identical whole | identical full | search ms whole | search ms full |
| --- | --- | --- | --- | --- |
| 16 | 18 | 15 | 16.8 | 80.5 |
| 32 | 23 | 21 | 31.0 | 118.2 |
| 64 | 29 | 27 | 44.8 | 200.7 |
| 128 | 35 | 33 | 62.6 | 318.9 |
| 256 | 46 | 42 | 114.8 | 472.4 |

Exact search on the same queries in this run: 70.8 ms (whole), 370.0 ms (full).
Rows: `data/auto_regions/ann/select_ivf.json`.

Even probing a quarter of the gallery keeps the top 5 for only 46/66 photos, and is
already slower than the exact scan. The nearest references of these dev photos are spread
over many clusters (low, nearly tied similarities), so no partition recovers them cheaply.

## Decision

**No approximate setting passes; the exact search stays the default**
(`GALLERY_SEARCH=exact`). The IVF-PQ index (53 MB, `gallery-index-historic.ivfpq.faiss`)
and both search modes remain available but unused.

## What the latency problem actually is

On a warm page cache the exact search is fast on the Mac: **70.8 ms for the whole
photo and 370.0 ms with automatic crops** (66 dev photos, median). The 5–9 s
searches measured earlier (experiment 14 and the first demo query) happened when the
3.1 GB descriptor matrix was not resident: under memory pressure macOS evicts its pages and
every search re-reads it from disk. The cost is memory, not computation, and an index that
must still re-score from the same matrix does not remove it.

End-to-end Mac latency for `whole` and `full` was **not measured**: after the selection
runs, swap stood at 7.3 GB with the demo API resident (4.9 GB), above the 6 GB limit set
for these runs. Estimate from the parts measured on the Mac: the first demo query with
crops took 14.1 s, of which 9.1 s was a cold search; with a warm search (0.3 s) that query
would take ≈ 5.3 s, dominated by verification (3.8 s) and MegaLoc (1.0 s).
