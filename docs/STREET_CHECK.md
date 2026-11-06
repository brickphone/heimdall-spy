# Automated Linkoping street sanity check

Completed follow-up: [paper reproduction and dense Linkoping results](PAPER_REPRODUCTION.md). The measurements below retain their original experiment context.

Measured 2026-11-06 against the rebuilt 317-reference collection and the existing
pretrained CLIP retrieval API, including the cosine-roundoff response correction.
This is a separate Mapillary street test, not the independent user-photo baseline.

A subsequent [MegaLoc comparison](MEGALOC_COMPARISON.md) uses these exact queries
and references. Candidate retrieval improved, while precise averaged-pin results
were mixed. The CLIP artifacts and original index were preserved.

## Fixed selection and leakage safeguards

- Examined 1,483 metadata records within the configured central-Linkoping pilot;
  671 unique candidates passed ID, sequence, coordinate and attribution checks.
- Excluded all 399 local raw image IDs and all 92 capture sequences represented
  by the indexed references. Reference provenance was fetched from Mapillary and
  retained privately with source links, creator names and capture timestamps.
- Selected 30 queries from 27 separate capture sequences and 11 creators using
  seed 42, at least 100m query spacing and at most three queries per sequence.
  All 30 downloaded candidates passed exact-byte and perceptual-hash screening
  against local raw references and previously accepted queries. The spacing rule
  rejected 48 metadata candidates. Query photos occupy approximately 7.6 MiB.
- Selection and the query manifest were frozen before any predictions. There is
  no overlap with reference sequences. Query SHA256 hashes were rechecked before
  inference; full reference-index fingerprints matched before and after both
  acquisition and evaluation. No reference vectors/payloads were changed.
- Called the running real `/predict` API, respecting its request rate. No
  synthetic predictions, training, GPU rental, calibration or model changes.

## Results

| Measure | Result |
| --- | --- |
| Query count | 30 street photos |
| Median averaged-pin error | 1,901.77 m |
| Averaged pin within 100m | 0/30 (0%) |
| Averaged pin within 500m | 5/30 (16.67%) |
| Averaged pin within 2km | 18/30 (60%) |
| Any top-five candidate within 100m | 1/30 (3.33%) |
| Any top-five candidate within 500m | 7/30 (23.33%) |
| Top-ranked candidate within 500m | 2/30 (6.67%) |
| Median top-ranked candidate error | 2,202.63 m |
| Median best-of-five candidate error | 800.84 m |
| Median distance to nearest indexed reference | 248.66 m |
| Queries with any reference within 100m | 5/30 (16.67%) |
| Queries with any reference within 500m | 29/30 (96.67%) |

The current pipeline has weak location accuracy on this sample. Sparse reference
coverage limits 100m recall: only five query locations have an indexed reference
within that radius. Coverage alone does not explain the 500m results: 29 queries
have a reference within 500m, but retrieval returns one for only seven queries.
Retrieval/ranking needs improvement alongside denser references. The averaged pin
also requires evaluation against alternative ways of presenting the candidates.

Do not infer general-photo, window or elevated-view accuracy from this test.
Mapillary GPS labels were not manually verified; queries and references share a
provider, some queries share a capture sequence, and perceptual hashes cannot
prove complete visual independence. Thirty queries provide limited evidence and
do not justify calibrated confidence or precise performance claims.

## Artifacts and verification

Private local artifacts are under ignored
`swespy/data/mapillary_eval/20261106T095121.923253Z/`:

- `manifest.json`: frozen attributed query selection and exclusion provenance.
- `photos/`: the 30 original query images, separate from reference ingestion.
- `report-20261106T095433.665610Z.json`: metrics, individual predictions, coverage,
  index identity, manifest/evaluator/inference-source hashes and cached CLIP revision.
- `report-20261106T095433.665610Z.html`: interactive camera/prediction/candidate map
  and attributed query-photo cards. Map resources require internet.

Headless Chromium verified all 30 photo cards and the initialized map; a screenshot
was inspected. The final container regression suite passed **43 tests**, with the
independent private-photo snapshot test skipped. New evaluator/tests pass Ruff
0.9.1. Only evaluation Python sources are packaged in the Docker image; query
photos, provenance and reports remain in mounted ignored storage.

Repeat the frozen test using the command in
[`swespy/eval/README.md`](../swespy/eval/README.md). It writes a new timestamped
report and refuses a changed reference index. This run did not create or replace
`eval/results/baseline.json`, snapshots or user-owned evaluation photos.

Mapillary imagery attribution is retained per photo; the report identifies its
resized previews and links to the source and CC-BY-SA license. Source guidance:
https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data.
