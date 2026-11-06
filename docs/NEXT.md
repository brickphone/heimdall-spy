# Deferred work and design gaps

2026-11-06 update: [STREET_CHECK.md](STREET_CHECK.md) records the first automated
same-provider street sanity check. Median pin error is 1.90km on 30 queries;
coverage and retrieval both need improvement. Independent multi-viewpoint
evaluation and calibrated confidence remain pending. The notes below are the
historical preliminary plan.

2026-10-16: preliminary notes from Part 1, not the completed Part 5 proposal.
See [AUDIT.md](AUDIT.md). No baseline photos/labels, deployed index or evaluation
numbers are available. No model choice can yet be justified by measured accuracy.

## Coarse-to-fine localisation

There is no online coarse region stage. Pretrained CLIP + Qdrant is the sole
retrieval stage; S2 level 14 labels are stored metadata. The separate training
classifier is disconnected from indexing/API and has a held-out-label validation
fault. Do not wire it in, change S2 level, replace CLIP or alter Qdrant schema now.
Current ingestion defaults cover Sweden; actual index coverage must be inspected
before baseline evaluation. Keep future work limited to Linköping.

After the frozen retrieval baseline, compare an independently pretrained
vision-language coarse localiser followed by the same retrieval against retrieval
alone. Measure coarse-stage failure propagation, latency and accuracy by viewpoint
before deciding whether it beats fine-tuning. A local city reference database
cannot establish reliable country/global coverage or justify precise pins for
out-of-coverage queries. GeoSpy is product inspiration only: no branding,
proprietary methods/data, scraping, calls or claims of equivalent performance.

## Viewpoint evidence and future experiments

The window failure and car-perspective training distribution are user-reported;
the repository contains no measured viewpoint comparison. First collect the
user-owned fixed set with street/window/elevated/indoor_view labels, excluded
from public Git history and all reference/training ingestion. Record reference
index identity, model revision, environment and geographic top-k hit tolerance.
Separate calibration fitting from final evaluation, or use a documented
cross-validation approach if data is scarce; report bucket sample counts and
uncertainty rather than fitting thresholds to a tiny holdout and declaring success.

Then propose comparisons of crops/perspective/lighting augmentation against
additional genuinely elevated/window references. Synthetic transformations cannot
reveal buildings hidden behind street-level occlusions. Before acquiring any
source, document license/terms, redistribution and model-use permissions,
provenance and privacy constraints. No new images are downloaded in this task.
No named new dataset or license has yet been vetted.

A measured Part 5 proposal must include median metres, within 100m/500m/2km,
geographic top-k recall and calibration by viewpoint, tolerances, latency,
reference coverage and failure examples. It must compare retrieval-only,
augmentation, additional licensed references and coarse-to-fine options.
Initial planning allowance only: 1-2 engineer-days to design experiments after
baseline, 2-5 days for augmentation comparison, and 3-10 days for a coarse-stage
prototype; acquiring permissioned viewpoint data is not estimable yet. Hardware
costs depend on image count, GPU availability, throughput and provider rates;
no currency estimate or accuracy gain is defensible yet. There is no authorized
paid API or compute spend. Replace these provisional ranges with measured
throughput and actual quotes in Part 5.

## Honest response contract (Part 4)

Keep ranked alternatives with coordinates and clearly distinguish similarity
from calibrated confidence. Plan explicit low-confidence/out-of-coverage states,
including no matches, weak similarity and widely dispersed candidates. Avoid
presenting an averaged pin as certainty. Thresholds require baseline/calibration
evidence; do not invent them in the audit. No API/schema change has been made.

## Privacy and B2B deployment prerequisites

The route uses in-memory bytes but multipart spooling may write uploads to temp
disk. It also logs a stable image hash and predicted coordinates. Part 3 should
enforce bounded memory-only upload processing through the parsing layer and
remove image-identifying logs; merely deleting a later application save is
insufficient. Retention settings should govern approved operational metadata,
reference data and logs, with query-image persistence disabled. Check temporary
storage, reverse proxies, crash dumps and external log sinks as part of deployment.

Before B2B deployment, plan authenticated client identities, tenant separation,
per-client quotas/rate limits using shared storage, access control and a
retention-limited security audit log of who requested what operation and when.
Do not put uploaded images, content hashes or precise inferred coordinates into
those logs by default. Define abuse review/reporting, permitted uses, credential
revocation and operator access policies. Provide no public upload page; restrict
interactive API docs and direct API/Qdrant access. Existing per-IP SlowAPI
middleware and request logs do not satisfy these requirements. These protections
are recorded only and are not built in this task.
