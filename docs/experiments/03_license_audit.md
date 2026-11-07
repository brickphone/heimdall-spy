# Part 3 — License and provenance audit

Date: 2026-11-07. Full table: [`docs/LICENSES.md`](../LICENSES.md).

## Decisions

- **Retrieval:** MegaLoc code and weights are MIT, so commercial use is allowed.
  Caveat: its training data includes non-commercial datasets (MSLS, ScanNet) and
  Street View-derived ones. The weights' MIT licence doesn't settle that, so get
  legal review before any commercial launch.
- **Re-ranking (Part 5):** ALIKED (BSD-3) + LightGlue (Apache-2.0, ALIKED weights)
  is the default. DISK, XFeat and OpenCV SIFT are allowed alternatives. **SuperPoint
  and SuperPoint-LightGlue are excluded** (Magic Leap non-commercial licence).
- **Privacy (Part 4.4):** EgoBlur face + licence-plate models (Apache-2.0) are the
  default. YuNet (MIT) is a faces-only fallback.
- **Data:** Mapillary (CC BY-SA 4.0, official API only, attribution + share-alike,
  re-identification safeguards), KartaView (CC BY-SA 4.0), Commons (CC0/PD/CC BY/
  CC BY-SA only; NC and ND excluded), self-collected walks.

## Provenance contract

`swespy/eval/provenance.py` requires every stored reference to carry `source`,
`source_id`, `license_url` (commercial allowlist), `captured_at`, `lat`, `lng` and a
`heading` key (`null` when unknown). Unit tests are in `tests/unit/test_provenance.py`.
All 3,388 current references pass when mapped from the legacy Mapillary rows.

## Open points for the user

1. Legal review of MegaLoc's training-data provenance before a commercial launch.
2. Mapillary bulk volume: the current terms allow commercial use through the
   official API and forbid unapproved scraping. Ask Mapillary first if the full
   polygon fetch turns out to be very large.
