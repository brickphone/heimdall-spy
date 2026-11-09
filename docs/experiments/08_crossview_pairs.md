# Cross-view pairs and re-ranking speed (Parts 1–2)

Date: 2026-11-09. Dev set only (66 photos); test set untouched. Parts 3–4 await the
user's decision.

## Part 1: re-ranking speed

The weight (0.005) and threshold (25 inliers) are unchanged from Part 5. K = 10 and 20
reuse the first candidates of the K = 50 inliers, because MegaLoc's order does not
change. The 512-keypoint settings were recomputed for the top 20.

| K | Keypoints | Within 100 / 500 m | ok: n (≤500 m) | low: n (≤500 m) | Rule threshold | MPS cold | MPS warm* |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 50 | 1024 | 27 / 31 | 24 (92%) | 42 (21%) | 25 | 32.6 s | 14.4 s |
| 20 | 1024 | 27 / 31 | 24 (92%) | 42 (21%) | 25 | 16.3 s | 7.8 s |
| 10 | 1024 | 27 / 31 | 24 (92%) | 42 (21%) | 25 | 7.7 s | 6.1 s |
| **5** | **1024** | **27 / 31** | **24 (92%)** | **42 (21%)** | **25** | **4.0 s** | **3.4 s** |
| 20 | 512 | 27 / 31 | 18 (100%) | 48 (27%) | 20 | 9.3 s | 3.7 s |
| 10 | 512 | 27 / 31 | 18 (100%) | 48 (27%) | 20 | 4.9 s | 1.8 s |

\*Warm = reference features already stored on disk; cold = extracted per query.
Median over 6 dev photos, MegaLoc on MPS, ALIKED on MPS, LightGlue on CPU.

- All of the re-ranking gain is in the top 5 candidates; larger K adds only latency.
- 512 keypoints lowers inlier counts, so fewer photos pass the confidence threshold.
  At 25 inliers: 18 ok / 48 low (27% of low still correct). With the rule's
  recalibrated 20 inliers: 17 ok (94%) / 49 low (30.6%). Either way the split shifts
  by 6–10 points.
- **Default: K = 5, 1,024 keypoints.** 4.0 s cold, identical accuracy and identical
  split, no recalibration needed. Config `rerank_k = 5`; the retrieval depth for
  diagnostics stays 50.
- **API equals eval, with re-ranking on, on 10 dev photos:**
  `tests/integration/test_api_rerank_matches_eval.py` **passed**. Top-1, top-5 set,
  confidence state and predicted location are identical; inliers agree within ±3.

## Part 2: feasibility (metadata only)

### Training cities

Polygons come from SCB "Statistiska tätorter 2023" (`stat:Tatorter_2023`, CC0),
simplified with Douglas-Peucker at 100 m (area change < 1%), and are stored in
`swespy/swespy/cities.py`. Config refuses to load if any polygon overlaps the
Linköping polygon (unit-tested). The 1 km dev/test exclusion guard removed 0 images
(dev reported; test sealed, counts only).

### Viewpoint rules (fixed before review)

| Class | Rule (order of application) |
| --- | --- |
| panorama | `is_pano` or camera type spherical/equirectangular |
| unknown | < 5 timed steps between frames (gaps > 60 s ignored) |
| car | median speed > 7 m/s (25 km/h) or 90th percentile > 12 m/s |
| bike | 2.5 < median ≤ 7 m/s |
| pedestrian | median ≤ 2.5 m/s |

Speeds come from consecutive frames in the Mapillary vector tiles (capture time +
position). Camera make, model and type come from one Graph API lookup per sequence.

### Reliability check (50 random sequences per class, one 256 px thumbnail each)

Only review thumbnails were fetched, into contact sheets (`data/crossview/review_*.jpg`);
no training images were downloaded.

| Class | Correct for car vs non-car | Notes |
| --- | --- | --- |
| car | 47/50 (94%) | the 3 errors are a train and bus/tram window views; all 50 are vehicle-mounted |
| bike | 47/50 non-car (94%) | 3 look car-mounted; bike vs walking not separable from thumbnails |
| pedestrian | 48/50 non-car (96%) | pavements, parks, people, photographer's shadow |
| panorama | 50/50 | mixed platforms: car roof, helmet, handheld |

### Counts per city (inside SCB urban areas)

| City | Images | Car | Bike | Pedestrian | Panorama | Unknown | Commons (allowed) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Norrköping | 78,921 | 30,811 | 21,764 | 24,820 | 692 | 834 | 367 |
| Uppsala | 91,233 | 35,772 | 41,196 | 7,371 | 1,372 | 5,522 | 1,066 |
| Örebro | 106,848 | 87,423 | 15,404 | 3,351 | 6 | 664 | 777 |
| Västerås | 185,838 | 104,191 | 67,046 | 6,920 | 6,832 | 849 | 1,264 |
| Umeå | 104,564 | 49,741 | 38,306 | 4,931 | 10,106 | 1,480 | 478 |
| **Total** | **567,404** | **307,938** | **183,716** | **47,393** | **19,008** | **9,349** | **3,952** |

Sequences per class: car 2,158; bike 1,065; pedestrian 702; panorama 197; unknown
2,493. Commons decisions are recorded per city in `data/crossview/estimate.json`. Only
CC0, PD, CC BY and CC BY-SA files are allowed; for example, Uppsala excluded 275 NC/ND
files.

### Estimated cross-view pairs

A candidate is a non-car image with a car image from another sequence within 25 m and
a heading difference under 60°. Panoramas and Commons photos use distance only (no
reliable heading).

| City | Pedestrian | Bike | Panorama | Commons | Images with ≥ 1 car candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Norrköping | 5,074 | 8,370 | 494 | 111 | 14,049 |
| Uppsala | 1,787 | 10,970 | 1,372 | 260 | 14,389 |
| Örebro | 749 | 7,821 | 1 | 265 | 8,836 |
| Västerås | 851 | 16,915 | 3,763 | 380 | 21,909 |
| Umeå | 1,533 | 12,613 | 3,473 | 128 | 17,747 |
| **Total** | **9,994** | **56,689** | **9,103** | **1,144** | **76,930** |

These are candidates, before verification. If geometric verification keeps 10% / 30% /
50% of them (yield unknown until Part 3), that is about 7,700 / 23,000 / 38,000
verified pairs. The 5,000 target is reached even at 10%, and from pedestrian + Commons
alone (11,138 candidates) at about 45%.

### Diversity caveats

- Candidate non-car images come from 316 pedestrian sequences (41 creators), 811 bike
  sequences (37 creators) and 114 panorama sequences (9 creators).
- One creator supplies most candidates in some cities: Västerås 82%, Umeå 76%,
  Norrköping 61%, Örebro 47%, Uppsala 29%. Part 3 should cap any single creator's share
  of the verified pairs.
- "Bike" is the largest non-car class and mostly road- or bike-path-level. It is less
  like a handheld query than "pedestrian" and Commons, which are the closest match to
  the Linköping failures.
