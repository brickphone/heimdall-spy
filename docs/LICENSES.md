# Licenses and provenance

Audited 2026-11-07 from the upstream license files and terms pages linked below.
This is an engineering audit, not legal advice. **Rule:** a component marked *No* or
*Unclear* for commercial use is not used in the default pipeline.

## Models and code

| Component | Role | License | Commercial use | Source |
| --- | --- | --- | --- | --- |
| MegaLoc code (pinned rev `5fe0dd6`) | Global retrieval model definition | MIT | Yes | [github.com/gmberton/MegaLoc](https://github.com/gmberton/MegaLoc) (`LICENSE` stored under `data/megaloc/assets/`) |
| MegaLoc weights (HF rev `a0f3472`) | Global retrieval weights | MIT (model card + `LICENSE`) | Yes, see caveat 1 | [huggingface.co/gberton/MegaLoc](https://huggingface.co/gberton/MegaLoc) |
| DINOv2 (MegaLoc backbone) | Backbone architecture/initialisation | Apache-2.0 | Yes | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) |
| SALAD reference implementation | Aggregation method MegaLoc uses | GPL-3.0 | Yes (copyleft on distribution); see caveat 2 | [serizba/salad](https://github.com/serizba/salad) |
| FAISS (`faiss-cpu`) | Exact vector search | MIT | Yes | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| PyTorch / torchvision | Inference runtime | BSD-3-Clause | Yes | [pytorch/pytorch](https://github.com/pytorch/pytorch) |

### Local features and matchers considered for Part 5

| Component | License | Commercial use | Decision | Source |
| --- | --- | --- | --- | --- |
| **LightGlue** code and weights (ALIKED/DISK variants) | Apache-2.0 | Yes | **Default matcher** (commit `5a650c4`; the package also ships `superpoint.py`, which is never imported) | [cvg/LightGlue](https://github.com/cvg/LightGlue) — README: "pre-trained weights of LightGlue and the code ... are released under the Apache-2.0 license" |
| **ALIKED** code and weights | BSD-3-Clause | Yes | **Default extractor** | [Shiaoming/ALIKED](https://github.com/Shiaoming/ALIKED) |
| DISK code and weights | Apache-2.0 | Yes | Allowed alternative | [cvlab-epfl/disk](https://github.com/cvlab-epfl/disk) |
| SuperPoint (original Magic Leap weights and `superpoint.py` inference file in LightGlue) | Magic Leap academic/non-profit non-commercial licence | **No** | **Excluded** | [magicleap/SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE) |
| SuperPoint + LightGlue weights | Inherit SuperPoint restriction | **No** | **Excluded** | LightGlue README (above) |
| XFeat | Apache-2.0 | Yes | Allowed alternative | [verlab/accelerated_features](https://github.com/verlab/accelerated_features) |
| SIFT (OpenCV) | Apache-2.0; SIFT patent expired 2020 | Yes | Allowed fallback | [opencv/opencv](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| OpenCV 4.10 (USAC-MAGSAC RANSAC used for inlier counting) | Apache-2.0 | Yes | **In use** | [opencv/opencv](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| kornia (LightGlue/DISK ports, RANSAC utilities) | Apache-2.0 | Yes | Allowed | [kornia/kornia](https://github.com/kornia/kornia) |
| glue-factory | Apache-2.0 | Yes | Not needed (training framework) | [cvg/glue-factory](https://github.com/cvg/glue-factory) |

### Upload decoding

| Component | License | Commercial use | Decision | Source |
| --- | --- | --- | --- | --- |
| Pillow | MIT-CMU (HPND) | Yes | In use | [python-pillow/Pillow](https://github.com/python-pillow/Pillow) |
| pillow-heif 1.8.1 (HEIC) | Source BSD-3; **binary wheels GPL-2.0** (bundled x265 encoder; libheif 1.23.6 and libde265 are LGPL-3.0) | Yes, but copyleft obligations if the wheel is distributed | **Optional extra `heic`**; installed locally for the demo, not in the default Docker image | [bigcat88/pillow_heif](https://github.com/bigcat88/pillow_heif) |
| pi-heif 1.4.0 | LGPL-3.0 | — | **Rejected:** discontinued; bundles libheif 1.23.0 with a heap-overflow CVE on crafted HEIC | [pi-heif](https://pypi.org/project/pi-heif/) |

### Demo web app (`swespy/web`)

| Component | License | Commercial use | Decision | Source |
| --- | --- | --- | --- | --- |
| Next.js 16.4.5, React 19.3 | MIT | Yes | In use | [vercel/next.js](https://github.com/vercel/next.js) |
| Tailwind CSS 4 | MIT | Yes | In use | [tailwindlabs/tailwindcss](https://github.com/tailwindlabs/tailwindcss) |
| Leaflet 1.9.4 | BSD-2-Clause | Yes | In use (plain Leaflet) | [Leaflet/Leaflet](https://github.com/Leaflet/Leaflet) |
| react-leaflet 5 | Hippocratic-2.1 (not OSI; use restrictions) | Unclear | **Rejected** | [PaulLeCam/react-leaflet](https://github.com/PaulLeCam/react-leaflet) |
| Geist fonts (via `next/font`, self-hosted at build) | SIL OFL 1.1 | Yes | In use | [vercel/geist-font](https://github.com/vercel/geist-font) |
| OpenStreetMap tiles | ODbL data; tile usage policy | Light use only | Demo only; attribution shown. Heavy or public use needs another tile provider | [OSM tile policy](https://operations.osmfoundation.org/policies/tiles/) |

### Privacy (face / licence-plate blurring for self-collected images)

| Component | License | Commercial use | Decision | Source |
| --- | --- | --- | --- | --- |
| **EgoBlur** face + licence-plate models | Apache-2.0 ("The model is licensed under the Apache 2.0 license") | Yes | **Default** (faces and plates) | [facebookresearch/EgoBlur](https://github.com/facebookresearch/EgoBlur) |
| YuNet face detector (OpenCV Zoo) | MIT | Yes | Allowed fallback (faces only) | [opencv_zoo/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| LPD-YuNet plate detector (OpenCV Zoo) | Apache-2.0 | Yes | Not used: trained on Chinese plates | [opencv_zoo/license_plate_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/license_plate_detection_yunet) |
| RetinaFace (biubug6 PyTorch) | MIT | Yes | Allowed alternative | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) |

## Data sources (Part 4)

| Source | Image license | Commercial use | Conditions | Source |
| --- | --- | --- | --- | --- |
| Mapillary (via official Graph API / vector tiles with a registered token) | CC BY-SA 4.0 | Yes | Attribute images (Mapillary logo + link to image page); share-alike for adapted images (panorama crops are adaptations); terms §5 forbid data mining/robots "not approved by Mapillary"; §12 requires safeguards against re-identification/unblurring and allows commercial use for developing products, services and datasets. See caveat 3 | [mapillary.com/terms](https://www.mapillary.com/terms), [licence help page](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data) |
| KartaView | CC BY-SA 4.0 | Yes | Credit "© Grab and KartaView Contributors"; share-alike. The terms page did not render for our fetcher; licence taken from KartaView/OSM documentation. Confirm API terms before bulk use | [kartaview.org/terms](https://kartaview.org/terms), [OSM wiki](https://wiki.openstreetmap.org/wiki/KartaView) |
| Wikimedia Commons | Per file | Only CC0, Public Domain Mark, CC BY, CC BY-SA | Exclude NC and ND files (crops/resizes are derivatives). Follow the Wikimedia User-Agent policy (already implemented: `HeimdallSpyResearchBot`) | [Commons licensing](https://commons.wikimedia.org/wiki/Commons:Licensing) |
| Self-collected walks | Owned by the user | Yes | Faces and plates blurred before storage; no private interiors | — |
| Flickr, Google Street View | — | — | Excluded (user decision / terms) | — |

## Mapillary terms checked

- **Version:** Mapillary Terms of Use, "Effective date: February 15, 2024", read on
  2026-11-07 at <https://www.mapillary.com/terms>.
- **§11 Developers:** register an app (client_id); the application should "materially
  supplement" Mapillary rather than merely redistribute content; Mapillary may
  throttle, revoke client_ids and monitor use.
- **§5:** no data mining, robots or extraction methods "not approved by Mapillary".
  We use only the official Graph API and vector tiles with our registered token,
  within the published rate limits and with backoff.
- **§12:** commercial use permitted for "improvement, training, and development of
  products, services, maps, studies, platforms, websites, applications, software,
  algorithms, datasets, solutions, or technologies"; requires safeguards against
  re-identification or unblurring and against inadvertent disclosure of Content.

## Attribution obligations for later display

When the app or any report shows a reference image, crop or data derived from it:

- **Mapillary (terms §11 + CC BY-SA 4.0):** show the Mapillary logo and link to the
  image page (`https://www.mapillary.com/app/?pKey=<id>`); name the contributor
  (`creator`); state "CC BY-SA 4.0" with a link to the licence; say whether the
  image was modified (panorama crops, resizing). Adapted images (e.g. panorama
  crops) that are shared must be released under CC BY-SA 4.0. Data integrated from
  street-level images: Mapillary logo + link to the Mapillary homepage.
- **KartaView (CC BY-SA 4.0):** "© Grab and KartaView Contributors", link to the
  photo and licence, indicate modifications, share-alike for adaptations.
- **Commons (CC BY / CC BY-SA):** author as given in the file's `Artist` field, link
  to the file page, licence name and link, indicate modifications; share-alike for
  CC BY-SA adaptations. CC0/PD: attribution courteous, not required.
- Every gallery row stores `source_url`, `creator` and `license_url` so these can be
  displayed; the API already returns them per match.

## Caveats

1. **MegaLoc training data.** MegaLoc was trained on GSV-Cities, Mapillary SLS,
   MegaScenes, ScanNet and SF-XL ([paper](https://arxiv.org/abs/2502.17237)). Some of
   these carry non-commercial or restrictive terms (Mapillary SLS: CC BY-NC-SA;
   ScanNet: non-commercial terms of use; GSV-Cities and SF-XL derive from Google
   Street View imagery). The released weights are MIT-licensed, and whether
   training-data terms restrict use of a trained model is legally unsettled. MegaLoc
   stays the default (a decision already made), but **get legal review before a
   commercial launch.**
2. **SALAD.** The SALAD reference repository is GPL-3.0. The pinned MegaLoc file is
   MIT-labelled and credits only OpenGlue (MIT) for adapted code. We use only that
   file. Even under GPL-3.0, commercial use is permitted; copyleft obligations would
   apply on distribution of the software, not on serving an API.
3. **Mapillary volume.** Fetching every image in the 158 km² polygon uses the
   official API within its rate limits. An earlier version of the terms (quoted by
   third parties) prohibited "mass downloads or bulk feeds"; the current text does
   not contain that wording. If the volume is large, ask Mapillary first.

## Provenance contract (every stored reference)

`swespy/eval/provenance.py` enforces: `source` (`mapillary` | `kartaview` |
`wikimedia_commons` | `self_collected`), `source_id`, `license_url` (commercial
allowlist), `captured_at`, `capture_year`, `lat`, `lng`, `location_source`, and
`heading` (key required; `null` when unknown). Coordinate rules per source
(Mapillary `computed_geometry` first, GPX interpolation limits, Commons tagged as
lower precision) are in [`GALLERY_V2_SPEC.md`](GALLERY_V2_SPEC.md). On 2026-11-07 all 3,388 existing references (Mapillary, CC BY-SA 4.0, all
with headings) passed once mapped with `from_legacy_mapillary`.
