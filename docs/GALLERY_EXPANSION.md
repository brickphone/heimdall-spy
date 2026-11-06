# Linköping gallery expansion — 6 November 2026

The city-wide reference acquisition completed with **3,388 photos**, including
all **369** original central references. It added **3,019** unique images; one
additional selected image was excluded as an exact duplicate. No downloads failed.

## Scope and coverage

- Bounding rectangle: west 15.50, south 58.34, east 15.75, north 58.48.
- 80 metadata tiles; 9,174 records examined and 9,173 unique candidates normalized.
- 305 capture sequences from 31 contributors.
- 143 occupied 1km UTM grid cells; 706 occupied 100m cells.
- Gallery directory: approximately 281 MiB, including metadata and hard-linked
  central images. Accepted JPEG bytes total 272,858,878.

An occupied cell only means at least one reference camera is present. This is a
bounded first pass across the wider urban area and surroundings, not complete or
continuous coverage of every street. Mapillary returned no pagination cursor for
the sampled bounding-box searches; the largest response was 200 images per tile.
Consequently the 600-record per-tile budget and 10,000-photo cap were not exhausted.

## Selection and evaluation separation

The acquisition preserves the original central reference bytes. New references
are selected by newest capture within 20m position / 45-degree heading bins, then
balanced round-robin across 1km cells. The 39 protected capture sequences from the
old query set and central reserved sequences are excluded. Additional references
are checked against protected query SHA256 and perceptual hashes; gallery exact
duplicates are removed deterministically. These safeguards do not establish
complete independence or rule out pretrained model overlap.

GPS coordinates, headings, capture times, image hashes, creators, source URLs and
license URLs are retained. Coordinates and headings remain provider labels and
have not been independently verified. Original test manifests and the CLIP
Qdrant index were verified unchanged; the API health check returned HTTP 200.

## Artifacts and resume

Ignored local dataset:
[`swespy/data/linkoping_dense/city_v1/`](../swespy/data/linkoping_dense/city_v1/).

- [Interactive coverage map](../swespy/data/linkoping_dense/city_v1/coverage.html):
  actual reference positions with attributed photo previews. Requires internet
  access for the Leaflet library and OpenStreetMap tiles.
- [GeoJSON](../swespy/data/linkoping_dense/city_v1/coverage.geojson).
- [Frozen manifest](../swespy/data/linkoping_dense/city_v1/manifest.json).

From `swespy/`, rerun `.venv-benchmark/bin/python -u -m eval.expand_linkoping` to
validate the completed gallery. Interrupted downloads reuse validated local
images and metadata tile checkpoints. Setup and selection details are in
[the evaluation README](../swespy/eval/README.md#city-wide-linkoping-reference-acquisition).

All accepted file hashes and coordinate-bearing paths were checked, with all
369 central reference hashes preserved and no protected capture sequences present.
The benchmark-environment suite passed **64 tests**, with one private snapshot
test skipped. Changed Python modules passed Ruff and mypy.

## Status at acquisition completion and follow-up

These images have **not yet been encoded into an expanded MegaLoc search index**.
No training or API model switch took place. Next, encode the frozen gallery with
the pinned MegaLoc model, measure retrieval against preserved queries, and add an
independent ordinary-photo test set. New accuracy results must come from that
evaluation; the earlier 11/18 central result is not a measurement of this gallery.

Follow-up completed: the gallery is now encoded and searchable through a separate
MegaLoc CLI. See [expanded-gallery evaluation](CITY_EVALUATION.md) for the actual
48-query results and remaining coverage/retrieval limitations. The API still uses
CLIP; no local model training has occurred.
