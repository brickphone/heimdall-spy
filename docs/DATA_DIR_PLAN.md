# Plan: move `data/` to an external drive (`DATA_DIR`)

Status: **plan only, not started.** Nothing has been moved or changed.

Goal: all data can live at `/Volumes/<name>/heimdall-data`. The location is set by one
setting, `DATA_DIR`, and no code assumes `data/` sits inside the repo.

## Current state (surveyed 2026-11-09)

| Item | Finding |
| --- | --- |
| Size | 13 GB: `gallery_v2` 6.9, `megaloc` 3.2, `benchmarks` 1.3, `huggingface` 0.9, `crossview` 0.4, `linkoping_dense` 0.3, other < 0.3 |
| Hard-coded `data/...` in code | 25 files: `swespy/config.py` (3 settings), `swespy/model/megaloc.py` (HF cache), `swespy/data/pipeline.py`, 18 `eval/*.py` (module constants and argparse defaults), 2 opt-in integration tests, `scripts/*.sh`, `Dockerfile`, `docker-compose.yml` |
| Working-directory assumption | Every path is relative to the current directory, so everything only works when run from `swespy/` |
| Paths stored inside data files | 6 JSON files store `"data/..."` strings, among them **the gallery index** (`descriptor_dir: "data/megaloc/descriptors/<hash>"`, read by `locator.py:54`). Also `gallery-index-recent.json`, `walk-routes.json`, `city_eval_v1/manifest.json`, `recent_pilot_v1/protocol.json`, `eval/results/dev_historic_rerank.json`. One file in the sealed test-set folder also contains such a string (found by a file-name-only search; not opened) |
| Per-reference image paths | Already relative to the index file (`Gallery.root`), so they move with it |
| External drives attached now | None |

## Design

1. **One setting:** `Settings.data_dir` (environment variable `DATA_DIR`, or `.env`).
   - The default is `<project root>/data`, so nothing changes until it's set.
   - A relative value is resolved against the project root, not the current directory.
2. **One module, `swespy/paths.py`, is the only place that knows the layout:**
   - `data_dir()` returns the resolved root.
   - `data_path(*parts)` builds a path under it, e.g. `data_path("gallery_v2", "gallery-index-historic.json")`.
   - `resolve_stored(value)` maps a path read from a data file:
     - `"data/<rest>"` becomes `data_dir()/<rest>`;
     - absolute paths stay as they are;
     - anything else is resolved relative to the file it came from.
3. **Stored files are never rewritten.** The index checksums, frozen baselines and the
   sealed test set stay byte-identical; the mapping happens when files are read.
   - New files keep writing the logical form `"data/<rest>"`, so old and new files read the same way.
4. **Settings:**
   - `megaloc_root`, `gallery_index` and `field_photos_dir` become relative to `DATA_DIR`
     (e.g. `gallery_v2/gallery-index-historic.json`).
   - Old values starting with `data/` are still accepted through `resolve_stored`.
5. **Caches:**
   - The Hugging Face cache in `megaloc.py` moves to `data_path("huggingface", "hub")`.
   - `run_local.sh` sets `HF_HOME` and `TORCH_HOME` from `DATA_DIR`.
   - The Dockerfile already uses `/app/data/...` inside the container, which is unchanged.
6. **Eval scripts:** module constants and argparse defaults switch to `data_path(...)`.
   This is mechanical and doesn't change any logic.
7. **Fail fast if the drive isn't mounted:** if `DATA_DIR` is missing at startup, the API
   logs `DATA_DIR /Volumes/<name>/heimdall-data not found — is the drive mounted?` and
   `/ready` stays 503, instead of failing deep inside gallery loading. Eval scripts raise
   the same message.
8. **Docker:**
   - Compose mounts `${DATA_DIR:-./data}/...` read-only, as today.
   - Docker Desktop needs `/Volumes/<name>` added under Settings → Resources → File sharing.
9. **Tests:**
   - Unit tests for `resolve_stored` and for the setting (relative, absolute and missing directory).
   - An API test with `DATA_DIR` pointed at a temporary directory.
   - A guard test that fails if a literal `"data/"` appears in code outside `paths.py`
     (with a small allow-list for docs and comments).

## Drive requirements

- **Format: APFS.** Not exFAT: macOS writes `._name.jpg` AppleDouble files there, and
  several eval scripts glob `*.jpg`, so those files would be picked up as images. exFAT
  also has no Unix permissions.
- **SSD over USB-C or Thunderbolt.** Loading the gallery reads 91,714 small descriptor
  files (3.1 GB). On a spinning disk startup would take many minutes instead of about 60 s.
- **Space:** at least 100 GB, for 13 GB now plus growth (cross-view pairs, new galleries).
- **Turn off Spotlight indexing** on the data folder (`sudo mdutil -i off /Volumes/<name>`).
  Decide whether Time Machine should back it up; the gallery can be rebuilt, but the
  eval sets and field photos can't.

## Migration steps (after the code change is reviewed and committed by you)

1. Stop the servers.
2. Copy: `rsync -aH --info=progress2 data/ /Volumes/<name>/heimdall-data/`
3. Verify: `rsync -aHc --dry-run --itemize-changes data/ /Volumes/<name>/heimdall-data/`
   should list nothing.
   - This compares checksums only and reports counts.
   - The sealed test set is copied and checked as opaque bytes; no coordinates or images are read.
4. Set `DATA_DIR=/Volumes/<name>/heimdall-data` in `swespy/.env`.
5. Check:
   - unit tests;
   - `/ready`;
   - the opt-in API-equals-eval test (`SWESPY_RUN_REAL=1`, 10 dev photos);
   - one dev evaluation, whose numbers must be identical to `dev_historic_rerank.json`.
6. Keep the internal `data/` until you decide; rename it to `data.internal-backup` first
   rather than deleting it. `data/huggingface` and `data/benchmarks` are not deleted
   without your explicit go.

**Stopgap with no code change:** a symlink, `data -> /Volumes/<name>/heimdall-data`,
works today. It doesn't meet the goal, because the code still assumes `data/`, but it's
a way to free internal disk before the code change lands.

## Effort and risk

- **Effort:** about 25 files, mostly mechanical, plus `paths.py` and tests. Roughly half a day.
- **Main risk:** silently reading the wrong file in an eval script. This is mitigated by
  the guard test, the fail-fast check, and the identical-numbers check in step 5.

## Decisions needed from you

1. Drive name and whether it's APFS.
2. Should field photos (`field_photos`) also live on the drive, or stay internal?
3. Copy (keep both) or move (free about 13 GB internally) at the end?
