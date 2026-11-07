"""Shared evaluation helpers: provenance-safe writes, duplicate screening and reports."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"


def write_new(path: Path, data: dict[str, Any]) -> None:
    """Write a new JSON artifact without replacing an existing baseline or snapshot."""
    serialized = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def record(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Require explicit sequence, attribution and finite coordinates; fail closed."""
    try:
        image_id = str(raw["id"])
        sequence = raw["sequence"]
        creator = raw["creator"]["username"]
        lng, lat = raw["geometry"]["coordinates"]
        lat, lng = float(lat), float(lng)
        captured_at = int(raw["captured_at"])
        if not (
            image_id.isascii()
            and image_id.isdecimal()
            and isinstance(sequence, str)
            and sequence
            and isinstance(creator, str)
            and creator
            and math.isfinite(lat)
            and math.isfinite(lng)
            and -90 <= lat <= 90
            and -180 <= lng <= 180
            and captured_at > 0
        ):
            return None
        return {
            "image_id": image_id,
            "sequence": sequence,
            "creator": creator,
            "captured_at": captured_at,
            "lat": lat,
            "lng": lng,
            "source_url": f"https://www.mapillary.com/app/?pKey={image_id}&focus=photo",
            "license_url": LICENSE,
            "thumb_url": raw.get("thumb_2048_url"),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def signature(path: Path) -> dict[str, Any]:
    """Hash exact bytes and low-frequency structure for conservative duplicate screening."""
    contents = path.read_bytes()
    with Image.open(io.BytesIO(contents)) as original:
        image = ImageOps.exif_transpose(original).convert("L")
        small = np.asarray(image.resize((9, 8), Image.Resampling.LANCZOS))
        dhash_bits = (small[:, 1:] > small[:, :-1]).ravel()
        pixels = np.asarray(
            image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
        )
    basis = np.cos(np.pi / 32 * (np.arange(32) + 0.5)[None, :] * np.arange(8)[:, None])
    dct = (basis @ pixels @ basis.T).ravel()[1:]  # Ignore brightness/DC coefficient.

    def bits(values: np.ndarray) -> int:
        result = 0
        for bit in values:
            result = (result << 1) | int(bit)
        return result

    return {
        "sha256": hashlib.sha256(contents).hexdigest(),
        "dhash": bits(dhash_bits),
        "phash": bits(dct > np.median(dct)),
    }


def duplicate(sig: dict[str, Any], others: list[dict[str, Any]]) -> bool:
    """Reject exact copies or close perceptual hashes; this is not a proof of independence."""
    return any(
        sig["sha256"] == other["sha256"]
        or (sig["phash"] ^ other["phash"]).bit_count() <= 6
        or (sig["dhash"] ^ other["dhash"]).bit_count() <= 5
        for other in others
    )


def thumbnail(path: Path) -> str:
    """Embed a small preview so the local report does not require an image server."""
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((360, 220))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def render_map(path: Path, report: dict[str, Any]) -> None:
    """Write a local interactive Leaflet report with attribution and a table fallback."""
    payload = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Linköping street retrieval check</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{font:15px system-ui;margin:24px;color:#14213d;background:#f7f8fa}
h1{font-size:25px}#map{height:510px;border-radius:12px;margin:20px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:16px}
article{background:white;padding:16px;border:1px solid #ddd;border-radius:12px}
img{width:100%;height:180px;object-fit:contain;background:#eef}button{padding:8px;cursor:pointer}
.muted{color:#526174}a{color:#165ca3}.badge{font-weight:600}.legend{margin:12px 0}
</style><h1 id="title">Linköping street retrieval check</h1><p id="summary"></p>
<p class="muted" id="context-note">Mapillary sequence holdout · street-level only ·
similarity is uncalibrated.
Same-provider results may be optimistic. Hash screening cannot prove independence.</p>
<div class="legend">🟢 Camera label &nbsp; 🔴 <span id="prediction-label">Averaged prediction</span>
&nbsp; 🔵 Ranked candidates</div>
<div id="map"></div><div class="grid" id="cards"></div>
<p class="muted" id="attribution-note">Previews are resized Mapillary images,
attributed individually under CC BY-SA 4.0.
The map needs internet for Leaflet and OpenStreetMap tiles; photo cards remain self-contained.</p>
<script id="report" type="application/json">REPORT_DATA</script>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const r=JSON.parse(document.getElementById('report').textContent),m=r.metrics.overall;
if(r.model_label){document.getElementById('title').textContent=r.model_label;
 document.title=r.model_label;}
if(r.prediction_label)document.getElementById('prediction-label').textContent=r.prediction_label;
if(r.context_note)document.getElementById('context-note').textContent=r.context_note;
if(r.attribution_note)document.getElementById('attribution-note').textContent=
 r.attribution_note+' Map tiles and library require internet.';
document.getElementById('summary').textContent=`${m.count} queries from
${r.query_sequence_count} ${r.query_group_label ?? 'sequences'}.
Median pin error: ${m.median_error_m.toFixed(0)} m. Within 100 m: ${m.pct_within_100m.toFixed(1)}%.
Within 500 m: ${m.pct_within_500m.toFixed(1)}%.
Top-${r.top_k ?? 5} within ${r.hit_radius_m ?? 100} m: ${m.top_k_hit_pct.toFixed(1)}%.`;
const map=typeof L!=='undefined'?L.map('map'):null, bounds=[];
if(map)L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
attribution:'© OpenStreetMap contributors',maxZoom:19}).addTo(map);
const detail=map?L.layerGroup().addTo(map):null;
function pin(lat,lng,color,label,group){return L.circleMarker([lat,lng],{
radius:6,color,fillOpacity:.8}).addTo(group||map).bindTooltip(label)}
function focus(q){if(!map)return;detail.clearLayers();
 q.prediction.top_matches.forEach((c,i)=>pin(c.lat,c.lng,'#1269b0',`Candidate ${i+1}`,detail));
 L.circle([q.lat,q.lng],{radius:100,color:'#16834a',fillOpacity:.06}).addTo(detail);
 const pts=[[q.lat,q.lng],[q.prediction.predicted_lat,q.prediction.predicted_lng],
 ...q.prediction.top_matches.map(c=>[c.lat,c.lng])];map.fitBounds(pts,{padding:[35,35],maxZoom:17});
 document.getElementById('map').scrollIntoView({behavior:'smooth'});}
r.queries.forEach((q,i)=>{const p=q.prediction,actual=[q.lat,q.lng];
 const guess=[p.predicted_lat,p.predicted_lng];
 if(map){bounds.push(actual,guess);
 pin(...actual,'#16834a',`Query ${i+1}: camera`).on('click',()=>focus(q));
 pin(...guess,'#cf3030',`Query ${i+1}: prediction`).on('click',()=>focus(q));
 L.polyline([actual,guess],{color:'#999',weight:1}).addTo(map)}
 const card=document.createElement('article'),img=document.createElement('img');
 img.src=q.preview;img.alt=`Query ${i+1}`;card.append(img);
 const h=document.createElement('h3');
 h.textContent=`Query ${i+1} · ${q.error_m.toFixed(0)} m pin error`;
 card.append(h);const text=document.createElement('p');
 text.textContent=`Nearest reference: ${q.nearest_reference_m.toFixed(0)} m.
 Best of ${r.top_k ?? 5}: ${q.best_candidate_error_m?.toFixed(0) ?? 'n/a'} m.
 Top-ranked: ${q.top_1_error_m?.toFixed(0) ?? 'n/a'} m.
 Camera: ${q.lat.toFixed(6)}, ${q.lng.toFixed(6)}.
 Captured ${new Date(q.captured_at).toISOString().slice(0,10)}.`;
 card.append(text);const source=document.createElement('a');source.href=q.source_url;
 source.textContent=`${q.provider ?? 'Mapillary'} photo by ${q.creator}`;
 source.target='_blank';source.rel='noopener';
 card.append(source,document.createTextNode(' · '));const license=document.createElement('a');
 license.href=q.license_url;license.textContent=q.license_name ?? 'CC BY-SA 4.0';
 card.append(license);
 const button=document.createElement('button');button.textContent='Inspect candidates on map';
 button.onclick=()=>focus(q);card.append(document.createElement('p'),button);
 document.getElementById('cards').append(card)});
if(map&&bounds.length)map.fitBounds(bounds,{padding:[20,20]});
if(!map)document.getElementById('map').textContent=
 'Map library unavailable. Photo results are below.';
</script></html>"""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(page.replace("REPORT_DATA", payload))

