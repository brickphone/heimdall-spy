// A compact MapLibre style over OpenFreeMap's OpenMapTiles vector tiles (no API key).
// Every colour is read from the design tokens in globals.css, so the map and the UI
// share one source. Glyphs must come from the tile host, so map labels use Noto Sans
// (the UI font cannot be used inside the map canvas).
import type { StyleSpecification, ExpressionSpecification } from "maplibre-gl";

export const OPENFREEMAP_TILES = "https://tiles.openfreemap.org/planet";
// One attribution line, set on the tile source so it replaces the source's own:
// OSM (ODbL), OpenFreeMap (host) and OpenMapTiles (schema, CC BY 4.0) are all credited.
export const ATTRIBUTION =
  '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>, ' +
  '<a href="https://openfreemap.org">OpenFreeMap</a>, ' +
  '© <a href="https://www.openmaptiles.org/">OpenMapTiles</a>';

export function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const px = (name: string): number => parseFloat(token(name));

const byZoom = (z1: number, v1: number, z2: number, v2: number): ExpressionSpecification => [
  "interpolate",
  ["exponential", 1.5],
  ["zoom"],
  z1,
  v1,
  z2,
  v2,
];

const MINOR = ["minor", "service"];
const MAJOR = ["primary", "secondary", "tertiary", "trunk", "motorway"];
const PATHS = ["path", "track"];

export function mapStyle(): StyleSpecification {
  const c = (n: string) => token(`--color-map-${n}`);
  const paper = token("--color-paper");
  const font = ["Noto Sans Regular"];
  return {
    version: 8,
    glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
    sources: {
      omt: { type: "vector", url: OPENFREEMAP_TILES, attribution: ATTRIBUTION },
    },
    layers: [
      { id: "land", type: "background", paint: { "background-color": c("land") } },
      {
        id: "landuse",
        type: "fill",
        source: "omt",
        "source-layer": "landuse",
        filter: ["match", ["get", "class"], ["residential", "suburb", "neighbourhood", "commercial", "industrial", "retail"], true, false],
        paint: { "fill-color": c("landuse") },
      },
      {
        id: "forest",
        type: "fill",
        source: "omt",
        "source-layer": "landcover",
        filter: ["match", ["get", "class"], ["wood", "forest"], true, false],
        paint: { "fill-color": c("forest") },
      },
      {
        id: "park",
        type: "fill",
        source: "omt",
        "source-layer": "park",
        paint: { "fill-color": c("park") },
      },
      {
        id: "grass",
        type: "fill",
        source: "omt",
        "source-layer": "landcover",
        filter: ["match", ["get", "class"], ["grass"], true, false],
        paint: { "fill-color": c("park") },
      },
      {
        id: "water",
        type: "fill",
        source: "omt",
        "source-layer": "water",
        paint: { "fill-color": c("water") },
      },
      {
        id: "waterway",
        type: "line",
        source: "omt",
        "source-layer": "waterway",
        paint: { "line-color": c("water"), "line-width": byZoom(12, 1, 17, 4) },
      },
      {
        id: "building",
        type: "fill",
        source: "omt",
        "source-layer": "building",
        minzoom: 13,
        maxzoom: 15,
        paint: { "fill-color": c("building"), "fill-outline-color": c("building-line") },
      },
      // 3D buildings from OpenFreeMap heights, only at zoom >= 15. MapLibre shades one
      // colour per extrusion, so walls use the side colour and a thin cap on top of each
      // building carries the roof colour.
      {
        id: "building-3d",
        type: "fill-extrusion",
        source: "omt",
        "source-layer": "building",
        minzoom: 15,
        paint: {
          "fill-extrusion-color": c("building-side"),
          "fill-extrusion-height": ["coalesce", ["get", "render_height"], 6],
          "fill-extrusion-base": ["coalesce", ["get", "render_min_height"], 0],
          "fill-extrusion-vertical-gradient": false,
        },
      },
      {
        id: "building-3d-roof",
        type: "fill-extrusion",
        source: "omt",
        "source-layer": "building",
        minzoom: 15,
        paint: {
          "fill-extrusion-color": c("building-top"),
          "fill-extrusion-height": ["+", ["coalesce", ["get", "render_height"], 6], 0.3],
          "fill-extrusion-base": ["coalesce", ["get", "render_height"], 6],
          "fill-extrusion-vertical-gradient": false,
        },
      },
      {
        id: "path",
        type: "line",
        source: "omt",
        "source-layer": "transportation",
        filter: ["match", ["get", "class"], PATHS, true, false],
        minzoom: 13,
        paint: { "line-color": c("path"), "line-width": 0.8, "line-dasharray": [3, 2] },
      },
      {
        id: "minor-casing",
        type: "line",
        source: "omt",
        "source-layer": "transportation",
        filter: ["match", ["get", "class"], MINOR, true, false],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": c("minor-casing"), "line-width": byZoom(12, 1.5, 17, 9) },
      },
      {
        id: "minor",
        type: "line",
        source: "omt",
        "source-layer": "transportation",
        filter: ["match", ["get", "class"], MINOR, true, false],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": c("road"), "line-width": byZoom(12, 0.6, 17, 7) },
      },
      {
        id: "major-casing",
        type: "line",
        source: "omt",
        "source-layer": "transportation",
        filter: ["match", ["get", "class"], MAJOR, true, false],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": c("major-casing"), "line-width": byZoom(10, 2, 17, 15) },
      },
      {
        id: "major",
        type: "line",
        source: "omt",
        "source-layer": "transportation",
        filter: ["match", ["get", "class"], MAJOR, true, false],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": c("road"), "line-width": byZoom(10, 1, 17, 12) },
      },
      {
        id: "rail",
        type: "line",
        source: "omt",
        "source-layer": "transportation",
        filter: ["match", ["get", "class"], ["rail", "transit"], true, false],
        paint: { "line-color": c("rail"), "line-width": 1 },
      },
      {
        id: "rail-ticks",
        type: "line",
        source: "omt",
        "source-layer": "transportation",
        filter: ["match", ["get", "class"], ["rail", "transit"], true, false],
        minzoom: 13,
        paint: { "line-color": c("rail"), "line-width": 5, "line-dasharray": [0.2, 3] },
      },
      {
        id: "street-names",
        type: "symbol",
        source: "omt",
        "source-layer": "transportation_name",
        minzoom: 14,
        layout: {
          "symbol-placement": "line",
          "text-field": ["get", "name"],
          "text-font": font,
          "text-size": px("--text-map"),
          "text-max-angle": 30,
        },
        paint: {
          "text-color": c("street-label"),
          "text-halo-color": c("road"),
          "text-halo-width": 1.5,
        },
      },
      {
        id: "water-names",
        type: "symbol",
        source: "omt",
        "source-layer": "water_name",
        layout: {
          "text-field": ["get", "name"],
          "text-font": ["Noto Sans Italic"],
          "text-size": px("--text-xs"),
        },
        paint: {
          "text-color": c("water-label"),
          "text-halo-color": paper, // Read against paper, so the label meets 4.5:1.
          "text-halo-width": 1.5,
        },
      },
      {
        id: "district-names",
        type: "symbol",
        source: "omt",
        "source-layer": "place",
        filter: ["match", ["get", "class"], ["suburb", "quarter", "neighbourhood", "village", "town", "city"], true, false],
        layout: {
          "text-field": ["get", "name"],
          "text-font": font,
          "text-size": px("--text-body"),
          "text-padding": 6,
        },
        paint: {
          "text-color": c("label"),
          "text-halo-color": paper,
          "text-halo-width": px("--halo-district"),
        },
      },
    ],
  };
}
