export interface LatLng {
  lat: number;
  lng: number;
}

const EARTH_M = 6_371_000;

export function distanceM(a: LatLng, b: LatLng): number {
  const p1 = (a.lat * Math.PI) / 180;
  const p2 = (b.lat * Math.PI) / 180;
  const dp = p2 - p1;
  const dl = ((b.lng - a.lng) * Math.PI) / 180;
  const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * EARTH_M * Math.asin(Math.sqrt(h));
}

/** "40 m", "850 m", "2.4 km". */
export function formatDistance(m: number): string {
  if (m < 1000) return `${Math.round(m / 10) * 10 || Math.round(m)} m`;
  return `${(m / 1000).toFixed(m < 10_000 ? 1 : 0)} km`;
}

/** "58.40905° N, 15.57135° E" */
export function formatCoords(p: LatLng): string {
  const ns = p.lat >= 0 ? "N" : "S";
  const ew = p.lng >= 0 ? "E" : "W";
  return `${Math.abs(p.lat).toFixed(5)}° ${ns}, ${Math.abs(p.lng).toFixed(5)}° ${ew}`;
}

/** Plain "58.40905, 15.57135" for the clipboard. */
export function plainCoords(p: LatLng): string {
  return `${p.lat.toFixed(5)}, ${p.lng.toFixed(5)}`;
}

/** Metres per screen pixel at a zoom level (Web Mercator, 512 px tiles as in MapLibre). */
export function metresPerPixel(lat: number, zoom: number): number {
  return (40_075_016.686 * Math.cos((lat * Math.PI) / 180)) / (512 * 2 ** zoom);
}

/** A closed ring of ``n`` points approximating a circle of ``radiusM`` metres. */
export function circle(center: LatLng, radiusM: number, n = 72): [number, number][] {
  const out: [number, number][] = [];
  const dLat = radiusM / 111_320;
  const dLng = radiusM / (111_320 * Math.cos((center.lat * Math.PI) / 180));
  for (let i = 0; i <= n; i++) {
    const t = (2 * Math.PI * i) / n;
    out.push([center.lng + dLng * Math.cos(t), center.lat + dLat * Math.sin(t)]);
  }
  return out;
}
