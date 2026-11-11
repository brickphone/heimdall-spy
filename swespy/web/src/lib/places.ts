// Reverse geocoding from OpenStreetMap data we ship (public/data/places.json, ODbL):
// the nearest named street within STREET_M and the nearest district within DISTRICT_M.
import { type LatLng, distanceM } from "./geo";

interface PlacesFile {
  names: string[];
  streets: [number, number, number][];
  districts: [number, number, string, string][];
}

const STREET_M = 120;
const DISTRICT_M = 2500;
let cache: Promise<PlacesFile> | null = null;

export function loadPlaces(): Promise<PlacesFile> {
  cache ??= fetch("/data/places.json").then((r) => r.json() as Promise<PlacesFile>);
  return cache;
}

export interface Place {
  street: string | null;
  district: string | null;
}

export function lookup(places: PlacesFile, p: LatLng): Place {
  let street: string | null = null;
  let best = STREET_M;
  for (const [lat, lng, k] of places.streets) {
    if (Math.abs(lat - p.lat) > 0.002 || Math.abs(lng - p.lng) > 0.004) continue;
    const d = distanceM(p, { lat, lng });
    if (d < best) {
      best = d;
      street = places.names[k];
    }
  }
  let district: string | null = null;
  let bestD = DISTRICT_M;
  for (const [lat, lng, name, kind] of places.districts) {
    // Prefer named parts of town over hamlets at equal distance.
    const d = distanceM(p, { lat, lng }) * (kind === "hamlet" ? 1.5 : 1);
    if (d < bestD) {
      bestD = d;
      district = name;
    }
  }
  return { street, district };
}

/** "Rydsvägen, Ryd", "Ryd" or "Near Linköping". */
export function placeLabel(place: Place | null): string {
  if (!place) return "Locating…";
  if (place.street && place.district && place.street !== place.district) {
    return `${place.street}, ${place.district}`;
  }
  return place.street ?? place.district ?? "Near Linköping";
}
