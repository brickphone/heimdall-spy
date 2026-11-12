/**
 * Collision-free marker layout in screen space.
 *
 * The prediction pin stays at its true position. Each candidate square, in rank order,
 * keeps its true position if it overlaps nothing placed so far; otherwise it moves to the
 * first free slot on a ring around its true position (8 directions, then a wider ring),
 * and a leader line points back to the true spot. Overlap is measured as boxes of
 * ``size + GAP`` pixels, so markers never touch each other or the pin. ``blocked``
 * rectangles (the callout) are avoided the same way.
 */
export interface ScreenPoint {
  id: number;
  x: number;
  y: number;
  fixed?: boolean; // The prediction pin: never moved.
}

export interface Rect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

const GAP = 4;
const RINGS = [1.4, 2.6, 3.8]; // Slot distance, in marker sizes.
const ANGLES = [-45, 45, -135, 135, 0, 180, -90, 90].map((d) => (d * Math.PI) / 180);

const clear = (x: number, y: number, placed: { x: number; y: number }[], min: number, blocked: Rect[]) =>
  placed.every((p) => Math.max(Math.abs(p.x - x), Math.abs(p.y - y)) >= min) &&
  blocked.every((r) => x + min / 2 <= r.left || x - min / 2 >= r.right || y + min / 2 <= r.top || y - min / 2 >= r.bottom);

/** Offsets [dx, dy] in pixels per marker id. */
export function layoutMarkers(points: ScreenPoint[], size: number, blocked: Rect[] = []): Map<number, [number, number]> {
  const min = size + GAP;
  const out = new Map<number, [number, number]>();
  const placed: { x: number; y: number }[] = [];
  for (const p of points.filter((q) => q.fixed)) {
    placed.push({ x: p.x, y: p.y });
    out.set(p.id, [0, 0]);
  }
  for (const p of points.filter((q) => !q.fixed).sort((a, b) => a.id - b.id)) {
    let offset: [number, number] = [0, 0];
    if (!clear(p.x, p.y, placed, min, blocked)) {
      search: for (const r of RINGS) {
        for (const a of ANGLES) {
          const dx = Math.round(Math.cos(a) * r * size);
          const dy = Math.round(Math.sin(a) * r * size);
          if (clear(p.x + dx, p.y + dy, placed, min, blocked)) {
            offset = [dx, dy];
            break search;
          }
        }
      }
    }
    placed.push({ x: p.x + offset[0], y: p.y + offset[1] });
    out.set(p.id, offset);
  }
  return out;
}
