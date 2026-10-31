/**
 * Pure playback engine for the ops toolpath track.
 *
 * A track is an ordered list of segments; each carries the flat point
 * polyline of the /ops payload plus its cumulative distance range.
 * The engine is unit-agnostic: it operates on whatever 2D space the
 * polylines are expressed in (world mm for us) so it stays testable
 * without three.js or stores.
 */

export interface TrackSourceSegment {
  points: number[];
  power: number;
  cutting: boolean;
}

export interface TrackSegment extends TrackSourceSegment {
  startDistance: number;
  endDistance: number;
  /** Per-point cumulative distance inside the segment. */
  cumulative: number[];
}

export interface Track {
  segments: TrackSegment[];
  totalDistance: number;
}

export interface TrackSample {
  x: number;
  y: number;
  segmentIndex: number;
  power: number;
  cutting: boolean;
}

const EPSILON = 1e-6;
/** Fallback average speed when no time estimate is available (mm/s). */
export const DEFAULT_RATE_MM_S = 25;

export function buildTrack(segments: TrackSourceSegment[]): Track {
  const out: TrackSegment[] = [];
  let total = 0;
  for (const segment of segments) {
    const cumulative: number[] = [0];
    let length = 0;
    for (let i = 2; i + 1 < segment.points.length; i += 2) {
      length += Math.hypot(
        segment.points[i]! - segment.points[i - 2]!,
        segment.points[i + 1]! - segment.points[i - 1]!,
      );
      cumulative.push(length);
    }
    const startDistance = total;
    total += length;
    out.push({
      ...segment,
      startDistance,
      endDistance: total,
      cumulative,
    });
  }
  return { segments: out, totalDistance: total };
}

export function clampPosition(track: Track, position: number): number {
  return Math.min(Math.max(position, 0), track.totalDistance);
}

function segmentAt(track: Track, position: number): TrackSegment | null {
  if (track.segments.length === 0) return null;
  const pos = clampPosition(track, position);
  let low = 0;
  let high = track.segments.length - 1;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (track.segments[mid]!.endDistance >= pos - EPSILON) high = mid;
    else low = mid + 1;
  }
  return track.segments[low]!;
}

function pointOnSegment(
  segment: TrackSegment,
  distance: number,
): { x: number; y: number } {
  const points = segment.points;
  if (points.length < 4) {
    return { x: points[0] ?? 0, y: points[1] ?? 0 };
  }
  const local = Math.min(
    Math.max(distance - segment.startDistance, 0),
    segment.endDistance - segment.startDistance,
  );
  const cum = segment.cumulative;
  let low = 0;
  let high = cum.length - 2;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (cum[mid + 1]! >= local) high = mid;
    else low = mid + 1;
  }
  const span = cum[low + 1]! - cum[low]!;
  const t = span > EPSILON ? (local - cum[low]!) / span : 0;
  const ax = points[low * 2]!;
  const ay = points[low * 2 + 1]!;
  const bx = points[low * 2 + 2]!;
  const by = points[low * 2 + 3]!;
  return { x: ax + (bx - ax) * t, y: ay + (by - ay) * t };
}

export function sampleAt(track: Track, position: number): TrackSample | null {
  const segment = segmentAt(track, position);
  if (!segment) return null;
  const point = pointOnSegment(segment, clampPosition(track, position));
  return {
    x: point.x,
    y: point.y,
    segmentIndex: track.segments.indexOf(segment),
    power: segment.power,
    cutting: segment.cutting,
  };
}

/** End distance of the first non-degenerate segment after `position`. */
export function nextSegmentBoundary(track: Track, position: number): number {
  let best = track.totalDistance;
  for (const segment of track.segments) {
    if (segment.endDistance - segment.startDistance <= EPSILON) continue;
    if (
      segment.endDistance > position + EPSILON &&
      segment.endDistance < best
    ) {
      best = segment.endDistance;
    }
  }
  return best;
}

/** Start of the current non-degenerate segment (previous one if already there). */
export function prevSegmentBoundary(track: Track, position: number): number {
  let best = 0;
  for (const segment of track.segments) {
    if (segment.endDistance - segment.startDistance <= EPSILON) continue;
    if (
      segment.startDistance < position - EPSILON &&
      segment.startDistance > best
    ) {
      best = segment.startDistance;
    }
  }
  return best;
}

export function advancePosition(
  track: Track,
  position: number,
  deltaMm: number,
): number {
  return clampPosition(track, position + deltaMm);
}

/**
 * Distance travelled in `dtSeconds` at the given speed multiplier.
 * 1x plays back at the average rate implied by the time estimate, so
 * a full pass takes about estimated_seconds at 1x.
 */
export function playbackDeltaMm(
  track: Track,
  speed: number,
  dtSeconds: number,
  estimatedSeconds: number | null,
): number {
  const rate =
    estimatedSeconds !== null && estimatedSeconds > 0
      ? track.totalDistance / estimatedSeconds
      : DEFAULT_RATE_MM_S;
  return rate * Math.max(speed, 0) * Math.max(dtSeconds, 0);
}
