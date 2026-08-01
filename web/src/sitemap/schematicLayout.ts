import type { CameraStatus } from '@/api/engineClient'
import type { NormalizedPoint } from '@/sitemap/placement'
import { clampPoint } from '@/sitemap/placement'

/**
 * The empty-floorplan state: with no operator-supplied image, a zone-grouped
 * schematic beats a blank rectangle demanding every camera be dragged from
 * nowhere. Zones (`room` | `corridor` | `dayroom`) already exist on
 * `GET /cameras` for exactly this — this module is the only place that turns
 * them into a layout, so `FloorplanStage` never has to know the grouping
 * rule itself.
 */

type ZoneName = 'room' | 'corridor' | 'dayroom'

/** Fixed display order — stable across renders regardless of camera list order. */
const ZONE_ORDER: readonly ZoneName[] = ['room', 'corridor', 'dayroom']

const ZONE_LABELS: Record<ZoneName, string> = {
  room: 'Rooms',
  corridor: 'Corridors',
  dayroom: 'Dayrooms',
}

/** A camera with `zone: null` is ungrouped, not broken — see `CameraStatus.zone`'s own doc comment. It gets its own trailing band rather than being folded into a real zone or dropped. */
export const UNGROUPED_BAND_KEY = 'ungrouped'
export const UNGROUPED_BAND_LABEL = 'Ungrouped'

export interface SchematicBand {
  key: string
  label: string
  cameraIds: string[]
}

/**
 * Groups camera ids into bands by `zone`, in the fixed order above, then an
 * ungrouped band last (present only if at least one camera actually has no
 * zone). A zone with no cameras produces no band — nothing renders an empty
 * "Corridors" strip just because the enum has three members.
 */
export function buildSchematicBands(cameras: readonly CameraStatus[]): SchematicBand[] {
  const byKey = new Map<string, string[]>()
  for (const camera of cameras) {
    const key = camera.zone ?? UNGROUPED_BAND_KEY
    const list = byKey.get(key)
    if (list) list.push(camera.camera_id)
    else byKey.set(key, [camera.camera_id])
  }

  const bands: SchematicBand[] = []
  for (const zone of ZONE_ORDER) {
    const cameraIds = byKey.get(zone)
    if (cameraIds && cameraIds.length > 0) {
      bands.push({ key: zone, label: ZONE_LABELS[zone], cameraIds })
    }
  }
  const ungrouped = byKey.get(UNGROUPED_BAND_KEY)
  if (ungrouped && ungrouped.length > 0) {
    bands.push({ key: UNGROUPED_BAND_KEY, label: UNGROUPED_BAND_LABEL, cameraIds: ungrouped })
  }
  return bands
}

/**
 * A default normalized position for every camera: one horizontal band per
 * zone (stacked top to bottom in `ZONE_ORDER`, ungrouped last), cameras
 * spread evenly left to right within their band. Never returns a point
 * exactly on an edge (0 or 1) so a marker is never clipped by the stage
 * border.
 */
export function buildSchematicPositions(
  cameras: readonly CameraStatus[],
): Map<string, NormalizedPoint> {
  const bands = buildSchematicBands(cameras)
  const positions = new Map<string, NormalizedPoint>()
  if (bands.length === 0) return positions

  const bandHeight = 1 / bands.length
  bands.forEach((band, bandIndex) => {
    const y = bandHeight * (bandIndex + 0.5)
    const columnWidth = 1 / band.cameraIds.length
    band.cameraIds.forEach((cameraId, cameraIndex) => {
      const x = columnWidth * (cameraIndex + 0.5)
      positions.set(cameraId, clampPoint({ x, y }))
    })
  })
  return positions
}

export interface EffectivePlacement {
  point: NormalizedPoint
  /** True when this camera has never been explicitly placed, so `point` is the schematic fallback rather than something the operator dragged. */
  isDefault: boolean
}

/**
 * The position `FloorplanStage` actually renders a camera at: the operator's
 * own stored placement if one exists, otherwise the schematic default —
 * computed the SAME way whether or not a floorplan image is set, so a camera
 * nobody has placed yet on a real image is still visible and clickable
 * (with `isDefault: true` so the UI can mark it as not yet placed) rather
 * than silently missing from the map.
 */
export function resolveEffectivePlacements(
  cameras: readonly CameraStatus[],
  stored: Readonly<Record<string, NormalizedPoint>>,
): Map<string, EffectivePlacement> {
  const fallback = buildSchematicPositions(cameras)
  const resolved = new Map<string, EffectivePlacement>()
  for (const camera of cameras) {
    const explicit = stored[camera.camera_id]
    if (explicit) {
      resolved.set(camera.camera_id, { point: clampPoint(explicit), isDefault: false })
    } else {
      resolved.set(camera.camera_id, {
        point: fallback.get(camera.camera_id) ?? { x: 0.5, y: 0.5 },
        isDefault: true,
      })
    }
  }
  return resolved
}
