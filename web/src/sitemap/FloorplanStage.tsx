import { useRef } from 'react'
import type { CameraStatus } from '@/api/engineClient'
import type { Tone } from '@/lib/severity'
import { CameraMarker } from '@/sitemap/CameraMarker'
import { buildSchematicBands, resolveEffectivePlacements } from '@/sitemap/schematicLayout'
import type { NormalizedPoint, NudgeDirection, StageRect } from '@/sitemap/placement'
import { nudgePoint } from '@/sitemap/placement'

export interface FloorplanStageProps {
  cameras: readonly CameraStatus[]
  placements: Readonly<Record<string, NormalizedPoint>>
  toneByCameraId: ReadonlyMap<string, Tone>
  floorplanImage: string | null
  editable: boolean
  onPlace: (cameraId: string, point: NormalizedPoint) => void
}

/**
 * The plan itself: an operator-supplied floorplan image, or — with none —
 * the zone-grouped schematic (`schematicLayout.ts`), with every camera's
 * marker on top. `resolveEffectivePlacements` decides each camera's actual
 * position: its stored placement if the operator has dragged it somewhere,
 * the schematic default otherwise — computed identically whether or not a
 * real image is set, so a newly-added camera is never simply missing from
 * the map before someone gets around to placing it.
 */
export function FloorplanStage({
  cameras,
  placements,
  toneByCameraId,
  floorplanImage,
  editable,
  onPlace,
}: FloorplanStageProps) {
  const stageRef = useRef<HTMLDivElement>(null)
  const resolved = resolveEffectivePlacements(cameras, placements)
  const bands = floorplanImage ? [] : buildSchematicBands(cameras)

  function getStageRect(): StageRect | null {
    return stageRef.current?.getBoundingClientRect() ?? null
  }

  return (
    <div
      ref={stageRef}
      data-testid="floorplan-stage"
      className="relative aspect-video w-full overflow-hidden rounded-md border border-line bg-panel-2"
    >
      {floorplanImage ? (
        <img
          src={floorplanImage}
          alt=""
          aria-hidden="true"
          className="absolute inset-0 h-full w-full object-cover"
        />
      ) : (
        <div className="absolute inset-0 flex flex-col">
          {bands.map((band) => (
            <div
              key={band.key}
              className="flex-1 border-b border-line last:border-b-0"
              style={{ flexBasis: `${100 / bands.length}%` }}
            >
              <span className="eyebrow inline-block p-2">{band.label}</span>
            </div>
          ))}
        </div>
      )}

      {cameras.map((camera) => {
        const placement = resolved.get(camera.camera_id)
        if (!placement) return null
        const tone = toneByCameraId.get(camera.camera_id) ?? 'inert'
        return (
          <CameraMarker
            // Keyed on tone as well as id: a tone change remounts the
            // marker, which is what replays the dot's rise-in animation —
            // motion tied to the state transition itself, never to an
            // unrelated re-render (a drag on another marker, a data
            // refetch that changes nothing about this camera).
            key={`${camera.camera_id}:${tone}`}
            cameraId={camera.camera_id}
            zone={camera.zone ?? null}
            point={placement.point}
            tone={tone}
            isDefaultPosition={placement.isDefault}
            editable={editable}
            getStageRect={getStageRect}
            onDragTo={(point) => onPlace(camera.camera_id, point)}
            onNudge={(direction: NudgeDirection, step) =>
              onPlace(camera.camera_id, nudgePoint(placement.point, direction, step))
            }
          />
        )
      })}
    </div>
  )
}
