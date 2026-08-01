import { useRef, useState, type ChangeEvent } from 'react'
import { useCameras } from '@/api/queries'
import { useEngineEventStream } from '@/api/eventStream'
import { useSiteMapStore } from '@/sitemap/siteMapStore'
import { latestEventByCamera, markerTone } from '@/sitemap/markers'
import { FloorplanStage } from '@/sitemap/FloorplanStage'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { Button } from '@/components/ui/Button'
import { Notice } from '@/components/ui/Notice'
import { Absent } from '@/components/ui/Absent'
import { Pill } from '@/components/ui/Pill'
import type { Tone } from '@/lib/severity'
import { toneForSeverity } from '@/lib/severity'
import { formatAgo, formatClockTime, humanizeEnum } from '@/lib/format'

/** How many rows the live activity feed keeps on screen — the underlying merged list (`useEngineEventStream`) holds far more, this is a reading, not the whole record. */
const FEED_VISIBLE_ROWS = 20

const STATUS_LABEL: Record<'connecting' | 'live' | 'reconnecting', string> = {
  connecting: 'Connecting…',
  live: 'Live',
  reconnecting: 'Reconnecting…',
}

export function SiteMapPage() {
  const camerasQuery = useCameras()
  const { events, status } = useEngineEventStream()
  const placements = useSiteMapStore((state) => state.placements)
  const floorplanImage = useSiteMapStore((state) => state.floorplanImage)
  const setPlacement = useSiteMapStore((state) => state.setPlacement)
  const setFloorplanImage = useSiteMapStore((state) => state.setFloorplanImage)
  const [editing, setEditing] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const cameras = camerasQuery.data?.cameras ?? []
  const latestByCamera = latestEventByCamera(events)
  const toneByCameraId = new Map<string, Tone>(
    cameras.map((camera) => [camera.camera_id, markerTone(latestByCamera.get(camera.camera_id))]),
  )

  function handleFileChange(event: ChangeEvent<HTMLInputElement>): void {
    const file = event.target.files?.[0]
    event.target.value = '' // lets the operator pick the same file again later
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') setFloorplanImage(reader.result)
    }
    reader.readAsDataURL(file)
  }

  return (
    <div>
      <PageHeader
        eyebrow="AI engine"
        title="Site map"
        lede="Cameras placed on a floorplan, lighting up by severity as events arrive from the live stream. Click a marker for that camera's own page."
      />

      {camerasQuery.isError ? (
        <Notice tone="breach" className="mb-4">
          Cameras unreachable: {camerasQuery.error.message}
        </Notice>
      ) : null}

      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2.5">
          <Button
            variant={editing ? 'act' : 'quiet'}
            size="small"
            onClick={() => setEditing((was) => !was)}
          >
            {editing ? 'Done arranging' : 'Arrange cameras'}
          </Button>
          {editing ? (
            <span className="muted">
              Drag a marker, or focus it and use the arrow keys (hold Shift to move further).
            </span>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-2.5">
          <Button variant="quiet" size="small" onClick={() => fileInputRef.current?.click()}>
            {floorplanImage ? 'Replace floorplan image' : 'Upload floorplan image'}
          </Button>
          {floorplanImage ? (
            <Button variant="quiet" size="small" onClick={() => setFloorplanImage(null)}>
              Remove image
            </Button>
          ) : null}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            aria-label="Upload floorplan image"
            onChange={handleFileChange}
          />
        </div>
      </div>

      <Notice tone="inert" className="mb-4">
        Camera placements and the floorplan image are saved in <strong>this browser only</strong> —
        there is no site-map model on the engine yet, so nothing here is shared with another
        operator or another machine. A new browser starts from the zone-grouped default below.
      </Notice>

      {camerasQuery.isPending ? (
        <Panel>
          <p className="muted">Loading cameras…</p>
        </Panel>
      ) : cameras.length === 0 ? (
        <Absent title="No cameras configured.">
          Add one to <code>ai-engine/cameras.json</code> and restart the engine — the map
          populates automatically once <code>GET /cameras</code> reports one.
        </Absent>
      ) : (
        <FloorplanStage
          cameras={cameras}
          placements={placements}
          toneByCameraId={toneByCameraId}
          floorplanImage={floorplanImage}
          editable={editing}
          onPlace={setPlacement}
        />
      )}

      <div className="mt-3 flex flex-wrap gap-4 text-[12px] text-dim">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-full border-2 border-signal-d bg-signal" />
          nominal
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-full border-2 border-[#574000] bg-caution" />
          caution
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-full border-2 border-[#5c231b] bg-breach" />
          breach
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-full border-2 border-line-2 bg-panel-2" />
          no events yet
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-full border-2 border-dashed border-line-2 bg-panel-2" />
          not yet placed
        </span>
      </div>

      <Panel className="mt-4" data-testid="live-activity-panel">
        <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
          <h2>Live activity</h2>
          <span className="muted">{STATUS_LABEL[status]}</span>
        </div>
        <p className="lede">
          Every camera's escalations as they arrive on <code>/events/stream</code>, newest first.
          An event that gains a clip after the fact updates this same row — it never appears
          twice.
        </p>
        {events.length === 0 ? (
          <Absent title="No events yet">
            Nothing has escalated on any camera in this process yet — this list fills in the
            moment the first one does.
          </Absent>
        ) : (
          <ul className="max-h-[40vh] list-none overflow-y-auto p-0">
            {events.slice(0, FEED_VISIBLE_ROWS).map((event) => (
              <li key={event.event_id} className="border-b border-line py-2.5 last:border-b-0">
                <div className="flex flex-wrap items-center gap-2">
                  <Pill tone={toneForSeverity(event.severity)}>{event.severity}</Pill>
                  <span className="readout text-[12.5px] text-fg">{event.camera_id}</span>
                  <span className="text-[12.5px] text-fg">{humanizeEnum(event.reason)}</span>
                  {event.clip_uri ? (
                    <span className="rounded-sm border border-line-2 px-1.5 py-px font-mono text-[10px] uppercase tracking-[0.08em] text-dim">
                      Clip attached
                    </span>
                  ) : null}
                  <span className="readout ml-auto text-[12px] text-dim">
                    {formatClockTime(event.occurred_at)} · {formatAgo(event.occurred_at)}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  )
}
