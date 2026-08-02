import { Link } from 'react-router-dom'
import { useCameras, useEngineHealth } from '@/api/queries'
import { useEngineEventStream } from '@/api/eventStream'
import type { CameraStatus, HealthResponse, RecentEventEntry } from '@/api/engineClient'
import { latestEventByCamera, markerTone } from '@/sitemap/markers'
import { buildSchematicBands } from '@/sitemap/schematicLayout'
import { Tile } from '@/components/ui/Panel'
import { Pill } from '@/components/ui/Pill'
import { KvList, type KvRow } from '@/components/ui/Kv'
import { Absent } from '@/components/ui/Absent'
import { Notice } from '@/components/ui/Notice'
import { Reading } from '@/components/ui/Reading'
import { cameraLiveness, toneForLiveness, toneForModelState, worseTone } from '@/lib/severity'
import { formatAgo, formatCount, formatThreatScore, humanizeEnum } from '@/lib/format'

function CameraTile({
  camera,
  latestEvent,
  showZone,
}: {
  camera: CameraStatus
  latestEvent: RecentEventEntry | undefined
  /**
   * Whether this tile prints its own zone label. False when the grid is
   * already grouped into zone bands (`CameraGrid` below) — the band heading
   * says the zone once for the whole group, and repeating it on every tile
   * inside would be the same text twice for no benefit. True in the flat,
   * one-band case, where a tile's zone label is the only place that fact
   * appears at all.
   */
  showZone: boolean
}) {
  const liveness = cameraLiveness(camera.last_frame_at)
  const livenessTone = toneForLiveness(liveness)
  const livenessLabel =
    liveness === 'live' ? 'Live' : liveness === 'stale' ? 'Stale' : 'No data yet'
  const severityTone = markerTone(latestEvent)
  const tone = worseTone(livenessTone, severityTone)
  const zoneLabel = camera.zone ? humanizeEnum(camera.zone) : 'Ungrouped'

  const rows: KvRow[] = [
    {
      key: 'last-frame',
      label: 'Last frame',
      value: camera.last_frame_at !== null ? formatAgo(camera.last_frame_at) : '—',
    },
    { key: 'escalations', label: 'Escalations', value: formatCount(camera.escalations) },
  ]
  if (latestEvent) {
    rows.push(
      { key: 'threat', label: 'Latest threat', value: formatThreatScore(latestEvent.threat_score) },
      { key: 'latest-at', label: 'Latest event', value: formatAgo(latestEvent.occurred_at) },
    )
  }

  return (
    <Link to={`/cameras/${encodeURIComponent(camera.camera_id)}`} className="block">
      <Tile tone={tone} className="cursor-pointer transition-colors hover:bg-panel-2">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h3>{camera.label}</h3>
          <Pill tone={livenessTone}>{livenessLabel}</Pill>
        </div>
        <div className="mb-2.5 flex flex-wrap items-center gap-2">
          {showZone ? <span className="eyebrow">{zoneLabel}</span> : null}
          {latestEvent ? (
            <Pill tone={severityTone}>{latestEvent.severity}</Pill>
          ) : (
            <Pill tone="inert">no events yet</Pill>
          )}
        </div>
        <KvList rows={rows} />
      </Tile>
    </Link>
  )
}

/**
 * Zone-grouped when there is more than one band to show (reusing the site
 * map's own zone bands, `buildSchematicBands`, so the two screens agree on
 * what a zone is and cannot drift apart) — a flat grid otherwise, since a
 * single "Ungrouped" heading over every camera would read worse than none.
 */
function CameraGrid() {
  const query = useCameras()
  const { events } = useEngineEventStream()

  if (query.isPending) {
    return (
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {[0, 1, 2].map((key) => (
          <Tile key={key} tone="inert" className="animate-pulse motion-reduce:animate-none">
            <div className="h-4 w-24 rounded-sm bg-panel-2" />
          </Tile>
        ))}
      </div>
    )
  }

  if (query.isError) {
    return (
      <Absent title="Camera grid unavailable">
        {query.error.message} The engine at <code>/engine</code> could not be reached.
        This section will populate automatically once it answers again.
      </Absent>
    )
  }

  if (query.data.cameras.length === 0) {
    return (
      <Absent title="No cameras configured.">
        Add one to <code>ai-engine/cameras.json</code> and restart the engine — see{' '}
        <code>cameras.example.json</code> for the shape.
      </Absent>
    )
  }

  const cameras = query.data.cameras
  const byId = new Map(cameras.map((camera) => [camera.camera_id, camera]))
  const latestByCamera = latestEventByCamera(events)
  const bands = buildSchematicBands(cameras)

  const grid = (cameraIds: readonly string[], showZone: boolean) => (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {cameraIds.map((cameraId) => {
        const camera = byId.get(cameraId)
        if (!camera) return null
        return (
          <CameraTile
            key={cameraId}
            camera={camera}
            latestEvent={latestByCamera.get(cameraId)}
            showZone={showZone}
          />
        )
      })}
    </div>
  )

  if (bands.length <= 1) {
    return grid(
      cameras.map((camera) => camera.camera_id),
      true,
    )
  }

  return (
    <div className="flex flex-col gap-5">
      {bands.map((band) => (
        <div key={band.key}>
          <div className="eyebrow mb-2">{band.label}</div>
          {grid(band.cameraIds, false)}
        </div>
      ))}
    </div>
  )
}

function modelVramTotal(health: HealthResponse): number {
  return health.models.reduce((sum, model) => sum + model.vram_mib, 0)
}

function SystemHealth() {
  const query = useEngineHealth()

  if (query.isPending) {
    return (
      <Tile tone="inert">
        <div className="eyebrow">System health</div>
        <p className="muted mt-2">Contacting the engine…</p>
      </Tile>
    )
  }

  if (query.isError) {
    return (
      <Tile tone="breach">
        <div className="eyebrow">System health</div>
        <p className="err mt-2">{query.error.message}</p>
      </Tile>
    )
  }

  const data = query.data

  return (
    <>
      <Reading label="Model VRAM (reported, loaded models)" value={formatCount(modelVramTotal(data))} unit="MiB" />
      {data.models.map((model) => (
        <Tile key={model.key} tone={toneForModelState(model.state)}>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-mono text-[13px]">{model.key}</h3>
            <Pill tone={toneForModelState(model.state)}>{humanizeEnum(model.state)}</Pill>
          </div>
          <p className="muted m-0">{model.detail || 'No detail reported.'}</p>
          <p className="readout mt-2 text-[12.5px] text-dim">{formatCount(model.vram_mib)} MiB</p>
        </Tile>
      ))}
    </>
  )
}

export function DashboardPage() {
  const camerasQuery = useCameras()
  const healthQuery = useEngineHealth()
  const engineUnreachable = camerasQuery.isError && healthQuery.isError

  const totalEscalations =
    camerasQuery.data?.cameras.reduce((sum, camera) => sum + camera.escalations, 0) ?? null

  return (
    <div>
      <h1>Dashboard</h1>
      <p className="lede">
        Every camera the engine knows about, as a tile — click one (or tab to it and press
        Enter) for its live scene description, threat chart and notifications.
      </p>

      {engineUnreachable ? (
        <Notice tone="breach" className="mb-4">
          The AI engine is unreachable. Every tile below is showing its last-known
          state or an explicit gap — nothing here has been replaced with a zero.
        </Notice>
      ) : null}

      <section className="mb-7">
        <div className="eyebrow mb-2.5">Camera grid</div>
        <CameraGrid />
      </section>

      <section>
        <div className="eyebrow mb-2.5">System health</div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <SystemHealth />
          {totalEscalations !== null ? (
            <Reading
              label="Escalations (cumulative, since engine start)"
              value={formatCount(totalEscalations)}
            />
          ) : null}
        </div>
      </section>
    </div>
  )
}
