import { Link } from 'react-router-dom'
import { useAlerts, useCameras, useEngineHealth } from '@/api/queries'
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
import { AlertRail } from '@/alerts/AlertRail'
import { CAPABILITY_LABELS, unseenAlerts } from '@/lib/alerts'
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
  const liveness = cameraLiveness(camera.last_frame_epoch ?? null)
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
      // `last_frame_epoch`, never `last_frame_at`: the latter is the camera's own
      // source timeline (monotonic, or seconds into a replayed file), and rendering
      // it as an age printed "20712d ago" against a camera delivering normally.
      value: camera.last_frame_epoch != null ? formatAgo(camera.last_frame_epoch) : '—',
    },
    { key: 'escalations', label: 'Escalations', value: formatCount(camera.escalations) },
  ]
  if (camera.falls_suspected > 0) {
    rows.push({
      key: 'falls',
      label: 'Possible falls',
      value: formatCount(camera.falls_suspected),
    })
  }
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
        {/*
          What this camera is actually running. Abbreviated to initials with the
          full name on hover: at tile density the words would wrap and push the
          readouts below the fold, and the point of the strip is that an operator
          can see at a glance which tiles are watching for what.

          A camera running nothing says so in words rather than showing an empty
          row — an absent strip and a deliberately quiet camera must not look the
          same.
        */}
        <div className="mt-2.5 flex flex-wrap items-center gap-1">
          {camera.capabilities.length === 0 ? (
            <span className="font-mono text-[10.5px] text-dimmer">nothing running</span>
          ) : (
            camera.capabilities.map((capability) => (
              <abbr
                key={capability}
                title={CAPABILITY_LABELS[capability] ?? capability}
                className="rounded-sm border border-line-2 bg-panel-2 px-[5px] py-px font-mono text-[10px] uppercase tracking-[0.06em] text-dim no-underline"
              >
                {(CAPABILITY_LABELS[capability] ?? capability)
                  .split(' ')
                  .map((word: string) => word[0])
                  .join('')}
              </abbr>
            ))
          )}
        </div>
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

/**
 * The standing status line (spec §20).
 *
 * Four facts, in the order somebody walking up to the console needs them: is
 * anything waiting, are the cameras delivering, is the AI running, has anything
 * happened. Deliberately a single line of readouts rather than four cards — this
 * is instrumentation an operator glances at, and a card grid would give it the
 * weight of the page's main content when the camera wall below is that.
 */
function StatusLine() {
  const cameras = useCameras()
  const health = useEngineHealth()
  const alerts = useAlerts()

  const cameraList = cameras.data?.cameras ?? []
  const live = cameraList.filter((camera) => cameraLiveness(camera.last_frame_epoch ?? null) === 'live')
  const unseen = alerts.data ? unseenAlerts(alerts.data.alerts).length : 0
  const models = health.data?.models ?? []
  const unhealthy = models.filter((model) => toneForModelState(model.state) === 'breach')
  const escalations = cameraList.reduce((sum, camera) => sum + camera.escalations, 0)

  return (
    <div className="mb-6 grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line lg:grid-cols-4">
      {/*
        Every cell reads `—` until its data has actually arrived, and that is not
        pedantry about loading states. A `0` on a surveillance console is a claim —
        "nothing is waiting", "no cameras are up" — and showing one while the
        engine is unreachable or merely slow is the one kind of wrong this page
        must never be. `isError` alone is not enough: a pending query has no data
        either.
      */}
      <StatusCell
        label="Waiting for you"
        value={alerts.data ? String(unseen) : '—'}
        tone={unseen > 0 ? 'breach' : 'nominal'}
        detail={
          !alerts.data
            ? 'not known yet'
            : unseen === 0
              ? 'nothing unacknowledged'
              : 'unacknowledged alerts'
        }
      />
      <StatusCell
        label="Cameras delivering"
        value={cameras.data ? `${live.length}/${cameraList.length}` : '—'}
        tone={cameraList.length > 0 && live.length < cameraList.length ? 'caution' : 'nominal'}
        detail={
          !cameras.data
            ? 'not known yet'
            : cameraList.length === 0
              ? 'none configured'
              : 'frames in the last minute'
        }
      />
      <StatusCell
        label="Models"
        value={health.data ? String(models.length) : '—'}
        tone={unhealthy.length > 0 ? 'breach' : 'nominal'}
        detail={
          !health.data
            ? 'not known yet'
            : unhealthy.length > 0
              ? `${unhealthy.length} unhealthy`
              : models.length === 0
                ? 'none loaded'
                : 'all healthy'
        }
      />
      <StatusCell
        label="Escalations"
        value={cameras.data ? formatCount(escalations) : '—'}
        tone="inert"
        detail="since the engine started"
      />
    </div>
  )
}

function StatusCell({
  label,
  value,
  tone,
  detail,
}: {
  label: string
  value: string
  tone: 'nominal' | 'caution' | 'breach' | 'inert'
  detail: string
}) {
  const valueColour =
    tone === 'breach' ? 'text-breach' : tone === 'caution' ? 'text-caution' : 'text-fg'
  return (
    <div className="bg-panel p-[13px_15px]">
      <div className="eyebrow">{label}</div>
      <div className={`readout mt-1.5 text-[22px] leading-none ${valueColour}`}>{value}</div>
      <div className="muted mt-1.5 text-[11.5px]">{detail}</div>
    </div>
  )
}

export function DashboardPage() {
  const camerasQuery = useCameras()
  const healthQuery = useEngineHealth()
  const engineUnreachable = camerasQuery.isError && healthQuery.isError

  return (
    <div>
      <h1>Command centre</h1>
      <p className="lede">
        Every camera the engine is watching, and everything currently asking for
        your attention. Click a tile for that camera's live scene, or an alert for
        the situation behind it.
      </p>

      {engineUnreachable ? (
        <Notice tone="breach" className="mb-4">
          The AI engine is unreachable. Every reading below is showing its
          last-known state or an explicit gap — nothing here has been replaced with
          a zero.
        </Notice>
      ) : null}

      <StatusLine />

      {/*
        The wall and the rail sit side by side above 1280px, and the rail comes
        *first* in the source order so a screen reader and a phone both meet the
        alerts before the camera grid. On a narrow screen that is the right
        priority; on a wide one `lg:order-2` puts the rail back on the right where
        an operator's eye expects it.
      */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1fr_340px]">
        <section className="xl:order-2">
          <div className="mb-2.5 flex items-baseline justify-between gap-2">
            <span className="eyebrow">Needs attention</span>
            <Link
              to="/alerts"
              className="font-mono text-[11px] uppercase tracking-[0.07em] text-dim underline-offset-2 hover:text-fg hover:underline"
            >
              All alerts
            </Link>
          </div>
          <AlertRail />
        </section>

        <section className="xl:order-1">
          <div className="eyebrow mb-2.5">Camera wall</div>
          <CameraGrid />
        </section>
      </div>

      <section className="mt-7">
        <div className="eyebrow mb-2.5">System health</div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <SystemHealth />
        </div>
      </section>
    </div>
  )
}
