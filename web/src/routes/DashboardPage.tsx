import { Link } from 'react-router-dom'
import { useCameras, useEngineHealth } from '@/api/queries'
import type { CameraStatus, HealthResponse } from '@/api/engineClient'
import { Tile } from '@/components/ui/Panel'
import { Pill } from '@/components/ui/Pill'
import { KvList } from '@/components/ui/Kv'
import { Absent } from '@/components/ui/Absent'
import { Notice } from '@/components/ui/Notice'
import { Reading } from '@/components/ui/Reading'
import { cameraLiveness, toneForLiveness, toneForModelState } from '@/lib/severity'
import { formatAgo, formatCount, humanizeEnum } from '@/lib/format'

function CameraTile({ camera }: { camera: CameraStatus }) {
  const liveness = cameraLiveness(camera.last_frame_at)
  const tone = toneForLiveness(liveness)
  const livenessLabel =
    liveness === 'live' ? 'Live' : liveness === 'stale' ? 'Stale' : 'No data yet'

  return (
    <Link to={`/cameras/${encodeURIComponent(camera.camera_id)}`} className="block">
      <Tile tone={tone} className="cursor-pointer transition-colors hover:bg-panel-2">
        <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
          <h3>{camera.camera_id}</h3>
          <Pill tone={tone}>{livenessLabel}</Pill>
        </div>
        <KvList
          rows={[
            {
              key: 'last-frame',
              label: 'Last frame',
              value: camera.last_frame_at !== null ? formatAgo(camera.last_frame_at) : '—',
            },
            { key: 'frames-seen', label: 'Frames seen', value: formatCount(camera.frames_seen) },
            {
              key: 'frames-dropped',
              label: 'Frames dropped',
              value: formatCount(camera.frames_dropped),
            },
            { key: 'escalations', label: 'Escalations', value: formatCount(camera.escalations) },
            {
              key: 'discontinuities',
              label: 'Discontinuities',
              value: formatCount(camera.discontinuities),
            },
          ]}
        />
      </Tile>
    </Link>
  )
}

function CameraGrid() {
  const query = useCameras()

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

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {query.data.cameras.map((camera) => (
        <CameraTile key={camera.camera_id} camera={camera} />
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
        Live from the AI engine (health, cameras, telemetry). The event feed and daily
        counters below are mocked — see the note under &quot;Awaiting Phase 1C&quot;.
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

      <section className="mb-7">
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

      <section>
        <div className="eyebrow mb-2.5">Awaiting Phase 1C</div>
        <p className="muted mb-2.5 max-w-[68ch]">
          These arrive from Go + Postgres (spec §7-8), which is not built yet. Showing
          a number here would be inventing one — the tiles below say so instead.
        </p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Absent title="Storage usage">No endpoint yet. Arrives with MinIO reporting in Phase 1C.</Absent>
          <Absent title="Today&apos;s visitors">Requires the Go event store and daily aggregation.</Absent>
          <Absent title="Today&apos;s AI summary">Requires the daily journal (spec §26, Phase 7).</Absent>
          <Absent title="Live notifications">Beyond in-page toasts, requires the WebSocket hub (Phase 1C).</Absent>
        </div>
      </section>
    </div>
  )
}
