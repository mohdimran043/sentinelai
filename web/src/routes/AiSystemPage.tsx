import { useCameras, useEngineHealth } from '@/api/queries'
import type { CameraStatus, HealthResponse } from '@/api/engineClient'
import { Absent } from '@/components/ui/Absent'
import { Notice } from '@/components/ui/Notice'
import { Pill } from '@/components/ui/Pill'
import { Reading } from '@/components/ui/Reading'
import { CAPABILITY_LABELS } from '@/lib/alerts'
import { formatAgo, formatCount, humanizeEnum } from '@/lib/format'
import { cameraLiveness, toneForLiveness, toneForModelState } from '@/lib/severity'

function totalVram(health: HealthResponse): number {
  return health.models.reduce((sum, model) => sum + model.vram_mib, 0)
}

/**
 * AI system (spec §26).
 *
 * What this page deliberately does **not** do is invent numbers. The engine
 * reports which models are resident, what each measured, and per-camera frame
 * counters — so that is what is shown. There is no inference-FPS gauge and no
 * average-latency figure, because the engine does not measure them in production:
 * per-stage latency comes from the benchmark harness, which is a separate tool run
 * deliberately, and a dial here filled with a plausible number would be the exact
 * thing §33 forbids.
 *
 * The per-model VRAM figures are **marginal** costs — what each model adds to a
 * process that already has the others. That is the question residency planning
 * actually asks, and the page says so rather than letting a reader add them up and
 * expect `nvidia-smi` to agree.
 */
export function AiSystemPage() {
  const health = useEngineHealth()
  const cameras = useCameras()

  const cameraList: CameraStatus[] = cameras.data?.cameras ?? []
  const live = cameraList.filter((camera) => cameraLiveness(camera.last_frame_epoch ?? null) === 'live')
  // Only cameras that actually run detection. A camera with no detector drops
  // nothing by construction, so including it dilutes the figure toward zero — the
  // headline read 0.4% while the one camera doing the work was at 1%, which is the
  // wrong way round for a number an operator uses to judge whether a site is
  // keeping up.
  const detecting = cameraList.filter((camera) => camera.detections_run > 0)
  const dropped = detecting.reduce((sum, camera) => sum + camera.frames_dropped, 0)
  const seen = detecting.reduce((sum, camera) => sum + camera.frames_seen, 0)
  const dropRate = seen > 0 ? dropped / seen : 0

  return (
    <div>
      <h1>AI system</h1>
      <p className="lede">
        Which models are loaded, what they cost, and how much of the incoming video
        each camera is actually keeping up with.
      </p>

      {health.isError && cameras.isError ? (
        <Notice tone="breach" className="mb-4">
          The AI engine is unreachable. Nothing below has been replaced with a zero.
        </Notice>
      ) : null}

      <section className="mb-7">
        <div className="eyebrow mb-2.5">Right now</div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Reading
            label="Cameras delivering frames"
            value={`${live.length} / ${cameraList.length}`}
          />
          <Reading label="Models resident" value={formatCount(health.data?.models.length ?? 0)} />
          <Reading
            label="VRAM attributed to models"
            value={formatCount(health.data ? totalVram(health.data) : 0)}
            unit="MiB"
          />
          <Reading
            label="Frames dropped before detection"
            value={detecting.length === 0 ? '—' : `${(dropRate * 100).toFixed(1)}%`}
            // Above a fifth, a camera is no longer being watched the way an
            // operator would assume, and that is worth flagging in colour.
            alarm={dropRate > 0.2}
          />
        </div>
        <p className="muted mt-2 max-w-[80ch]">
          Across the {detecting.length} camera{detecting.length === 1 ? '' : 's'} actually
          running detection. Dropping frames is by design, not a fault: the pipeline
          keeps the newest frame and discards the stale one so it reports current
          reality rather than a delayed complete record. A high rate still means
          detection is running on a fraction of the video, which changes what the
          detectors can see.
        </p>
      </section>

      <section className="mb-7">
        <div className="eyebrow mb-2.5">Models</div>
        {health.isPending ? (
          <div className="h-[90px] animate-pulse rounded-md border border-line bg-panel motion-reduce:animate-none" />
        ) : health.isError ? (
          <Absent title="Model health unavailable">{health.error.message}</Absent>
        ) : (
          <>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {health.data.models.map((model) => (
                <div
                  key={model.key}
                  className="rounded-md border border-line bg-panel p-[15px]"
                >
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <h3 className="font-mono text-[12.5px] break-all">{model.key}</h3>
                    <Pill tone={toneForModelState(model.state)}>
                      {humanizeEnum(model.state)}
                    </Pill>
                  </div>
                  <p className="readout m-0 text-[13px] text-dim">
                    {formatCount(model.vram_mib)} MiB
                  </p>
                  {model.detail ? <p className="muted m-0 mt-1.5">{model.detail}</p> : null}
                </div>
              ))}
            </div>
            <p className="muted mt-2 max-w-[80ch]">
              These are <strong className="font-semibold text-fg">marginal</strong> figures —
              what each model adds to a process that already holds the others. They do
              not sum to what the driver reports, because the CUDA context and
              allocator workspace belong to no single model.
            </p>
          </>
        )}
      </section>

      <section>
        <div className="eyebrow mb-2.5">Per camera</div>
        {cameras.isPending ? (
          <div className="h-[120px] animate-pulse rounded-md border border-line bg-panel motion-reduce:animate-none" />
        ) : cameras.isError ? (
          <Absent title="Camera telemetry unavailable">{cameras.error.message}</Absent>
        ) : cameraList.length === 0 ? (
          <Absent title="No cameras configured." />
        ) : (
          <div className="table-scroll rounded-md border border-line bg-panel">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Camera</th>
                  <th scope="col">Running</th>
                  <th scope="col">Frames</th>
                  <th scope="col">Detected</th>
                  <th scope="col">Dropped</th>
                  <th scope="col">Escalations</th>
                  <th scope="col">Last frame</th>
                </tr>
              </thead>
              <tbody>
                {cameraList.map((camera) => {
                  const liveness = cameraLiveness(camera.last_frame_epoch ?? null)
                  const rate =
                    camera.frames_seen > 0 ? camera.frames_dropped / camera.frames_seen : 0
                  return (
                    <tr key={camera.camera_id}>
                      <td>
                        <span className="text-fg">{camera.label}</span>
                        <Pill tone={toneForLiveness(liveness)} className="ml-2">
                          {liveness === 'live' ? 'live' : liveness === 'stale' ? 'stale' : 'no data'}
                        </Pill>
                      </td>
                      <td className="text-dim">
                        {camera.capabilities.length === 0
                          ? 'nothing'
                          : camera.capabilities
                              .map((capability) => CAPABILITY_LABELS[capability] ?? capability)
                              .join(', ')}
                      </td>
                      <td className="readout">{formatCount(camera.frames_seen)}</td>
                      <td className="readout">{formatCount(camera.detections_run)}</td>
                      <td className={rate > 0.2 ? 'readout text-caution' : 'readout'}>
                        {(rate * 100).toFixed(0)}%
                      </td>
                      <td className="readout">{formatCount(camera.escalations)}</td>
                      <td className="readout text-dim">
                        {camera.last_frame_epoch != null ? formatAgo(camera.last_frame_epoch) : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
        <p className="muted mt-2 max-w-[80ch]">
          Per-stage latency is not on this page because the engine does not measure
          it while serving. Run{' '}
          <code className="font-mono text-[11.5px] text-dim">
            python -m sentinel_ai.benchmark
          </code>{' '}
          for that — a number here that nothing measured would be worse than its
          absence.
        </p>
      </section>
    </div>
  )
}
