import { Link, useParams } from 'react-router-dom'
import { useCameraTelemetry, useDescribeCameraNow } from '@/api/queries'
import { useEventFeed, useEventHistory } from '@/events/queries'
import { Panel } from '@/components/ui/Panel'
import { Reading } from '@/components/ui/Reading'
import { Pill } from '@/components/ui/Pill'
import { Notice } from '@/components/ui/Notice'
import { Absent } from '@/components/ui/Absent'
import { Button } from '@/components/ui/Button'
import { Ribbon } from '@/components/ui/Ribbon'
import { buildThreatRibbon } from '@/lib/ribbon'
import { cameraLiveness, toneForLiveness, toneForSeverity } from '@/lib/severity'
import { formatAgo, formatClockTime, formatCount, formatThreatScore, humanizeEnum } from '@/lib/format'
import type { SentinelAIAnomalyEvent } from '@/events/anomalyEvent.types'

function EventRow({ event }: { event: SentinelAIAnomalyEvent }) {
  const tone = toneForSeverity(event.severity)
  return (
    <li className="border-b border-line py-3 last:border-b-0">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Pill tone={tone}>{event.severity}</Pill>
          <span className="text-[13px] text-fg">{humanizeEnum(event.reason)}</span>
        </div>
        <span className="readout text-[12px] text-dim">{formatClockTime(event.occurred_at)}</span>
      </div>
      {event.description_unavailable ? (
        <p className="muted italic">
          Description unavailable — the vision model timed out or refused this scene.
        </p>
      ) : (
        <p className="m-0 text-[13px] text-fg">{event.description}</p>
      )}
      <p className="muted mt-1">
        Suggested: {event.suggested_action} · threat {formatThreatScore(event.threat_score)}
        {event.labels.length > 0 ? ` · ${event.labels.join(', ')}` : ''}
      </p>
    </li>
  )
}

export function CameraPage() {
  const { cameraId = '' } = useParams<{ cameraId: string }>()
  const telemetryQuery = useCameraTelemetry(cameraId)
  const feedQuery = useEventFeed(cameraId)
  const historyQuery = useEventHistory(cameraId)
  const describeMutation = useDescribeCameraNow(cameraId)

  const liveness =
    telemetryQuery.status === 'success' ? cameraLiveness(telemetryQuery.data.last_frame_at) : 'no-data'
  const tone = toneForLiveness(liveness)

  return (
    <div>
      <p className="mb-2">
        <Link to="/dashboard" className="muted hover:text-fg hover:underline">
          ← Dashboard
        </Link>
      </p>
      <div className="mb-1 flex flex-wrap items-center gap-2.5">
        <h1 className="mb-0">{cameraId}</h1>
        <Pill tone={tone}>
          {liveness === 'live' ? 'Live' : liveness === 'stale' ? 'Stale' : 'No data yet'}
        </Pill>
      </div>
      <p className="lede">
        Telemetry is live from the AI engine. The event feed and history below are
        mocked (MSW) — the real feed arrives via RabbitMQ → Go → Postgres in Phase 1C.
      </p>

      {telemetryQuery.isError ? (
        <Notice tone="breach" className="mb-4">
          Telemetry unreachable: {telemetryQuery.error.message}
        </Notice>
      ) : null}

      <Panel className="mb-4">
        <div className="mb-1 flex flex-wrap items-center justify-between gap-3">
          <h2>Threat over time</h2>
          <Button
            variant="act"
            size="small"
            onClick={() => describeMutation.mutate()}
            disabled={describeMutation.isPending}
          >
            {describeMutation.isPending ? 'Describing…' : 'Describe now'}
          </Button>
        </div>
        {describeMutation.isError ? (
          <p className="err mt-1" data-testid="describe-feedback" role="alert">
            {describeMutation.error.message}
          </p>
        ) : null}
        {describeMutation.isSuccess ? (
          <p className="ok-text mt-1" data-testid="describe-feedback">
            Requested — event <code>{describeMutation.data.event_id}</code> queued.
          </p>
        ) : null}

        {historyQuery.isPending ? (
          <p className="muted mt-3">Loading history…</p>
        ) : historyQuery.isError ? (
          <Absent title="History unavailable" className="mt-3">
            The mocked event service did not respond.
          </Absent>
        ) : (
          <div className="mt-3 overflow-x-auto">
            <Ribbon
              cells={buildThreatRibbon(historyQuery.data)}
              ariaLabel={`Threat level over the last 2 hours for ${cameraId}, 48 buckets of 2.5 minutes each`}
            />
          </div>
        )}
        <div className="mt-3 flex flex-wrap gap-4 text-[12px] text-dim">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-[11px] w-[11px] rounded-[1px] bg-signal" /> nominal
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-[11px] w-[11px] rounded-[1px] bg-caution" /> caution
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-[11px] w-[11px] rounded-[1px] bg-breach" /> breach
          </span>
        </div>
      </Panel>

      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {telemetryQuery.isSuccess ? (
          <>
            <Reading
              label="Last frame"
              value={
                telemetryQuery.data.last_frame_at !== null
                  ? formatAgo(telemetryQuery.data.last_frame_at)
                  : '—'
              }
            />
            <Reading label="Frames seen" value={formatCount(telemetryQuery.data.frames_seen)} />
            <Reading label="Frames dropped" value={formatCount(telemetryQuery.data.frames_dropped)} />
            <Reading label="Detections run" value={formatCount(telemetryQuery.data.detections_run)} />
            <Reading label="Escalations" value={formatCount(telemetryQuery.data.escalations)} />
            <Reading
              label="Escalations dropped"
              value={formatCount(telemetryQuery.data.escalations_dropped)}
              alarm={telemetryQuery.data.escalations_dropped > 0}
            />
            <Reading label="Discontinuities" value={formatCount(telemetryQuery.data.discontinuities)} />
            <Reading
              label="Last escalation"
              value={
                telemetryQuery.data.last_escalation_at !== null
                  ? formatAgo(telemetryQuery.data.last_escalation_at)
                  : 'none yet'
              }
            />
          </>
        ) : (
          <p className="muted col-span-full">
            {telemetryQuery.isPending ? 'Loading telemetry…' : 'Telemetry unavailable.'}
          </p>
        )}
      </div>

      <Panel>
        <h2>Event feed</h2>
        <p className="lede">Mocked — shaped by the anomaly event contract.</p>
        {feedQuery.isPending ? (
          <p className="muted">Loading…</p>
        ) : feedQuery.isError ? (
          <Absent title="Event feed unavailable">The mocked event service did not respond.</Absent>
        ) : feedQuery.data.length === 0 ? (
          <Absent title="No events recorded yet">Nothing has happened on this camera yet.</Absent>
        ) : (
          <ul className="max-h-[46vh] list-none overflow-y-auto p-0">
            {feedQuery.data.map((event) => (
              <EventRow key={event.event_id} event={event} />
            ))}
          </ul>
        )}
      </Panel>
    </div>
  )
}
