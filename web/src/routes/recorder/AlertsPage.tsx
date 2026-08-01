import { useMemo, useState } from 'react'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { Reading } from '@/components/ui/Reading'
import { Notice } from '@/components/ui/Notice'
import { Absent } from '@/components/ui/Absent'
import { Pill } from '@/components/ui/Pill'
import { Button } from '@/components/ui/Button'
import { Select } from '@/components/ui/Select'
import { KvList } from '@/components/ui/Kv'
import { formatCount, formatNanosDateTime, formatNanosPrecise } from '@/lib/format'
import { useRecorderAlerts } from '@/recorder/queries'
import { useRecorderAlertStream, type AlertStreamStatus } from '@/recorder/alertStream'
import { toneForDeliveryStatus, toneForRecorderEventType } from '@/recorder/tone'
import {
  ALL,
  DEFAULT_ALERT_QUERY,
  cameraIdsIn,
  deliveriesForAlert,
  eventTypesIn,
  paginate,
  selectAlerts,
  type AlertOrder,
  type AlertQuery,
} from '@/recorder/alertQuery'
import type { RecorderAlert, RecorderAlertsResponse } from '@/recorder/recorder.types'

/**
 * 500 alerts arrive in one response and the DOM does not need 500 rows to
 * answer "what happened, and when". Fifty is roughly two screens — enough to
 * scan, small enough that filtering feels instant on a low-power console.
 */
const PAGE_SIZE = 50

export function AlertsPage() {
  const alerts = useRecorderAlerts()
  const streamStatus = useRecorderAlertStream()
  const [query, setQuery] = useState<AlertQuery>(DEFAULT_ALERT_QUERY)
  const [pageIndex, setPageIndex] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  /** Any filter change invalidates both the page position and the open detail. */
  const updateQuery = (patch: Partial<AlertQuery>) => {
    setQuery((current) => ({ ...current, ...patch }))
    setPageIndex(0)
    setSelectedId(null)
  }

  const data: RecorderAlertsResponse | undefined = alerts.data
  const allAlerts = useMemo(() => data?.alerts ?? [], [data])
  const cameraOptions = useMemo(() => cameraIdsIn(allAlerts), [allAlerts])
  const eventTypeOptions = useMemo(() => eventTypesIn(allAlerts), [allAlerts])
  const matched = useMemo(() => selectAlerts(allAlerts, query), [allAlerts, query])
  const page = useMemo(() => paginate(matched, pageIndex, PAGE_SIZE), [matched, pageIndex])
  const selected = useMemo(
    () => allAlerts.find((alert) => alert.id === selectedId) ?? null,
    [allAlerts, selectedId],
  )

  return (
    <>
      <PageHeader
        eyebrow="delivered facts"
        title="Alerts"
        lede="Every alert the recorder has raised about a camera, and what happened when it tried to deliver it. None of these is a statement about a person — this tier detects none."
      />

      {alerts.isError ? (
        <Notice tone="breach">
          {alerts.error.message} Nothing below has been replaced with a zero; the feed is simply
          not being read right now.
        </Notice>
      ) : null}

      {alerts.isPending ? <p className="muted">Loading the alert window…</p> : null}

      {data ? (
        <>
          <Notice tone="inert">{data.note}</Notice>

          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Reading
              label="failed deliveries"
              value={formatCount(data.failed_deliveries)}
              alarm={data.failed_deliveries > 0}
            />
            <Reading
              label="routed nowhere"
              value={formatCount(data.unrouted_deliveries)}
              alarm={data.unrouted_deliveries > 0}
            />
            <Reading
              label="dropped, never attempted"
              value={formatCount(data.dropped.total)}
              alarm={data.dropped.total > 0}
            />
            <Reading
              label="of those, unsuppressible"
              value={formatCount(data.dropped.welfare_locked)}
              alarm={data.dropped.welfare_locked > 0}
            />
          </div>

          {data.dropped.welfare_locked > 0 ? (
            <Notice tone="breach" className="mt-3.5">
              {formatCount(data.dropped.welfare_locked)} alert
              {data.dropped.welfare_locked === 1 ? '' : 's'} reporting lost observation or lost
              privacy enforcement {data.dropped.welfare_locked === 1 ? 'was' : 'were'} discarded
              before delivery was even attempted. These are the alerts that may never be
              suppressed. Treat it as a fault in the system, not as a quiet night.
            </Notice>
          ) : null}

          <AlertFilters
            query={query}
            cameraOptions={cameraOptions}
            eventTypeOptions={eventTypeOptions}
            onChange={updateQuery}
          />

          {selected ? (
            <AlertDetail
              alert={selected}
              response={data}
              onClose={() => setSelectedId(null)}
            />
          ) : null}

          <div className="mt-3.5 flex flex-wrap items-baseline justify-between gap-3">
            <h2 className="flex items-center gap-2">
              Feed
              <AlertStreamIndicator status={streamStatus} />
            </h2>
            <p className="muted readout m-0" data-testid="alert-count">
              {page.total === 0
                ? `no alerts match · ${formatCount(allAlerts.length)} in the window`
                : `${formatCount(page.firstItemNumber)}–${formatCount(page.lastItemNumber)} of ${formatCount(page.total)} · ${formatCount(allAlerts.length)} in the window`}
            </p>
          </div>

          {page.total === 0 ? (
            <Absent
              title={
                <>
                  <Pill tone="inert">no match</Pill>
                  {allAlerts.length === 0 ? 'Nothing has been delivered' : 'No alert matches those filters'}
                </>
              }
            >
              {allAlerts.length === 0
                ? 'No camera has raised a fact since this record began. That means no integrity change and no loss of observation was detected — it is not a statement that anyone is well, and nothing here watches people.'
                : 'Widen the camera or event-type filter. The window itself is not empty.'}
            </Absent>
          ) : (
            <>
              <Panel className="mt-3.5 p-0">
                <div className="table-scroll">
                  <table className="data-table">
                    <caption className="sr-only">
                      Alerts, {query.order === 'newest' ? 'newest first' : 'oldest first'}
                    </caption>
                    <thead>
                      <tr>
                        <th scope="col" className="w-[180px]">
                          When
                        </th>
                        <th scope="col">Camera</th>
                        <th scope="col">Event</th>
                        <th scope="col">Detail</th>
                        <th scope="col" className="w-[90px]">
                          <span className="sr-only">Actions</span>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {page.items.map((alert) => (
                        <tr
                          key={alert.id}
                          className={alert.id === selectedId ? 'bg-panel-2' : undefined}
                        >
                          <td className="readout whitespace-nowrap">
                            {formatNanosDateTime(alert.at_ns)}
                          </td>
                          <td className="readout">{alert.camera_id}</td>
                          <td>
                            <Pill tone={toneForRecorderEventType(alert.event_type)}>
                              {alert.event_type}
                            </Pill>
                          </td>
                          <td className="muted">{alert.detail || '—'}</td>
                          <td>
                            <Button
                              variant="quiet"
                              size="small"
                              aria-pressed={alert.id === selectedId}
                              aria-label={`Details for ${alert.event_type} on ${alert.camera_id} at ${formatNanosDateTime(alert.at_ns)}`}
                              onClick={() =>
                                setSelectedId((current) =>
                                  current === alert.id ? null : alert.id,
                                )
                              }
                            >
                              Details
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>

              <Pager page={page} onChange={setPageIndex} />
            </>
          )}
        </>
      ) : null}
    </>
  )
}

/**
 * Reflects `GET /api/alerts/stream`'s own connection state, not this page's
 * REST poll — the two are independent. `reconnecting` is deliberately the
 * SAME tone as `connecting`: both mean "the feed is not live right now",
 * which is the one fact this pill exists to surface, and the table below it
 * still shows the last-known window either way, not a blank.
 */
function AlertStreamIndicator({ status }: { status: AlertStreamStatus }) {
  if (status === 'live') {
    return <Pill tone="nominal">live</Pill>
  }
  return <Pill tone="caution">{status === 'connecting' ? 'connecting…' : 'reconnecting…'}</Pill>
}

function AlertFilters({
  query,
  cameraOptions,
  eventTypeOptions,
  onChange,
}: {
  query: AlertQuery
  cameraOptions: string[]
  eventTypeOptions: string[]
  onChange: (patch: Partial<AlertQuery>) => void
}) {
  return (
    <Panel className="mt-3.5">
      <div className="grid gap-3 md:grid-cols-3">
        <div>
          <label htmlFor="alert-camera" className="eyebrow mb-1.5 block">
            Camera
          </label>
          <Select
            id="alert-camera"
            value={query.cameraId}
            onChange={(event) => onChange({ cameraId: event.target.value })}
          >
            <option value={ALL}>All cameras</option>
            {cameraOptions.map((cameraId) => (
              <option key={cameraId} value={cameraId}>
                {cameraId}
              </option>
            ))}
          </Select>
        </div>
        <div>
          <label htmlFor="alert-event-type" className="eyebrow mb-1.5 block">
            Event type
          </label>
          <Select
            id="alert-event-type"
            value={query.eventType}
            onChange={(event) => onChange({ eventType: event.target.value })}
          >
            <option value={ALL}>All event types</option>
            {eventTypeOptions.map((eventType) => (
              <option key={eventType} value={eventType}>
                {eventType}
              </option>
            ))}
          </Select>
        </div>
        <div>
          <label htmlFor="alert-order" className="eyebrow mb-1.5 block">
            Order
          </label>
          <Select
            id="alert-order"
            value={query.order}
            onChange={(event) => onChange({ order: event.target.value as AlertOrder })}
          >
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
          </Select>
        </div>
      </div>
      <p className="muted mt-3 mb-0">
        The recorder serves the whole window in one response and ignores query parameters, so these
        narrow what you are shown — they do not ask it for less. Only cameras and event types that
        actually appear in the window are offered.
      </p>
    </Panel>
  )
}

function Pager({
  page,
  onChange,
}: {
  page: { pageIndex: number; pageCount: number }
  onChange: (pageIndex: number) => void
}) {
  if (page.pageCount <= 1) return null
  return (
    <nav aria-label="Alert pages" className="mt-3.5 flex items-center gap-3">
      <Button
        variant="quiet"
        size="small"
        disabled={page.pageIndex === 0}
        onClick={() => onChange(page.pageIndex - 1)}
      >
        Previous
      </Button>
      <span className="readout text-[12.5px] text-dim">
        Page {page.pageIndex + 1} of {page.pageCount}
      </span>
      <Button
        variant="quiet"
        size="small"
        disabled={page.pageIndex >= page.pageCount - 1}
        onClick={() => onChange(page.pageIndex + 1)}
      >
        Next
      </Button>
    </nav>
  )
}

function AlertDetail({
  alert,
  response,
  onClose,
}: {
  alert: RecorderAlert
  response: RecorderAlertsResponse
  onClose: () => void
}) {
  const deliveries = deliveriesForAlert(response.deliveries, alert.id)

  return (
    <Panel
      className="mt-3.5 border-t-2 border-t-signal"
      aria-labelledby="alert-detail-heading"
      role="region"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 id="alert-detail-heading" className="flex items-center gap-2.5">
          <Pill tone={toneForRecorderEventType(alert.event_type)}>{alert.event_type}</Pill>
          {alert.camera_id}
        </h2>
        <Button variant="quiet" size="small" onClick={onClose}>
          Close
        </Button>
      </div>

      <p className="lede mt-2 mb-3">
        {alert.detail || 'The recorder recorded no further detail for this alert.'}
      </p>

      <KvList
        rows={[
          { key: 'when', label: 'When', value: formatNanosPrecise(alert.at_ns) },
          { key: 'at_ns', label: 'Recorder timestamp (ns)', value: String(alert.at_ns) },
          { key: 'camera', label: 'Camera', value: alert.camera_id },
          { key: 'type', label: 'Event type', value: alert.event_type },
          { key: 'id', label: 'Alert id', value: alert.id },
        ]}
      />

      <h3 className="mt-4 mb-2 text-[13px]">Delivery attempts</h3>
      {deliveries.length === 0 ? (
        <p className="muted m-0">
          No delivery attempt is recorded against this alert. That is not the same as a successful
          delivery.
        </p>
      ) : (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">When</th>
                <th scope="col">Channel</th>
                <th scope="col">Outcome</th>
                <th scope="col">Tries</th>
                <th scope="col">Why</th>
              </tr>
            </thead>
            <tbody>
              {deliveries.map((delivery, index) => (
                <tr key={`${delivery.channel}-${index}`}>
                  <td className="readout whitespace-nowrap">
                    {formatNanosDateTime(delivery.at_ns)}
                  </td>
                  <td className="readout">{delivery.channel || '—'}</td>
                  <td>
                    <Pill tone={toneForDeliveryStatus(delivery.status)}>{delivery.status}</Pill>
                    {delivery.overridden ? (
                      <Pill tone="caution" className="ml-1.5">
                        config overridden
                      </Pill>
                    ) : null}
                  </td>
                  <td className="readout">{formatCount(delivery.attempts)}</td>
                  <td className="muted">{delivery.error ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="muted mt-3 mb-0">
        This is a fact about a camera, not about a person. Nothing in the recorder detects people.
      </p>
    </Panel>
  )
}
