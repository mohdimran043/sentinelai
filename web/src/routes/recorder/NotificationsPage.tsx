import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { Reading } from '@/components/ui/Reading'
import { Notice } from '@/components/ui/Notice'
import { Pill } from '@/components/ui/Pill'
import { formatCount } from '@/lib/format'
import { useRecorderNotifications } from '@/recorder/queries'
import { joinNotificationRules, type JoinedNotificationRow } from '@/recorder/notificationJoin'
import type { RecorderNotificationsResponse } from '@/recorder/recorder.types'

export function NotificationsPage() {
  const notifications = useRecorderNotifications()
  const data: RecorderNotificationsResponse | undefined = notifications.data

  const rows: JoinedNotificationRow[] = data
    ? joinNotificationRules(data.event_types, data.rules, data.channels)
    : []
  const lockedCount = data ? data.locked.length : 0
  const configuredChannelCount = data
    ? data.channels.filter((channel) => channel.configured).length
    : 0
  const undeliverableLockedRows = rows.filter((row) => row.lockedButUndeliverable)

  return (
    <>
      <PageHeader
        eyebrow="operator visibility"
        title="Notifications"
        lede="The sixteen event types the recorder can raise, which channels each is routed to, and which of them an operator may never suppress. Nothing here is a statement about a person — every type describes a camera or its pipeline."
      />

      {notifications.isError ? (
        <Notice tone="breach">
          {notifications.error.message} None of the routing below can be trusted as current until
          the recorder answers again.
        </Notice>
      ) : null}

      {notifications.isPending ? <p className="muted">Loading notification routing…</p> : null}

      {data ? (
        <>
          <Notice tone="inert">{data.delivery_note}</Notice>

          <div className="mt-3.5 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Reading label="event types" value={formatCount(data.event_types.length)} />
            <Reading
              label="locked — never suppressible"
              value={formatCount(lockedCount)}
              alarm={lockedCount > 0}
            />
            <Reading
              label="channels configured"
              value={`${formatCount(configuredChannelCount)} of ${formatCount(data.channels.length)}`}
            />
            <Reading
              label="delivery wired"
              value={data.delivery_wired ? 'Yes' : 'No'}
              alarm={!data.delivery_wired}
            />
          </div>

          <ChannelsPanel channels={data.channels} />

          <Notice tone="breach" className="mt-3.5">
            {data.note}
          </Notice>

          {undeliverableLockedRows.length > 0 ? (
            <Notice tone="breach" className="mt-3.5">
              {formatCount(undeliverableLockedRows.length)} locked type
              {undeliverableLockedRows.length === 1 ? ' has' : 's have'} no channel that can
              actually deliver it right now (
              {undeliverableLockedRows.map((row) => row.type).join(', ')}
              ). That directly contradicts the guarantee above — treat it as a fault, not a
              configuration choice.
            </Notice>
          ) : null}

          <EventTypesPanel rows={rows} />
        </>
      ) : null}
    </>
  )
}

function ChannelsPanel({ channels }: { channels: RecorderNotificationsResponse['channels'] }) {
  return (
    <Panel className="mt-3.5 p-0">
      <div className="table-scroll">
        <table className="data-table">
          <caption className="sr-only">Notification delivery channels</caption>
          <thead>
            <tr>
              <th scope="col">Channel</th>
              <th scope="col">Configured</th>
              <th scope="col">Detail</th>
            </tr>
          </thead>
          <tbody>
            {channels.map((channel) => (
              <tr key={channel.name}>
                <td className="readout">{channel.name}</td>
                <td>
                  {channel.configured ? (
                    <Pill tone="nominal">configured</Pill>
                  ) : (
                    <Pill tone="breach">not configured</Pill>
                  )}
                </td>
                <td className="muted">{channel.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

function EventTypesPanel({ rows }: { rows: JoinedNotificationRow[] }) {
  return (
    <Panel className="mt-3.5">
      <div className="eyebrow">event type descriptions</div>
      <p className="muted mt-2 mb-3">
        Written for operators — read the description, not just the type name. A locked row cannot
        be disabled, left without a channel, or routed only to a channel that cannot deliver; there
        is no control here that would let you try.
      </p>
      <ul className="m-0 flex list-none flex-col gap-2.5 p-0">
        {rows.map((row) => (
          <EventTypeRow key={row.type} row={row} />
        ))}
      </ul>
    </Panel>
  )
}

function EventTypeRow({ row }: { row: JoinedNotificationRow }) {
  return (
    <li
      className="rounded-sm border border-line bg-panel-2 p-3"
      aria-label={`${row.type}${row.locked ? ', locked, cannot be suppressed' : ''}`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="readout text-[13px] font-semibold text-fg">{row.type}</span>
        {row.locked ? (
          <Pill tone="breach">locked — cannot be suppressed</Pill>
        ) : (
          <Pill tone="inert">not locked</Pill>
        )}
        {row.enabled === undefined ? (
          <Pill tone="inert">no rule reported</Pill>
        ) : row.enabled ? (
          <Pill tone="nominal">enabled</Pill>
        ) : (
          <Pill tone="inert">disabled</Pill>
        )}
      </div>
      <p className="m-0 mt-1.5 text-[13px] text-fg">{row.description}</p>
      <p className="muted m-0 mt-1.5">
        {row.channels.length === 0
          ? 'Routed to no channel.'
          : `Routed to: ${row.channels.join(', ')}.`}
        {row.enabled !== undefined && row.enabled !== row.defaultEnabled
          ? ` Ships enabled by default: ${row.defaultEnabled ? 'yes' : 'no'}.`
          : ''}
      </p>
    </li>
  )
}
