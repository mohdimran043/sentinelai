import { describe, expect, it } from 'vitest'
import { screen, within } from '@testing-library/react'
import { renderWithProviders } from '@/test/renderWithProviders'
import { server } from '@/test/mswServer'
import { NotificationsPage } from '@/routes/recorder/NotificationsPage'
import {
  recorderErrorHandler,
  recorderNotificationsHandler,
  recorderUnreachableHandler,
} from '@/recorder/mocks/handlers'
import { REAL_NOTIFICATIONS } from '@/recorder/mocks/fixtures'
import type { RecorderNotificationsResponse } from '@/recorder/recorder.types'

function withNotifications(patch: Partial<RecorderNotificationsResponse>) {
  server.use(recorderNotificationsHandler({ ...REAL_NOTIFICATIONS, ...patch }))
}

async function eventTypeRow(type: string) {
  const label = await screen.findByText(type)
  // The row is the nearest <li> ancestor.
  return label.closest('li') as HTMLElement
}

describe('NotificationsPage — real payload', () => {
  it('renders all 16 event types with their real descriptions', async () => {
    renderWithProviders(<NotificationsPage />)

    expect(await screen.findByText('STREAM_LOST')).toBeInTheDocument()
    expect(screen.getAllByText(/^[A-Z_]+$/).length).toBeGreaterThanOrEqual(16)
    expect(
      screen.getByText('The GPU pipeline failed. Decode, masking and encode all stop.'),
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        'No frames arrived for long enough to count as an outage. Observation of this camera has stopped.',
      ),
    ).toBeInTheDocument()
  })

  it('marks exactly the eight locked types as locked, and the rest as not locked', async () => {
    renderWithProviders(<NotificationsPage />)

    await screen.findByText('STREAM_LOST')
    expect(screen.getAllByText('locked — cannot be suppressed')).toHaveLength(8)
    expect(screen.getAllByText('not locked')).toHaveLength(8)

    const lockedRow = await eventTypeRow('CUDA_ERROR')
    expect(within(lockedRow).getByText('locked — cannot be suppressed')).toBeInTheDocument()

    const unlockedRow = await eventTypeRow('OBSTRUCTED')
    expect(within(unlockedRow).getByText('not locked')).toBeInTheDocument()
  })

  it('reads the summary tiles off the real payload', async () => {
    renderWithProviders(<NotificationsPage />)

    expect(await screen.findByText('event types')).toBeInTheDocument()
    const eventTypesTile = screen.getByText('event types').parentElement!
    expect(within(eventTypesTile).getByText('16')).toBeInTheDocument()

    const lockedTile = screen.getByText('locked — never suppressible').parentElement!
    expect(within(lockedTile).getByText('8')).toBeInTheDocument()

    const channelsTile = screen.getByText('channels configured').parentElement!
    expect(within(channelsTile).getByText('1 of 2')).toBeInTheDocument()

    const wiredTile = screen.getByText('delivery wired').parentElement!
    expect(within(wiredTile).getByText('Yes')).toBeInTheDocument()
  })

  it('renders the recorder’s own note about what a locked type means, verbatim', async () => {
    renderWithProviders(<NotificationsPage />)
    expect(
      await screen.findByText(
        'Locked event types report loss of observation or loss of privacy enforcement and cannot be disabled, left without channels, or routed only to a channel that cannot deliver. No configuration may suppress them.',
      ),
    ).toBeInTheDocument()
  })

  it('lists both channels with their real configured state and detail', async () => {
    renderWithProviders(<NotificationsPage />)

    const table = await screen.findByRole('table', { name: /channels/i })
    const dashboardRow = within(table)
      .getAllByRole('row')
      .find((row) => row.textContent?.includes('dashboard'))!
    expect(within(dashboardRow).getByText('configured')).toBeInTheDocument()

    const webhookRow = within(table)
      .getAllByRole('row')
      .find((row) => row.textContent?.includes('webhook'))!
    expect(within(webhookRow).getByText('not configured')).toBeInTheDocument()
    expect(within(webhookRow).getByText(/reaches nobody/i)).toBeInTheDocument()
  })

  it('shows each row’s current enabled state, honouring a default-off type', async () => {
    renderWithProviders(<NotificationsPage />)

    const enabledRow = await eventTypeRow('OBSTRUCTED')
    expect(within(enabledRow).getByText('enabled')).toBeInTheDocument()

    const disabledRow = await eventTypeRow('DEGRADED_QUALITY')
    expect(within(disabledRow).getByText('disabled')).toBeInTheDocument()
  })

  it('does not raise the undeliverable-lock fault when every locked type routes to the configured channel', async () => {
    renderWithProviders(<NotificationsPage />)
    await screen.findByText('STREAM_LOST')
    expect(screen.queryByText(/contradicts the guarantee above/i)).not.toBeInTheDocument()
  })
})

describe('NotificationsPage — locked types cannot be toggled', () => {
  it('renders no interactive control anywhere that would let an operator flip a rule', async () => {
    renderWithProviders(<NotificationsPage />)
    await screen.findByText('STREAM_LOST')

    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument()
    expect(screen.queryByRole('switch')).not.toBeInTheDocument()
    expect(document.querySelector('input')).not.toBeInTheDocument()
    // This page has no button that could suppress or edit a rule.
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })
})

describe('NotificationsPage — a locked type that cannot actually deliver', () => {
  it('flags it as a fault, not a quiet configuration choice', async () => {
    withNotifications({
      rules: REAL_NOTIFICATIONS.rules.map((rule) =>
        rule.event_type === 'CUDA_ERROR' ? { ...rule, channels: ['webhook'] } : rule,
      ),
    })
    renderWithProviders(<NotificationsPage />)

    const fault = await screen.findByText(/contradicts the guarantee above/i)
    expect(fault).toBeInTheDocument()
    expect(fault.textContent).toContain('CUDA_ERROR')
  })

  it('still shows the type as locked even though it cannot deliver — never silently unlocking it', async () => {
    withNotifications({
      rules: REAL_NOTIFICATIONS.rules.map((rule) =>
        rule.event_type === 'CUDA_ERROR' ? { ...rule, channels: [] } : rule,
      ),
    })
    renderWithProviders(<NotificationsPage />)

    const row = await eventTypeRow('CUDA_ERROR')
    expect(within(row).getByText('locked — cannot be suppressed')).toBeInTheDocument()
    expect(within(row).getByText('Routed to no channel.')).toBeInTheDocument()
  })
})

describe('NotificationsPage — a declared type with no matching rule', () => {
  it('says no rule was reported rather than guessing enabled or disabled', async () => {
    withNotifications({
      rules: REAL_NOTIFICATIONS.rules.filter((rule) => rule.event_type !== 'CLOCK_STEP_DETECTED'),
    })
    renderWithProviders(<NotificationsPage />)

    const row = await eventTypeRow('CLOCK_STEP_DETECTED')
    expect(within(row).getByText('no rule reported')).toBeInTheDocument()
  })
})

describe('NotificationsPage — recorder unavailable', () => {
  it('says the recorder is unreachable rather than rendering an empty page', async () => {
    server.use(recorderUnreachableHandler('/notifications'))
    renderWithProviders(<NotificationsPage />)

    expect(
      await screen.findByText(/The recorder is unreachable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
    expect(screen.queryByText('STREAM_LOST')).not.toBeInTheDocument()
  })

  it('reports a recorder error response in its own words', async () => {
    server.use(recorderErrorHandler('/notifications', 500, 'notification store unavailable'))
    renderWithProviders(<NotificationsPage />)

    expect(
      await screen.findByText(/notification store unavailable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
  })
})
