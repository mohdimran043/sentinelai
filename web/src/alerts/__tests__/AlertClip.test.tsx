import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { AlertClip } from '@/alerts/AlertClip'

const ALERT_ID = '77917175-a7cc-469d-992f-1ae10edeb6e7'
const FULL = 's3://sentinel-clips/linkou/abc.mp4'
const SHORT = 's3://sentinel-clips/linkou/abc-notify.mp4'

function video(): HTMLVideoElement {
  return screen.getByTestId('alert-clip-video') as HTMLVideoElement
}

describe('AlertClip, what plays by default', () => {
  it('plays the three-second copy, not the full recording', () => {
    // The whole point of the short clip: on a page of rows, three seconds gets
    // watched and fourteen gets scrolled past.
    renderWithProviders(<AlertClip alertId={ALERT_ID} clipUri={FULL} notifyClipUri={SHORT} />)

    expect(video()).toHaveAttribute('data-variant', 'short')
    expect(video().getAttribute('src')).toContain('short=true')
    expect(screen.getByText('first 3 seconds')).toBeInTheDocument()
  })

  it('asks the engine for the clip by alert id and never by storage uri', () => {
    // The alert id is the whole of the authorisation. A player that named the object
    // would make the console the thing deciding what the engine may serve.
    renderWithProviders(<AlertClip alertId={ALERT_ID} clipUri={FULL} notifyClipUri={SHORT} />)

    const src = video().getAttribute('src') ?? ''
    expect(src).toContain(`/alerts/${ALERT_ID}/clip`)
    expect(src).not.toContain('s3://')
    expect(src).not.toContain('sentinel-clips')
  })

  it('never autoplays, never loops, and fetches nothing until asked', () => {
    // These are recordings of real people in distress. A wall of them playing on loop
    // is worse triage and worse for the people in it — and `preload="none"` means
    // opening the page does not pull a video per row off the engine either.
    renderWithProviders(<AlertClip alertId={ALERT_ID} clipUri={FULL} notifyClipUri={SHORT} />)

    expect(video()).not.toHaveAttribute('autoplay')
    expect(video()).not.toHaveAttribute('loop')
    expect(video()).toHaveAttribute('preload', 'none')
    expect(video()).toHaveAttribute('controls')
  })
})

describe('AlertClip, reaching the full recording', () => {
  it('switches to the full clip and back', async () => {
    const user = userEvent.setup()
    renderWithProviders(<AlertClip alertId={ALERT_ID} clipUri={FULL} notifyClipUri={SHORT} />)

    await user.click(screen.getByRole('button', { name: /play full clip/i }))

    expect(video()).toHaveAttribute('data-variant', 'full')
    expect(video().getAttribute('src')).toContain('short=false')
    expect(screen.getByText('full recording')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /back to 3s/i }))

    expect(video()).toHaveAttribute('data-variant', 'short')
  })

  it('offers no switch when the engine has only one length to give', () => {
    // No short copy was trimmed, so `short=true` already serves the full recording.
    // A button labelled "play full clip" that changes nothing is a control that lies.
    renderWithProviders(<AlertClip alertId={ALERT_ID} clipUri={FULL} notifyClipUri={null} />)

    expect(screen.queryByRole('button', { name: /play full clip/i })).not.toBeInTheDocument()
    expect(screen.getByText('full recording')).toBeInTheDocument()
  })
})

describe('AlertClip, when there is nothing to play', () => {
  it('says no clip has landed rather than rendering an empty player', () => {
    // `clip_uri: null` is normal early in an episode — clips finish after their event
    // is assembled — so this has to read as "not yet", not as a failure.
    renderWithProviders(<AlertClip alertId={ALERT_ID} clipUri={null} notifyClipUri={null} />)

    expect(screen.queryByTestId('alert-clip-video')).not.toBeInTheDocument()
    expect(screen.getByText(/no clip yet/i)).toBeInTheDocument()
  })
})
