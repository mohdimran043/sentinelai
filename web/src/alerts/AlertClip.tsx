import { useState } from 'react'
import { alertClipUrl } from '@/api/engineClient'
import { cn } from '@/lib/cn'

export interface AlertClipProps {
  alertId: string
  /** `AlertEntry.clip_uri`. Null means no recording landed — see below. */
  clipUri: string | null
  /** `AlertEntry.notify_clip_uri`. Null means no short copy was trimmed. */
  notifyClipUri: string | null
}

/**
 * The recording behind one alert, playable in the row.
 *
 * **Why the short clip is the default.** The engine already writes two copies of the
 * same moment: the evidence clip, with its pre-roll and post-roll, and a three-second
 * trim made for notifications. On this page an operator is scanning a list deciding
 * where to go, and three seconds is a length that gets watched — a fourteen-second clip
 * on every row is a page nobody plays. The full recording is one click away for the row
 * they choose, which is the moment its extra context is worth the extra time.
 *
 * **It is never autoplaying and never looping.** These are recordings of real people in
 * distress. A wall of them playing on loop turns an alert queue into ambient footage,
 * which is both worse for triage and worse for the people in it. `preload="none"` means
 * nothing is fetched until somebody deliberately presses play — so opening this page
 * does not pull a video per row off the engine either.
 *
 * **Absence is a real state with three different causes**, and this says which one
 * rather than rendering an empty box: no clip was recorded, the clip is still being
 * written, or it has aged past its retention window. Only the first two are
 * distinguishable here — `clip_uri` is null for both — so this reports what it can
 * honestly tell apart and leaves the engine's 404 to say the rest.
 */
export function AlertClip({ alertId, clipUri, notifyClipUri }: AlertClipProps) {
  const [showFull, setShowFull] = useState(false)
  const [failed, setFailed] = useState(false)

  if (clipUri === null) {
    return (
      <p className="readout mt-2 mb-0 text-[11.5px] text-dimmer">
        No clip yet — a recording is attached once it finishes writing.
      </p>
    )
  }

  const short = !showFull
  // The short copy is a fallback chain, not a guarantee: the engine serves the full
  // recording when no trim was made, so the label has to say which one is actually
  // playing rather than which one was asked for.
  const playingShort = short && notifyClipUri !== null

  return (
    <div className="mt-2.5">
      <video
        // Remounts on the variant switch. Without it the element keeps the old buffer
        // and shows the first frame of a clip it is no longer playing.
        key={short ? 'short' : 'full'}
        src={alertClipUrl(alertId, { short })}
        controls
        preload="none"
        playsInline
        muted
        onError={() => setFailed(true)}
        onLoadedData={() => setFailed(false)}
        data-testid="alert-clip-video"
        data-variant={short ? 'short' : 'full'}
        className="block max-h-[190px] w-full max-w-[340px] rounded-sm border border-line-2 bg-black"
      />
      <div className="readout mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[11px] text-dimmer">
        <span>{playingShort ? 'first 3 seconds' : 'full recording'}</span>
        {notifyClipUri === null ? null : (
          <button
            type="button"
            onClick={() => setShowFull((current) => !current)}
            className={cn(
              'rounded-sm px-1.5 py-px font-mono text-[10px] uppercase tracking-[0.07em]',
              'border border-line-2 text-dim hover:border-signal-d hover:text-signal',
            )}
          >
            {short ? 'Play full clip' : 'Back to 3s'}
          </button>
        )}
      </div>
      {failed ? (
        <p className="readout mt-1 mb-0 text-[11px] text-caution">
          This clip could not be played. It may have passed its retention window.
        </p>
      ) : null}
    </div>
  )
}
