import { useEffect, useMemo, useRef, useState } from 'react'
import type Hls from 'hls.js'
import { MEDIAMTX_BASE_URL } from '@/api/config'
import { hlsPlaylistUrl } from '@/live/hlsUrl'
import { canPlayHls, supportsNativeHls } from '@/live/hlsSupport'
import { useLiveStream } from '@/live/useLiveStream'
import type { LiveStreamState } from '@/live/liveStreamState'
import { Absent } from '@/components/ui/Absent'
import { Notice } from '@/components/ui/Notice'
import { formatCount } from '@/lib/format'
import { cn } from '@/lib/cn'

/**
 * Plays a camera's live HLS stream straight from mediamtx — never through the
 * AI engine. Video delivery is mediamtx's job; the engine does inference.
 * Routing frames through the engine (a proxy, a snapshot endpoint) would
 * duplicate work mediamtx already does and give the engine a job it has no
 * business doing — see `api/config.ts`'s `MEDIAMTX_BASE_URL` doc comment.
 *
 * Safari plays HLS through the plain `<video>` element with no help; every
 * other current browser needs `hls.js`, imported dynamically only once a
 * manifest is confirmed reachable and only on a browser that needs it, so a
 * Safari visitor — or one whose browser can do neither — never pays for the
 * library at all.
 */
export function HlsPlayer({ cameraId }: { cameraId: string }) {
  const url = useMemo(() => hlsPlaylistUrl(MEDIAMTX_BASE_URL, cameraId), [cameraId])
  const videoRef = useRef<HTMLVideoElement>(null)

  // Decided once, from the browser's own capabilities, before any network
  // request is made — see `hlsSupport.ts` and `liveStreamState.ts`'s note on
  // why `unsupported` never depends on the probe.
  const [supported] = useState(() => canPlayHls(document.createElement('video')))

  const { state, reportPlayError } = useLiveStream(url, supported)

  useEffect(() => {
    if (state.kind !== 'live') return
    const video = videoRef.current
    if (!video) return

    let hls: Hls | null = null
    let cancelled = false

    function onVideoError() {
      reportPlayError()
    }
    video.addEventListener('error', onVideoError)

    const attach = async () => {
      if (supportsNativeHls(video)) {
        video.src = url
        return
      }
      const { default: HlsCtor } = await import('hls.js')
      if (cancelled) return
      if (!HlsCtor.isSupported()) {
        // `supported` above already gated this; the runtime check is the
        // source of truth if the two ever disagree (e.g. a browser flag
        // flips MediaSource off between the mount check and this import).
        reportPlayError()
        return
      }
      hls = new HlsCtor()
      hls.attachMedia(video)
      hls.loadSource(url)
      hls.on(HlsCtor.Events.ERROR, (_event, data) => {
        if (data.fatal) reportPlayError()
      })
    }
    void attach()

    return () => {
      cancelled = true
      video.removeEventListener('error', onVideoError)
      hls?.destroy()
      video.removeAttribute('src')
      video.load()
    }
  }, [state.kind, url, reportPlayError])

  return (
    <div>
      <video
        ref={videoRef}
        data-testid="hls-video"
        aria-label={`Live video for ${cameraId}`}
        className={cn('aspect-video w-full rounded-sm bg-void', state.kind === 'live' ? 'block' : 'hidden')}
        controls
        muted
        playsInline
      />
      {state.kind === 'live' ? null : <LiveVideoFallback state={state} cameraId={cameraId} />}
    </div>
  )
}

function LiveVideoFallback({
  state,
  cameraId,
}: {
  state: Exclude<LiveStreamState, { kind: 'live' }>
  cameraId: string
}) {
  switch (state.kind) {
    case 'unsupported':
      return (
        <Absent title="This browser can't play live video">
          No native HLS support and no Media Source Extensions were detected. Try a current
          Chrome, Firefox, Edge or Safari.
        </Absent>
      )
    case 'checking':
      return (
        <div
          className="flex items-center gap-2.5 rounded-md border border-line bg-panel-2 p-[15px] text-[12.5px] text-dim"
          role="status"
        >
          <span
            className="h-2 w-2 shrink-0 rounded-full bg-dim animate-throb motion-reduce:animate-none"
            aria-hidden="true"
          />
          Checking for a live stream on <code>{cameraId}</code>…
        </div>
      )
    case 'unavailable':
      return (
        <Absent title="No live stream reachable">
          Nothing is being served at this camera&apos;s video path right now — it may not have
          started publishing yet, or may have no live video path configured at all. Retrying
          automatically{state.attempt > 1 ? ` (attempt ${formatCount(state.attempt)})` : ''}.
        </Absent>
      )
    case 'dropped':
      return (
        <Notice tone="caution">
          The live stream dropped. Reconnecting automatically
          {state.attempt > 1 ? ` (attempt ${formatCount(state.attempt)})` : ''}…
        </Notice>
      )
  }
}
