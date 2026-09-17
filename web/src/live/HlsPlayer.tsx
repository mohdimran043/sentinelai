import { useEffect, useMemo, useRef, useState } from 'react'
import type Hls from 'hls.js'
import { MEDIAMTX_BASE_URL } from '@/api/config'
import { snapshotUrl } from '@/api/engineClient'
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

/**
 * The latest decoded frame, refreshed on a timer.
 *
 * Deliberately labelled as what it is. A refreshing still is not live video and an
 * operator must not read it as one — but it is what the engine actually has for a camera
 * with no mediamtx path, and it is far better than the empty panel that was here before.
 *
 * `REFRESH_MS` is the trade: each request costs the engine one JPEG encode, and a second
 * is often enough to see somebody walk through a doorway while being nowhere near a
 * frame rate that would make this pretend to be video.
 */
const REFRESH_MS = 1_000

function SnapshotView({ cameraId, attempt }: { cameraId: string; attempt: number }) {
  const [at, setAt] = useState(() => Date.now())
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const timer = setInterval(() => setAt(Date.now()), REFRESH_MS)
    return () => clearInterval(timer)
  }, [])

  if (failed) {
    return (
      <Absent title="Nothing to show yet">
        This camera has no live video path and has not delivered a frame the engine could
        hold on to. If it was just added, give it a moment; the panel fills itself in.
        Retrying{attempt > 1 ? ` (attempt ${formatCount(attempt)})` : ''}.
      </Absent>
    )
  }

  return (
    <figure className="m-0">
      <img
        src={snapshotUrl(cameraId, at)}
        alt={`Most recent frame from ${cameraId}`}
        data-testid="camera-snapshot"
        onError={() => setFailed(true)}
        onLoad={() => setFailed(false)}
        className="aspect-video w-full rounded-sm bg-void object-contain"
      />
      <figcaption className="muted mt-1 mb-0">
        Latest frame, refreshed every second — this camera has no live video path, so this
        is a still and not a stream.
      </figcaption>
    </figure>
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
      // Not an `Absent` any more. "No live stream" was true and useless on a camera the
      // engine is visibly decoding: an EarthCam page and a local file never pass through
      // mediamtx, so they have no playlist and never will, and the panel sat empty while
      // frames were arriving. The engine already holds the latest frame, so show that.
      return <SnapshotView cameraId={cameraId} attempt={state.attempt} />
    case 'dropped':
      return (
        <Notice tone="caution">
          The live stream dropped. Reconnecting automatically
          {state.attempt > 1 ? ` (attempt ${formatCount(state.attempt)})` : ''}…
        </Notice>
      )
  }
}
