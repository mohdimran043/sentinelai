import { useCallback, useEffect, useRef, useState } from 'react'
import { startManifestProbe } from '@/live/manifestProbe'
import {
  initialLiveStreamState,
  nextLiveStreamState,
  type LiveStreamEvent,
  type LiveStreamState,
} from '@/live/liveStreamState'

export interface LiveStreamHandle {
  state: LiveStreamState
  /** For the player to call when the media element itself reports a fatal error. */
  reportPlayError: () => void
}

/**
 * Wires `manifestProbe.ts` (network reachability) into `liveStreamState.ts`
 * (the pure reducer) as React state, and gives the caller a stable way to
 * feed in the one event the probe cannot see for itself: a playback error
 * from the media element already attached (`HlsPlayer.tsx`'s job, not this
 * hook's — it owns no video element or `hls.js` instance).
 *
 * `supported` gates the whole thing: an unsupported browser never starts
 * probing at all (see `liveStreamState.ts`'s `initialLiveStreamState`), so it
 * costs nothing beyond the one synchronous capability check `HlsPlayer.tsx`
 * already has to make.
 */
export function useLiveStream(url: string, supported: boolean): LiveStreamHandle {
  const [state, setState] = useState<LiveStreamState>(() => initialLiveStreamState(supported))
  const stateRef = useRef(state)

  const dispatch = useCallback((event: LiveStreamEvent) => {
    stateRef.current = nextLiveStreamState(stateRef.current, event)
    setState(stateRef.current)
  }, [])

  useEffect(() => {
    const initial = initialLiveStreamState(supported)
    stateRef.current = initial
    setState(initial)
    if (!supported) return

    const handle = startManifestProbe(url, {
      onOk: () => dispatch({ type: 'probe-ok' }),
      onFail: () => dispatch({ type: 'probe-fail' }),
    })
    return () => handle.stop()
  }, [url, supported, dispatch])

  const reportPlayError = useCallback(() => dispatch({ type: 'play-error' }), [dispatch])

  return { state, reportPlayError }
}
