/**
 * Polls an HLS playlist URL for reachability, independently of whatever
 * `hls.js` or the native `<video>` element are doing with it. Mirrors
 * `api/eventStream.ts`'s reconnect-with-backoff shape, but for a plain
 * request/response check rather than an open stream: a `GET` (not `HEAD` —
 * mediamtx does not special-case it, and a `GET` is what proves the manifest
 * body is actually there, not just that the route exists) with
 * `cache: 'no-store'` so a browser or intermediary cache can never report a
 * manifest reachable after mediamtx has stopped serving it.
 *
 * Reachable and unreachable get different rhythms. An unreachable stream
 * (mediamtx has no path configured, or a configured path with no publisher
 * yet — `liveStreamState.ts` explains why this build cannot tell those
 * apart) is retried with exponential backoff, because hammering mediamtx
 * changes nothing about whether a publisher exists. A reachable one is
 * rechecked on a slower, fixed interval — just often enough to notice the
 * publisher going away in between `hls.js`'s own error events, which is what
 * actually drives `dropped` in the common case.
 */

const HEALTHY_RECHECK_MS = 12_000
const PROBE_BASE_MS = 2_000
const PROBE_MAX_MS = 20_000

export interface ManifestProbeHandlers {
  onOk: () => void
  onFail: () => void
}

export interface ManifestProbeHandle {
  /** Stops polling for good — no further requests. Call on unmount or URL change. */
  stop: () => void
}

export function startManifestProbe(
  url: string,
  handlers: ManifestProbeHandlers,
): ManifestProbeHandle {
  let stopped = false
  let attempt = 0
  let timer: ReturnType<typeof setTimeout> | null = null
  let controller: AbortController | null = null

  async function tick(): Promise<void> {
    if (stopped) return
    controller = new AbortController()

    let ok: boolean
    try {
      const response = await fetch(url, { signal: controller.signal, cache: 'no-store' })
      ok = response.ok
    } catch (error) {
      if (stopped || (error instanceof DOMException && error.name === 'AbortError')) return
      ok = false
    }
    if (stopped) return

    if (ok) {
      attempt = 0
      handlers.onOk()
      timer = setTimeout(() => void tick(), HEALTHY_RECHECK_MS)
    } else {
      handlers.onFail()
      const delay = Math.min(PROBE_BASE_MS * 2 ** attempt, PROBE_MAX_MS)
      attempt += 1
      timer = setTimeout(() => void tick(), delay)
    }
  }

  void tick()

  return {
    stop() {
      stopped = true
      if (timer !== null) clearTimeout(timer)
      controller?.abort()
    },
  }
}
