/**
 * The state machine behind the camera page's live video player, as a
 * discriminated union keyed on `kind` so a renderer can `switch` over it and
 * TypeScript enforces exhaustiveness — the same shape `descriptionPanel.ts`
 * uses for the live scene panel.
 *
 * ## Why `unavailable` cannot say "no path" vs "no publisher yet"
 *
 * A camera built straight from a local file (`replay_01` in
 * `ai-engine/cameras.example.json`) has no mediamtx path at all. A camera
 * mediamtx is configured to receive from a publisher (`avenue_01`, before
 * `ffmpeg` starts pushing to it) has a path, just nothing arriving on it yet.
 * These are genuinely different situations — one will never come up, the
 * other will the moment `ffmpeg` starts — but nothing served over HTTP tells
 * them apart. Verified directly:
 *
 * ```
 * curl -i http://localhost:8888/replay_01/index.m3u8   -> 404, instant
 * curl -i http://localhost:8888/avenue_01/index.m3u8   -> 404, instant
 * ```
 *
 * Byte-for-byte identical responses. This console has no other channel to
 * `ai-engine`'s camera config — `cameras.json` is a file on the engine host,
 * not something any endpoint exposes, and adding one is out of scope for this
 * slice (touch `web/` only). So `unavailable` says only what is honestly true
 * of both cases: no stream answers right now. It does not guess which one it
 * is, the same way `descriptionPanel.ts` declines to sniff a description
 * string to guess whether the VLM's reply was malformed.
 *
 * The distinction this build DOES make, and can make honestly, is `dropped`
 * vs `unavailable`: `dropped` means this mount already had the stream playing
 * and it just stopped; `unavailable` means it has not come up yet in this
 * mount at all. Both retry forever on the same backoff (`manifestProbe.ts`),
 * so a camera that starts publishing after the page loads recovers into
 * `live` on its own with no reload — the two states differ only in the
 * message shown, never in whether recovery is possible.
 */
export type LiveStreamState =
  | { kind: 'unsupported' }
  | { kind: 'checking' }
  | { kind: 'unavailable'; attempt: number }
  | { kind: 'live' }
  | { kind: 'dropped'; attempt: number }

export type LiveStreamEvent =
  | { type: 'probe-ok' }
  | { type: 'probe-fail' }
  | { type: 'play-error' }

/**
 * `unsupported` is decided once, from the browser's own capabilities, before
 * a single byte is fetched — see `hlsSupport.ts`. It is never entered from
 * any other state and no event below ever leaves it: a browser's HLS support
 * does not change mid-session, so there is nothing to retry.
 */
export function initialLiveStreamState(supported: boolean): LiveStreamState {
  return supported ? { kind: 'checking' } : { kind: 'unsupported' }
}

export function nextLiveStreamState(
  state: LiveStreamState,
  event: LiveStreamEvent,
): LiveStreamState {
  if (state.kind === 'unsupported') return state // terminal — see above

  switch (event.type) {
    case 'probe-ok':
      return { kind: 'live' }
    case 'probe-fail':
      if (state.kind === 'unavailable') return { kind: 'unavailable', attempt: state.attempt + 1 }
      if (state.kind === 'dropped') return { kind: 'dropped', attempt: state.attempt + 1 }
      if (state.kind === 'live') return { kind: 'dropped', attempt: 1 }
      return { kind: 'unavailable', attempt: 1 } // from 'checking'
    case 'play-error':
      // A fatal playback error only means anything once something was
      // playing; if it fires while not live, the manifest probe already owns
      // the state and this is dropped as a stale signal (e.g. a leftover
      // event from a just-torn-down player).
      return state.kind === 'live' ? { kind: 'dropped', attempt: 1 } : state
  }
}
