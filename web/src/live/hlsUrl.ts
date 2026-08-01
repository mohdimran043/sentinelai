/**
 * Derives a camera's mediamtx HLS playlist URL from its id.
 *
 * mediamtx's path name is exactly the camera's `id` — that is the whole point
 * of `ai-engine/cameras.example.json`'s normalisation ("Every camera reaches
 * the engine as rtsp://, whatever the upstream is. mediamtx does the
 * normalising ... so RtspSource is the only network path in the engine"). The
 * same path serves HLS on mediamtx's HTTP port, at `<path>/index.m3u8`
 * (verified directly: `http://localhost:8888/demo_live/index.m3u8`).
 *
 * Deriving the URL this way does NOT mean every camera id resolves to a
 * working stream. `replay_01` in that same file is a plain filesystem path
 * fed straight into the engine's `FileSource` — mediamtx never hears about it
 * at all, so `hlsPlaylistUrl(base, 'replay_01')` is a well-formed URL that
 * will never answer. This function only says where a stream WOULD be if
 * mediamtx is serving one; whether it actually is belongs to
 * `liveStreamState.ts`, which is honest about not being able to tell "no path
 * configured" apart from "path configured, nothing publishing yet" over HTTP
 * alone.
 */
export function hlsPlaylistUrl(baseUrl: string, cameraId: string): string {
  return `${baseUrl.replace(/\/+$/, '')}/${encodeURIComponent(cameraId)}/index.m3u8`
}
