/**
 * Whether this browser can play HLS at all, and by which of the two routes.
 *
 * Safari (and other WebKit engines) play `.m3u8` through a plain `<video>`
 * element with no help from JavaScript — `video.canPlayType(...)` says so.
 * Every other current browser needs Media Source Extensions, which is what
 * `hls.js` builds an HLS demuxer on top of. Checking for `MediaSource` here,
 * rather than importing `hls.js` just to call its own `Hls.isSupported()`,
 * means a browser that can play neither way costs this page nothing: no
 * network request for the library, no MSE object created and immediately
 * discarded (`ManagedMediaSource` covers iOS Safari in low-power mode, which
 * drops plain `MediaSource` but keeps a managed variant).
 */
export function supportsNativeHls(video: HTMLVideoElement): boolean {
  return video.canPlayType('application/vnd.apple.mpegurl') !== ''
}

export function supportsMediaSourceHls(): boolean {
  return typeof window !== 'undefined' && ('MediaSource' in window || 'ManagedMediaSource' in window)
}

export function canPlayHls(video: HTMLVideoElement): boolean {
  return supportsNativeHls(video) || supportsMediaSourceHls()
}
