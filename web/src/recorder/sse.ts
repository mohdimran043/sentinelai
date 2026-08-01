/**
 * A transport-free parser for the `text/event-stream` wire format (server-sent
 * events). Deliberately not built on the browser's `EventSource`: jsdom (this
 * project's test environment) does not implement it at all, so anything built
 * on it would be untestable here. `fetch` + a `ReadableStream` reader, on the
 * other hand, both exist in this environment and are exactly what MSW can
 * mock — verified directly: a `ReadableStream` response body served through
 * `HttpResponse` reads correctly via `response.body.getReader()` in this
 * project's vitest/jsdom setup.
 *
 * This class is the one piece of the SSE stack that is pure text-in,
 * messages-out — no network, no DOM. `alertStream.ts` owns the fetch loop and
 * reconnection; this owns only "what does a byte stream of this format mean".
 */

export interface SseMessage {
  /** Defaults to `"message"` per the spec when the source sends no `event:` line. */
  event: string
  data: string
}

export class SseStreamParser {
  private buffer = ''

  /**
   * Feed the next chunk exactly as it arrived off the wire. Returns every
   * message the chunk completed. A message that straddles two chunks (the
   * normal case — chunk boundaries and message boundaries are unrelated)
   * yields nothing until the chunk that completes it arrives.
   */
  push(chunk: string): SseMessage[] {
    this.buffer += chunk
    const messages: SseMessage[] = []
    let boundary = this.buffer.indexOf('\n\n')
    // A message is terminated by a blank line, i.e. "\n\n". Drain every
    // complete message currently buffered; leave a trailing partial one for
    // the next push.
    while (boundary !== -1) {
      const raw = this.buffer.slice(0, boundary)
      this.buffer = this.buffer.slice(boundary + 2)
      const message = parseRawMessage(raw)
      if (message) messages.push(message)
      boundary = this.buffer.indexOf('\n\n')
    }
    return messages
  }
}

function parseRawMessage(raw: string): SseMessage | null {
  let event = 'message'
  const dataLines: string[] = []
  for (const line of raw.split('\n')) {
    if (line === '' || line.startsWith(':')) continue // blank/comment (keepalive) lines carry nothing
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    const value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '')
    if (field === 'event') event = value
    else if (field === 'data') dataLines.push(value)
    // `id:` and `retry:` are intentionally ignored: this build reconnects on
    // a fixed backoff rather than resuming by Last-Event-ID, and does not let
    // the server dictate a retry interval. A deliberate simplification, not
    // an oversight — see `alertStream.ts`.
  }
  if (dataLines.length === 0) return null // a pure comment/keepalive message
  return { event, data: dataLines.join('\n') }
}
