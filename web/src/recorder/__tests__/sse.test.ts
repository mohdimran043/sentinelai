import { describe, expect, it } from 'vitest'
import { SseStreamParser } from '@/recorder/sse'

describe('SseStreamParser', () => {
  it('parses a single complete message in one chunk', () => {
    const parser = new SseStreamParser()
    const messages = parser.push('event: backlog\ndata: [1,2,3]\n\n')
    expect(messages).toEqual([{ event: 'backlog', data: '[1,2,3]' }])
  })

  it('buffers a message split across chunks and yields it only once complete', () => {
    const parser = new SseStreamParser()
    expect(parser.push('event: backlog\ndata: [1,2')).toEqual([])
    expect(parser.push(',3]\n\n')).toEqual([{ event: 'backlog', data: '[1,2,3]' }])
  })

  it('yields every message present in a single chunk, in order', () => {
    const parser = new SseStreamParser()
    const messages = parser.push(
      'event: backlog\ndata: [1]\n\ndata: {"id":"a"}\n\ndata: {"id":"b"}\n\n',
    )
    expect(messages).toEqual([
      { event: 'backlog', data: '[1]' },
      { event: 'message', data: '{"id":"a"}' },
      { event: 'message', data: '{"id":"b"}' },
    ])
  })

  it('defaults the event name to "message" when the source sends no event: line', () => {
    const parser = new SseStreamParser()
    expect(parser.push('data: {"id":"a"}\n\n')).toEqual([{ event: 'message', data: '{"id":"a"}' }])
  })

  it('ignores a pure comment (keepalive) message and yields nothing for it', () => {
    const parser = new SseStreamParser()
    expect(parser.push(': ping\n\n')).toEqual([])
  })

  it('does not let a keepalive between two real messages merge or drop either one', () => {
    const parser = new SseStreamParser()
    const messages = parser.push('data: {"id":"a"}\n\n: ping\n\ndata: {"id":"b"}\n\n')
    expect(messages).toEqual([
      { event: 'message', data: '{"id":"a"}' },
      { event: 'message', data: '{"id":"b"}' },
    ])
  })

  it('joins multiple data: lines in one message with a newline, per the spec', () => {
    const parser = new SseStreamParser()
    const messages = parser.push('data: line one\ndata: line two\n\n')
    expect(messages).toEqual([{ event: 'message', data: 'line one\nline two' }])
  })
})
