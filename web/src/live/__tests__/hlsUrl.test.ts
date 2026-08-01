import { describe, expect, it } from 'vitest'
import { hlsPlaylistUrl } from '@/live/hlsUrl'

describe('hlsPlaylistUrl', () => {
  it('joins the base URL and camera id into a mediamtx playlist path', () => {
    expect(hlsPlaylistUrl('http://localhost:8888', 'demo_live')).toBe(
      'http://localhost:8888/demo_live/index.m3u8',
    )
  })

  it('tolerates a trailing slash on the base URL rather than doubling it', () => {
    expect(hlsPlaylistUrl('http://localhost:8888/', 'demo_live')).toBe(
      'http://localhost:8888/demo_live/index.m3u8',
    )
  })

  it('tolerates several trailing slashes', () => {
    expect(hlsPlaylistUrl('http://localhost:8888///', 'demo_live')).toBe(
      'http://localhost:8888/demo_live/index.m3u8',
    )
  })

  it('encodes a camera id that is not a bare URL-safe token', () => {
    expect(hlsPlaylistUrl('http://localhost:8888', 'front door / lobby')).toBe(
      'http://localhost:8888/front%20door%20%2F%20lobby/index.m3u8',
    )
  })

  it('derives a different URL per camera id from the same base, never a shared one', () => {
    const a = hlsPlaylistUrl('http://localhost:8888', 'avenue_01')
    const b = hlsPlaylistUrl('http://localhost:8888', 'demo_live')
    expect(a).not.toBe(b)
  })

  it('is configured through the base URL, not hardcoded to localhost', () => {
    expect(hlsPlaylistUrl('https://mediamtx.example.net:8888', 'avenue_01')).toBe(
      'https://mediamtx.example.net:8888/avenue_01/index.m3u8',
    )
  })
})
