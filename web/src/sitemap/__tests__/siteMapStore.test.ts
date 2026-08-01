import { beforeEach, describe, expect, it } from 'vitest'
import { SITE_MAP_STORAGE_KEY, useSiteMapStore } from '@/sitemap/siteMapStore'

/**
 * The store is a module-level singleton (zustand's usual shape), so it
 * persists across `it` blocks in this file unless reset — both its
 * in-memory state and the real `localStorage` it writes through to.
 */
beforeEach(() => {
  localStorage.clear()
  useSiteMapStore.setState({ placements: {}, floorplanImage: null })
})

describe('useSiteMapStore', () => {
  it('starts with no placements and no floorplan image', () => {
    expect(useSiteMapStore.getState().placements).toEqual({})
    expect(useSiteMapStore.getState().floorplanImage).toBeNull()
  })

  it('setPlacement records a camera position keyed by camera_id', () => {
    useSiteMapStore.getState().setPlacement('room_4b', { x: 0.3, y: 0.7 })
    expect(useSiteMapStore.getState().placements).toEqual({ room_4b: { x: 0.3, y: 0.7 } })
  })

  it('clamps an out-of-range placement rather than storing a marker off the plan', () => {
    useSiteMapStore.getState().setPlacement('room_4b', { x: 1.5, y: -2 })
    expect(useSiteMapStore.getState().placements.room_4b).toEqual({ x: 1, y: 0 })
  })

  it('setPlacement for a second camera does not disturb the first', () => {
    useSiteMapStore.getState().setPlacement('room_4b', { x: 0.1, y: 0.1 })
    useSiteMapStore.getState().setPlacement('corridor_1', { x: 0.9, y: 0.9 })
    expect(useSiteMapStore.getState().placements).toEqual({
      room_4b: { x: 0.1, y: 0.1 },
      corridor_1: { x: 0.9, y: 0.9 },
    })
  })

  it('removePlacement removes only the named camera', () => {
    useSiteMapStore.getState().setPlacement('room_4b', { x: 0.1, y: 0.1 })
    useSiteMapStore.getState().setPlacement('corridor_1', { x: 0.9, y: 0.9 })
    useSiteMapStore.getState().removePlacement('room_4b')
    expect(useSiteMapStore.getState().placements).toEqual({ corridor_1: { x: 0.9, y: 0.9 } })
  })

  it('setFloorplanImage stores and clears the data URL', () => {
    useSiteMapStore.getState().setFloorplanImage('data:image/png;base64,AAA')
    expect(useSiteMapStore.getState().floorplanImage).toBe('data:image/png;base64,AAA')
    useSiteMapStore.getState().setFloorplanImage(null)
    expect(useSiteMapStore.getState().floorplanImage).toBeNull()
  })

  it('actually writes through to localStorage under the documented key, proving this is real persistence and not just in-memory state', () => {
    useSiteMapStore.getState().setPlacement('room_4b', { x: 0.25, y: 0.6 })
    const raw = localStorage.getItem(SITE_MAP_STORAGE_KEY)
    expect(raw).not.toBeNull()
    const parsed = JSON.parse(raw!)
    expect(parsed.state.placements).toEqual({ room_4b: { x: 0.25, y: 0.6 } })
  })
})
