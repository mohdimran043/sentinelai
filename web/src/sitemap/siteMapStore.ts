import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { clampPoint, type NormalizedPoint } from '@/sitemap/placement'

/**
 * Where a camera sits on the site map, and the floorplan image itself.
 *
 * **Deliberately client-side (`localStorage`), not a backend model — read
 * this before assuming it should be "real" persistence:**
 *
 * There is no site or placement entity on the AI engine today — only `zone`
 * on `GET /cameras`. Building one now would be new, unauthenticated write
 * surface: `POST`/`PUT` with no JWT (Phase 1C) means any client on the
 * network could move another operator's cameras or overwrite their
 * floorplan. Shipping client-side first gets the feature working today,
 * with an honest limitation instead of a false one: **placements are
 * per-browser.** They do not follow an operator to a different machine and
 * are not shared between operators watching the same site — `SiteMapPage`
 * says so in its own copy, out loud, rather than letting an operator
 * assume a colleague sees the same arrangement.
 *
 * The shape is deliberately the shape a real backend endpoint would want:
 * normalized `x`/`y` in 0..1 (works at any plan size, no pixel migration
 * later) keyed by `camera_id` (the same id `GET /cameras` already uses).
 * Moving this to `PUT /site-map/placements/{camera_id}` later is a storage
 * swap, not a redesign — this module is the one place that would change.
 */
export interface SiteMapState {
  /** Placements, keyed by `camera_id`. Per-browser — see the module doc comment. */
  placements: Record<string, NormalizedPoint>
  /** The operator-supplied floorplan image, as a data URL, or `null` for the schematic empty state. Per-browser, same as `placements`. */
  floorplanImage: string | null
  setPlacement: (cameraId: string, point: NormalizedPoint) => void
  removePlacement: (cameraId: string) => void
  setFloorplanImage: (dataUrl: string | null) => void
}

/** Bumped if the persisted shape ever changes incompatibly. */
export const SITE_MAP_STORAGE_KEY = 'sentinelai.sitemap.v1'

export const useSiteMapStore = create<SiteMapState>()(
  persist(
    (set) => ({
      placements: {},
      floorplanImage: null,
      setPlacement: (cameraId, point) =>
        set((state) => ({
          placements: { ...state.placements, [cameraId]: clampPoint(point) },
        })),
      removePlacement: (cameraId) =>
        set((state) => {
          const next = { ...state.placements }
          delete next[cameraId]
          return { placements: next }
        }),
      setFloorplanImage: (dataUrl) => set({ floorplanImage: dataUrl }),
    }),
    {
      // Explicit localStorage, not the session store's sessionStorage
      // (`store/session.ts`): a camera arrangement is worth keeping across a
      // closed tab or a restarted browser, unlike the operator name label,
      // which is scoped to the current session on purpose.
      name: SITE_MAP_STORAGE_KEY,
    },
  ),
)
