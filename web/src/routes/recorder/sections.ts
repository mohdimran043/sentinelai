/**
 * The recorder console's sections, in the order the recorder's own rail lists
 * them (read off its shipped bundle: Dashboard, Cameras, Day report, Alerts,
 * Configure, Footage, Notifications, Models, Storage, Capabilities).
 *
 * `built: false` means the route exists and the rail shows the section, but the
 * screen behind it is a stated placeholder rather than a working page. It is
 * marked in the rail as well as on the page: a nav item that silently leads
 * nowhere useful is the same lie as a tile that renders a disabled capability
 * as if it worked.
 */
export interface RecorderSection {
  /** Path relative to `/recorder`. */
  path: string
  label: string
  built: boolean
  /** The endpoint(s) this section reads. Shown on the placeholder so the next pass knows where to start. */
  endpoints: string[]
  /** What the section will show, in the operator's terms. */
  summary: string
}

export const RECORDER_SECTIONS: readonly RecorderSection[] = [
  {
    path: 'cameras',
    label: 'Cameras',
    built: true,
    endpoints: ['GET /api/cameras', 'GET /api/status'],
    summary:
      'The full camera record the recorder holds: mode, space type, source, resolution, mask file, pre-roll, capacity, and the audio and face-recognition flags — several of which the recorder itself says are stored but read by nothing.',
  },
  {
    path: 'report',
    label: 'Day report',
    built: true,
    endpoints: ['GET /api/report?camera=&date='],
    summary:
      'What the recorder can account for on one camera on one day: recorded coverage against expected, the integrity state behind every gap, blind spots the mask hides from all of it, and retrieval holes where the underlying footage is already gone.',
  },
  {
    path: 'alerts',
    label: 'Alerts',
    built: true,
    endpoints: ['GET /api/alerts'],
    summary: 'Every alert raised about a camera, and what happened when delivery was attempted.',
  },
  {
    path: 'configure',
    label: 'Configure',
    built: true,
    endpoints: ['GET /api/settings'],
    summary:
      'Whether inference runs on this machine or offsite, whether sending frames offsite has been acknowledged, and the recorder’s warning about what that means in a custodial setting.',
  },
  {
    path: 'footage',
    label: 'Footage',
    built: true,
    endpoints: [
      'GET /api/snapshot/{camera_id}',
      'GET /api/journal/{camera_id}',
      'GET /api/journal/{camera_id}/{date}',
    ],
    summary:
      'There is no segment or clip index — /api/segments and /api/cameras/{id}/clips both 404. This assembles the most recent frame the recorder can produce with the sealed daily journal and its full revision history.',
  },
  {
    path: 'notifications',
    label: 'Notifications',
    built: true,
    endpoints: ['GET /api/notifications'],
    summary:
      'The sixteen alert types with their descriptions, which channels each is routed to, and which eight are locked so an operator may never suppress them.',
  },
  {
    path: 'models',
    label: 'Models',
    built: true,
    endpoints: ['GET /api/models'],
    summary: 'Which model reads the footage, where it runs, and what it costs.',
  },
  {
    path: 'storage',
    label: 'Storage',
    built: true,
    endpoints: ['GET /api/storage/untracked'],
    summary:
      'Footage on disk that the segment index has no record of, per camera, with byte totals and the recorder’s explanation of why none of it is ever deleted automatically.',
  },
  {
    path: 'capabilities',
    label: 'Capabilities',
    built: true,
    endpoints: ['GET /api/capabilities'],
    summary:
      'The thirteen capabilities the recorder declares, six available and seven not, each with the recorder’s own statement of what it can and cannot do.',
  },
  {
    path: 'masks',
    label: 'Masks',
    built: true,
    endpoints: [
      'GET /api/masks/{camera_id}',
      'GET /api/masks/{camera_id}/calibration-frame',
      'POST /api/masks/{camera_id}/validate',
      'PUT /api/masks/{camera_id}',
    ],
    summary:
      'Privacy regions that are never observed and never recorded, drawn over the camera’s own calibration frame and checked against the recorder’s own polygon rules before anything is saved. The recorder refuses to serve an unmasked frame for a camera under observation, and this screen honours that refusal rather than drawing over the (already masked) live snapshot instead.',
  },
]

export function findRecorderSection(path: string): RecorderSection | undefined {
  return RECORDER_SECTIONS.find((section) => section.path === path)
}
