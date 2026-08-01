/**
 * The recorder console's sections, in the order the recorder's own rail lists
 * them (read off its shipped bundle: Dashboard, Cameras, Day report, Alerts,
 * Configure, Footage, Notifications, Models, Storage, Capabilities).
 *
 * "Day report" is not on this console's parity checklist and so is not listed
 * here; nothing else is dropped.
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
    built: false,
    endpoints: ['GET /api/cameras', 'GET /api/status'],
    summary:
      'The full camera record the recorder holds: mode, space type, source, resolution, mask file, pre-roll, capacity, and the audio and face-recognition flags — several of which the recorder itself says are stored but read by nothing.',
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
    built: false,
    endpoints: ['(segment endpoint not yet located)'],
    summary:
      'Recorded video and segment coverage. The recorder answers 404 on /api/segments, so the endpoint has to be found before anything can be built here.',
  },
  {
    path: 'notifications',
    label: 'Notifications',
    built: false,
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
    built: false,
    endpoints: ['GET /api/capabilities'],
    summary:
      'The thirteen capabilities the recorder declares, six available and seven not, each with the recorder’s own statement of what it can and cannot do.',
  },
]

export function findRecorderSection(path: string): RecorderSection | undefined {
  return RECORDER_SECTIONS.find((section) => section.path === path)
}
