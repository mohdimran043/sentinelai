/**
 * The recorder console's sections — the ones this console is the right place for.
 *
 * Six of the recorder's ten were removed. Four because the AI engine already answers
 * the same question one group up the rail, and two screens answering it differently is
 * how an operator ends up trusting the wrong one; two more because nothing in this
 * deployment drives them:
 *
 * | Removed | Because |
 * |---|---|
 * | Alerts | the engine's alert queue, which is also the one you can acknowledge |
 * | Cameras | the camera wall, and the only place you can add or configure one |
 * | Models | AI system, which reports what is actually loaded and its VRAM |
 * | Capabilities | the per-camera capability panel, which can also change them |
 * | Notifications | it listed the recorder's own routing table, which no camera on this engine is wired to — the routing that actually fires is `notify_on` in the capability panel |
 * | Masks | the engine never asks the recorder for a frame, so a mask drawn here changed nothing it observes |
 *
 * The four that remain have no engine equivalent: they are about the recorder as a
 * recorder — what it wrote, what it kept, what it is configured to do. The routes are
 * gone with the links; `/recorder/masks` now falls through to the unknown-section page
 * rather than rendering a screen nothing links to.
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
    path: 'report',
    label: 'Day report',
    built: true,
    endpoints: ['GET /api/report?camera=&date='],
    summary:
      'What the recorder can account for on one camera on one day: recorded coverage against expected, the integrity state behind every gap, blind spots the mask hides from all of it, and retrieval holes where the underlying footage is already gone.',
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
    path: 'storage',
    label: 'Storage',
    built: true,
    endpoints: ['GET /api/storage/untracked'],
    summary:
      'Footage on disk that the segment index has no record of, per camera, with byte totals and the recorder’s explanation of why none of it is ever deleted automatically.',
  },
]

/**
 * Where `/recorder` lands, and where an unrecognised section is sent.
 *
 * One constant rather than two string literals, because the two have to agree and the
 * cost of them disagreeing is not a broken link but an infinite one: a redirect aimed
 * at a section this console no longer has resolves to the `:section` placeholder, which
 * redirects to the same place again. That is precisely what happened when this pointed
 * at `/recorder/alerts` and the alerts section was removed.
 */
export const RECORDER_LANDING_PATH = 'report'

export function findRecorderSection(path: string): RecorderSection | undefined {
  return RECORDER_SECTIONS.find((section) => section.path === path)
}
