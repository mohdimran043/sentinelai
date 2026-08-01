import type { CameraEventsResponse, RecentEventEntry } from '@/api/engineClient'

/**
 * The three states the backend can currently distinguish for the live scene
 * panel, as a discriminated union keyed on `kind` so a renderer can `switch`
 * over it and TypeScript enforces exhaustiveness — see `panelStateFor` below.
 *
 * ## A fourth state the wire cannot yet say
 *
 * `ai-engine/sentinel_ai/adapters/vision/qwen25vl.py`'s `_parse_response`
 * tolerates a VLM reply that ignores the requested JSON shape by falling back
 * to a fixed, safe description — but that fallback is reached by returning
 * normally, not by raising. `orchestrator/scheduler.py`'s `_describe` only sets
 * `description_unavailable=True` when the call itself failed (timeout, OOM, an
 * unloaded model) — never when the call succeeded but came back malformed. So
 * a formatting slip and a real description both arrive on the wire as
 * `latest_description_state: "available"`, `description_unavailable: false`:
 * indistinguishable, by the engine's own admission (see
 * `.superpowers/sdd/live-description-report.md`).
 *
 * This module does not paper over that. No heuristic here sniffs the fallback
 * copy to guess which case occurred — that would fabricate a signal the
 * backend does not provide, and it would silently break the moment the
 * fallback string's wording changes. The gap is left visible instead.
 *
 * When the backend gains a real fourth signal (for example a fourth
 * `latest_description_state` member), the change is: one more member on this
 * union, one more `case` in `panelStateFor`, and one more branch wherever a
 * `PanelState` is switched over — `renderPanelState` in `CameraPage.tsx` today
 * has exactly one such switch. It is not a rewrite.
 */
export type PanelState =
  | { kind: 'none' }
  | { kind: 'available'; event: RecentEventEntry }
  | { kind: 'unavailable'; event: RecentEventEntry }

/**
 * Derives the live panel's state directly from `latest_description_state`
 * rather than re-deriving it from `latest.description_unavailable`, so the
 * two can never drift apart — the engine already computes this discriminator
 * once (`_latest_description_state` in `api/schemas.py`) precisely so nothing
 * downstream has to recompute it and risk disagreeing.
 */
export function panelStateFor(response: CameraEventsResponse): PanelState {
  switch (response.latest_description_state) {
    case 'none':
      return { kind: 'none' }
    case 'available':
      // Contract guarantee (`CameraEventsResponse.latest`'s own description):
      // null exactly when latest_description_state is 'none'. Degrade to
      // 'none' rather than crash if that guarantee is ever violated — an
      // honest "nothing yet" beats a runtime error over a contract slip this
      // module cannot fix.
      return response.latest ? { kind: 'available', event: response.latest } : { kind: 'none' }
    case 'unavailable':
      return response.latest ? { kind: 'unavailable', event: response.latest } : { kind: 'none' }
  }
}
