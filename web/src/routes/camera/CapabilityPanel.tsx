import { useState } from 'react'
import type { Capability, CameraStatus, ConcernKind } from '@/api/engineClient'
import { EngineHttpError } from '@/api/engineClient'
import { useUpdateCamera } from '@/api/queries'
import { Notice } from '@/components/ui/Notice'
import { Panel } from '@/components/ui/Panel'
import { CAPABILITY_COST, CAPABILITY_LABELS, CAPABILITY_ORDER } from '@/lib/alerts'
import { CONCERN_KINDS } from '@/lib/cameraEdit'
import { humanizeEnum } from '@/lib/format'

/**
 * What this camera does, and what it tells a person about (spec §24).
 *
 * These two used to be separate panels — "Watching for" here, "Notifies on" in the
 * record form — and they read as the same thing twice, because both are a list of
 * checkboxes about what a camera cares about. They are not the same thing, and the
 * difference is exactly the sort that is invisible until it bites: **capabilities
 * decide what runs, `notify_on` decides what reaches a human.** A camera with
 * `fall_detection` on and `collapse` unchecked detects falls perfectly and tells
 * nobody about them.
 *
 * So they are one panel now, in that order, with the relationship stated between them
 * and the combination above called out where it occurs. They stay two lists rather than
 * one because they do not correspond: there are seven capabilities and six concern
 * kinds, `anomaly_detection` has no concern of its own, and `medication` belongs to no
 * capability. Forcing them into a single row each would invent a mapping the engine
 * does not have.
 *
 * Three things this panel is careful about, all of them consequences of the
 * engine's own honesty about capabilities:
 *
 * **It shows the cost.** §13's promise is that a disabled capability loads no
 * model at all. That only means something to an operator who can see what
 * enabling one would cost, so each row carries its models — including "No model"
 * for camera health, which is the row most worth switching on everywhere.
 *
 * **Enabling can be refused.** Which models exist is fixed when the engine
 * starts, so switching on a capability whose model was never loaded answers 409
 * and names the restart. The panel surfaces that verbatim rather than rewording
 * it, because the engine's message already explains what to do.
 *
 * **Person authorisation is marked.** It is the one capability whose being on is
 * a decision about people rather than about compute, and the panel says so where
 * the operator is making it rather than in documentation they will not open.
 */
export function CapabilityPanel({
  camera,
  writable,
}: {
  camera: CameraStatus
  writable: boolean
}) {
  const update = useUpdateCamera(camera.camera_id)
  // Local until saved, so toggling three capabilities is one write rather than
  // three — and so a refused save leaves the operator's intended set on screen to
  // correct rather than silently reverting it.
  const [draft, setDraft] = useState<Capability[] | null>(null)
  const [notifyDraft, setNotifyDraft] = useState<ConcernKind[] | null>(null)
  const current = draft ?? camera.capabilities
  const currentNotify = notifyDraft ?? camera.notify_on
  const dirty =
    (draft !== null && !sameSet(draft, camera.capabilities)) ||
    (notifyDraft !== null && !sameSet(notifyDraft, camera.notify_on))

  // The combination the merge exists to surface: this camera is watching for people
  // collapsing and would not tell anybody if it saw one.
  const detectsFalls = current.includes('fall_detection')
  const silentOnCollapse = detectsFalls && !currentNotify.includes('collapse')

  const refusal =
    update.error instanceof EngineHttpError && update.error.status === 409
      ? update.error.detail
      : null

  function toggleKind(kind: ConcernKind) {
    setNotifyDraft((previous) => {
      const base = previous ?? camera.notify_on
      return base.includes(kind) ? base.filter((item) => item !== kind) : [...base, kind]
    })
  }

  function toggle(capability: Capability) {
    setDraft((previous) => {
      const base = previous ?? camera.capabilities
      return base.includes(capability)
        ? base.filter((item) => item !== capability)
        : [...base, capability]
    })
  }

  return (
    <Panel>
      <div className="eyebrow mb-1">Watching for</div>
      <p className="muted mt-0 mb-3 max-w-[70ch]">
        A capability that is off loads no model and costs nothing. Turning one on
        needs its model already loaded by this engine — which is decided at startup.
      </p>

      {refusal ? (
        <Notice tone="caution" className="mb-3">
          {refusal}
        </Notice>
      ) : update.error ? (
        <Notice tone="breach" className="mb-3">
          {update.error.message}
        </Notice>
      ) : null}

      <ul className="m-0 flex list-none flex-col gap-px overflow-hidden rounded-sm border border-line p-0">
        {CAPABILITY_ORDER.map((capability) => {
          const on = current.includes(capability)
          const sensitive = capability === 'person_authorization'
          return (
            <li key={capability} className="bg-panel-2">
              <label
                className={
                  writable
                    ? 'flex cursor-pointer items-center gap-3 px-[11px] py-[9px] hover:bg-[#161f26]'
                    : 'flex items-center gap-3 px-[11px] py-[9px] opacity-70'
                }
              >
                <input
                  type="checkbox"
                  checked={on}
                  disabled={!writable || update.isPending}
                  onChange={() => toggle(capability)}
                  className="size-[14px] shrink-0 accent-[#76b900]"
                />
                <span className="min-w-0 flex-1">
                  <span className="block text-[13px] text-fg">
                    {CAPABILITY_LABELS[capability]}
                    {sensitive ? (
                      <span
                        className="ml-2 rounded-sm border border-[#574000] bg-[#171304] px-[5px] py-px font-mono text-[9.5px] uppercase tracking-[0.06em] text-caution"
                        title="Identifies individual people. Enable only where that is justified."
                      >
                        biometric
                      </span>
                    ) : null}
                  </span>
                  <span className="block font-mono text-[10.5px] text-dimmer">
                    {CAPABILITY_COST[capability]}
                  </span>
                </span>
              </label>
            </li>
          )
        })}
      </ul>

      <div className="eyebrow mt-5 mb-1">Tell a person about</div>
      <p className="muted mt-0 mb-3 max-w-[70ch]">
        Which welfare concerns reach a human, out of what the capabilities above
        detect. This is a different question from the one above and the two can
        disagree: detection still happens, clips are still recorded and events are
        still published — a concern left unchecked simply tells nobody.
      </p>

      {silentOnCollapse ? (
        <Notice tone="caution" className="mb-3">
          This camera watches for falls and would not tell anyone about one. Check{' '}
          <strong className="font-semibold text-fg">Collapse</strong> below, or turn off
          fall detection above — detecting something nobody is told about is the one
          combination these two lists make easy to reach by accident.
        </Notice>
      ) : null}

      <ul className="m-0 grid list-none grid-cols-2 gap-px overflow-hidden rounded-sm border border-line p-0">
        {CONCERN_KINDS.map((kind) => (
          <li key={kind} className="bg-panel-2">
            <label
              className={
                writable
                  ? 'flex cursor-pointer items-center gap-3 px-[11px] py-[9px] hover:bg-[#161f26]'
                  : 'flex items-center gap-3 px-[11px] py-[9px] opacity-70'
              }
            >
              <input
                type="checkbox"
                checked={currentNotify.includes(kind)}
                disabled={!writable || update.isPending}
                onChange={() => toggleKind(kind)}
                className="size-[14px] shrink-0 accent-[#76b900]"
              />
              <span className="text-[13px] text-fg">{humanizeEnum(kind)}</span>
            </label>
          </li>
        ))}
      </ul>

      {currentNotify.length === 0 ? (
        <p className="muted mt-2 mb-0" data-testid="capability-panel-muted-warning">
          Nothing checked means this camera notifies nobody. It keeps detecting, keeps
          recording clips and keeps publishing events — it just stops telling anyone.
        </p>
      ) : null}

      {!writable ? (
        <p className="muted mt-3 mb-0">
          Editing is off on this engine. Set{' '}
          <code className="font-mono text-[11.5px]">SENTINEL_ENABLE_CAMERA_WRITES=true</code>{' '}
          only where everything that can reach the port is permitted to reconfigure
          cameras — there is no authentication in front of it.
        </p>
      ) : (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={!dirty || update.isPending}
            onClick={() => {
              // One PATCH for both lists. They are saved together because they are
              // read together: an operator who turns on fall detection and checks
              // Collapse has made one decision, not two.
              update.mutate(
                {
                  ...(draft !== null ? { capabilities: draft } : {}),
                  ...(notifyDraft !== null ? { notify_on: notifyDraft } : {}),
                },
                {
                  onSuccess: () => {
                    setDraft(null)
                    setNotifyDraft(null)
                  },
                },
              )
            }}
            className="rounded-sm border border-signal-d bg-[#0f1610] px-[13px] py-[6px] font-mono text-[11.5px] uppercase tracking-[0.08em] text-signal disabled:opacity-40"
          >
            {update.isPending ? 'Saving…' : 'Save'}
          </button>
          {dirty ? (
            <button
              type="button"
              onClick={() => {
                setDraft(null)
                setNotifyDraft(null)
              }}
              className="font-mono text-[11px] uppercase tracking-[0.07em] text-dim underline-offset-2 hover:text-fg hover:underline"
            >
              Discard
            </button>
          ) : null}
          {current.length === 0 ? (
            <span className="muted">
              Nothing selected — this camera will be watched and decoded, and no model
              will run on it.
            </span>
          ) : null}
        </div>
      )}
    </Panel>
  )
}

function sameSet(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && [...left].sort().join() === [...right].sort().join()
}
