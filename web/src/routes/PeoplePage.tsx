import { useRef, useState } from 'react'
import { useCameras, useDeletePerson, useEnrollFace, usePeople, useUpsertPerson } from '@/api/queries'
import { FaceGallery } from '@/people/FaceGallery'
import type { PersonEntry } from '@/api/engineClient'
import { EngineHttpError } from '@/api/engineClient'
import { Absent } from '@/components/ui/Absent'
import { Notice } from '@/components/ui/Notice'
import { Panel } from '@/components/ui/Panel'
import { Pill } from '@/components/ui/Pill'
import { formatAgo } from '@/lib/format'

function newPersonId(): string {
  return crypto.randomUUID()
}

/**
 * People & authorisation (spec §23).
 *
 * The page is shaped by what it is **not** allowed to show. §12 keeps biometric
 * data out of this API entirely: there are no reference photographs here, because
 * the engine stores none — an enrolment photograph is detected, embedded,
 * encrypted and discarded. A console that displayed a face gallery would be
 * showing something that does not exist.
 *
 * So the visible record is a name, where somebody is authorised, and how many
 * reference faces they have. That last number is the one that earns its place:
 * "1" is usually why somebody is not being recognised at an angle, and the fix is
 * a second reference rather than a lower threshold.
 */
export function PeoplePage() {
  const people = usePeople()
  const cameras = useCameras()
  const upsert = useUpsertPerson()
  const remove = useDeletePerson()
  const enroll = useEnrollFace()
  const [draftName, setDraftName] = useState('')
  const [draftCameras, setDraftCameras] = useState<string[]>([])
  const [enrolling, setEnrolling] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const failure = upsert.error ?? remove.error ?? enroll.error
  const capabilityMissing =
    failure instanceof EngineHttpError && failure.status === 503

  const cameraOptions = cameras.data?.cameras ?? []

  function submit(event: React.FormEvent) {
    event.preventDefault()
    const name = draftName.trim()
    if (!name) return
    upsert.mutate(
      {
        personId: newPersonId(),
        person: {
          display_name: name,
          camera_ids: draftCameras,
          // Both carry server-side defaults, but the generated request type still
          // requires them — sending them explicitly is clearer than asserting
          // around the type, and it documents what a new enrolment starts as.
          status: 'active',
          notes: '',
        },
      },
      {
        onSuccess: () => {
          setDraftName('')
          setDraftCameras([])
        },
      },
    )
  }

  function pickFace(personId: string) {
    setEnrolling(personId)
    fileInput.current?.click()
  }

  function onFileChosen(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    const personId = enrolling
    event.target.value = ''
    setEnrolling(null)
    if (file && personId) enroll.mutate({ personId, image: file })
  }

  return (
    <div>
      <h1>People &amp; authorisation</h1>
      <p className="lede">
        Who is permitted where. Cameras with person authorisation enabled compare
        faces against this list and raise an alert when somebody matches nobody
        on it.
      </p>

      <Notice tone="inert" className="mb-4">
        <strong className="font-semibold text-fg">Your photographs are kept</strong>, so you
        can see who is enrolled and check a match by eye. Enrol several per person — a
        single face-on shot matches a camera at an angle poorly. Each is encrypted with
        the same key as the face embedding, never cached by the browser, and deleted with
        the person. Location and capture-time metadata is stripped on upload.
      </Notice>

      {capabilityMissing ? (
        <Notice tone="breach" className="mb-4">
          {failure.detail}
        </Notice>
      ) : failure ? (
        <Notice tone="breach" className="mb-4">
          {failure.message}
        </Notice>
      ) : null}

      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        onChange={onFileChosen}
        className="sr-only"
        aria-hidden="true"
        tabIndex={-1}
      />

      <Panel className="mb-6">
        <div className="eyebrow mb-3">Enrol somebody</div>
        {/*
          Capped rather than full-bleed. A name is short, and an input stretched
          across a 1250px column tells the operator to expect a paragraph — the
          field's width is the clearest hint a form gives about what belongs in it.
        */}
        <form onSubmit={submit} className="flex max-w-[52ch] flex-col gap-3">
          <label className="flex flex-col gap-1.5">
            <span className="text-[12.5px] text-dim">Name</span>
            <input
              value={draftName}
              onChange={(event) => setDraftName(event.target.value)}
              placeholder="e.g. Dana Whitfield"
              className="rounded-sm border border-line-2 bg-void px-[10px] py-[7px] text-[13px] text-fg placeholder:text-dimmer"
            />
          </label>

          <fieldset className="border-0 p-0">
            <legend className="text-[12.5px] text-dim">Authorised on</legend>
            <p className="muted m-0 mt-0.5 mb-2">
              Choosing none authorises them nowhere. That is the safe default, not an
              oversight — a person with no cameras assigned is permitted nowhere until
              somebody says where.
            </p>
            <div className="flex flex-wrap gap-1.5">
              {cameraOptions.length === 0 ? (
                <span className="muted">No cameras configured.</span>
              ) : (
                cameraOptions.map((camera) => {
                  const on = draftCameras.includes(camera.camera_id)
                  return (
                    <button
                      key={camera.camera_id}
                      type="button"
                      aria-pressed={on}
                      onClick={() =>
                        setDraftCameras((current) =>
                          on
                            ? current.filter((id) => id !== camera.camera_id)
                            : [...current, camera.camera_id],
                        )
                      }
                      className={
                        on
                          ? 'rounded-sm border border-signal-d bg-[#0f1610] px-[9px] py-[4px] font-mono text-[11px] text-signal'
                          : 'rounded-sm border border-line-2 bg-panel-2 px-[9px] py-[4px] font-mono text-[11px] text-dim hover:text-fg'
                      }
                    >
                      {camera.label}
                    </button>
                  )
                })
              )}
            </div>
          </fieldset>

          <div>
            <button
              type="submit"
              disabled={!draftName.trim() || upsert.isPending}
              className="rounded-sm border border-signal-d bg-[#0f1610] px-[13px] py-[6px] font-mono text-[11.5px] uppercase tracking-[0.08em] text-signal disabled:opacity-50"
            >
              {upsert.isPending ? 'Enrolling…' : 'Enrol'}
            </button>
          </div>
        </form>
      </Panel>

      <div className="eyebrow mb-2.5">Enrolled</div>
      {people.isPending ? (
        <div className="h-[120px] animate-pulse rounded-md border border-line bg-panel motion-reduce:animate-none" />
      ) : people.isError ? (
        <Absent title="Roster unavailable">{people.error.message}</Absent>
      ) : people.data.people.length === 0 ? (
        <Absent title="Nobody is enrolled.">
          Until somebody is, a camera with person authorisation enabled treats
          everybody it sees as unrecognised.
        </Absent>
      ) : (
        <div className="flex flex-col gap-2">
          {people.data.people.map((person) => (
            <PersonCard
              key={person.person_id}
              person={person}
              busy={enroll.isPending || remove.isPending}
              onEnrollFace={() => pickFace(person.person_id)}
              onDelete={() => remove.mutate(person.person_id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function PersonCard({
  person,
  busy,
  onEnrollFace,
  onDelete,
}: {
  person: PersonEntry
  busy: boolean
  onEnrollFace: () => void
  onDelete: () => void
}) {
  const thin = person.reference_faces < 2
  // Collapsed by default, and the gallery only fetches when open. Reference faces are
  // biometric data: a roster that pulled every one of them to draw a list of names
  // would move far more of it across the network than anybody asked to see.
  const [showFaces, setShowFaces] = useState(false)
  return (
    <article className="rounded-md border border-line bg-panel p-[14px_16px]">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-[15px]">{person.display_name}</h3>
        <Pill tone={person.status === 'active' ? 'nominal' : 'inert'}>{person.status}</Pill>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        {person.camera_ids.length === 0 ? (
          <span className="muted">Authorised nowhere</span>
        ) : (
          person.camera_ids.map((id) => (
            <span
              key={id}
              className="rounded-sm border border-line-2 bg-panel-2 px-[7px] py-[2px] font-mono text-[10.5px] text-dim"
            >
              {id}
            </span>
          ))
        )}
      </div>

      <div className="readout mt-2 flex flex-wrap items-center gap-x-3 text-[11.5px] text-dimmer">
        <span>
          {person.reference_faces} reference {person.reference_faces === 1 ? 'face' : 'faces'}
        </span>
        {person.expires_at ? <span>expires {formatAgo(person.expires_at)}</span> : null}
      </div>

      {thin && person.reference_faces > 0 ? (
        <p className="muted mt-2 mb-0">
          One reference only. A single face-on photograph matches a camera at an angle
          poorly — a second reference usually helps more than lowering the threshold.
        </p>
      ) : null}

      <FaceGallery personId={person.person_id} open={showFaces} />

      <div className="mt-3 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => setShowFaces((open) => !open)}
          aria-expanded={showFaces}
          className="rounded-sm border border-line-2 bg-panel-2 px-[11px] py-[5px] font-mono text-[11px] uppercase tracking-[0.08em] text-fg hover:border-signal-d hover:text-signal"
        >
          {showFaces ? 'Hide faces' : `Show faces (${person.reference_faces})`}
        </button>
        <button
          type="button"
          onClick={onEnrollFace}
          disabled={busy}
          className="rounded-sm border border-line-2 bg-panel-2 px-[11px] py-[5px] font-mono text-[11px] uppercase tracking-[0.08em] text-fg hover:border-signal-d hover:text-signal disabled:opacity-50"
        >
          Add a reference face
        </button>
        <button
          type="button"
          onClick={onDelete}
          disabled={busy}
          className="rounded-sm border border-line-2 bg-panel-2 px-[11px] py-[5px] font-mono text-[11px] uppercase tracking-[0.08em] text-dim hover:border-[#5c231b] hover:text-breach disabled:opacity-50"
        >
          Delete &amp; erase faces
        </button>
      </div>
    </article>
  )
}
