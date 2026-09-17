import { faceImageUrl, type EnrolledFaceEntry } from '@/api/engineClient'
import { useDeleteFace, useFaces } from '@/api/queries'
import { Notice } from '@/components/ui/Notice'
import { formatAgo } from '@/lib/format'

/**
 * The reference photographs on one person's record.
 *
 * The photographs as uploaded, re-encoded and bounded at 1024px, sealed with the same
 * key as the embedding. Not the detector's face crop, which this used to show: a
 * 112-pixel face cannot answer the questions an operator actually has — is this the
 * right person, is this a usable photo, is this the one I meant.
 *
 * Fetched only when open. A roster of fifty people would otherwise pull fifty face lists
 * and every thumbnail in them to draw a table that shows a count.
 */
export function FaceGallery({ personId, open }: { personId: string; open: boolean }) {
  const faces = useFaces(personId, open)
  const remove = useDeleteFace()

  if (!open) return null

  if (faces.isPending) {
    return <p className="muted mt-3 mb-0">Loading reference faces…</p>
  }

  if (faces.isError) {
    return (
      <Notice tone="caution" className="mt-3">
        Could not read this person&apos;s reference faces: {faces.error.message}
      </Notice>
    )
  }

  const entries = faces.data?.faces ?? []
  if (entries.length === 0) {
    return (
      <p className="muted mt-3 mb-0">
        No face enrolled yet, so this person cannot be recognised at all.
      </p>
    )
  }

  return (
    <div className="mt-3">
      <ul className="m-0 flex list-none flex-wrap gap-2 p-0">
        {entries.map((face) => (
          <FaceTile
            key={face.face_id}
            personId={personId}
            face={face}
            busy={remove.isPending}
            onRemove={() => remove.mutate({ personId, faceId: face.face_id })}
          />
        ))}
      </ul>
      {remove.isError ? (
        <Notice tone="caution" className="mt-2">
          Could not remove that reference: {remove.error.message}
        </Notice>
      ) : null}
    </div>
  )
}

function FaceTile({
  personId,
  face,
  busy,
  onRemove,
}: {
  personId: string
  face: EnrolledFaceEntry
  busy: boolean
  onRemove: () => void
}) {
  return (
    <li className="w-[104px] overflow-hidden rounded-sm border border-line-2 bg-panel-2">
      <div className="grid h-[104px] w-full place-items-center bg-panel">
        {face.has_image ? (
          <img
            src={faceImageUrl(personId, face.face_id)}
            /* Not the person's name: an alt attribute is read aloud and copied into
               bug reports, and a roster of names is the part of this page worth not
               spreading further. The tile already sits under the name it belongs to. */
            alt="Enrolled reference face"
            className="h-[104px] w-[104px] object-cover"
            loading="lazy"
          />
        ) : (
          /* A real state, not a loading one: enrolment can keep the embedding and no
             picture, and an operator should know which of the two they are seeing. */
          <span className="px-2 text-center font-mono text-[10px] uppercase tracking-[0.06em] text-dimmer">
            no photo kept
          </span>
        )}
      </div>
      <div className="flex items-center justify-between gap-1 px-[6px] py-[4px]">
        <span className="readout text-[10px] text-dimmer">
          {face.enrolled_at ? formatAgo(face.enrolled_at) : 'date unknown'}
        </span>
        <button
          type="button"
          onClick={onRemove}
          disabled={busy}
          title="Remove this reference face"
          aria-label="Remove this reference face"
          className="rounded-sm px-[5px] font-mono text-[11px] leading-none text-dimmer hover:text-breach disabled:opacity-50"
        >
          ×
        </button>
      </div>
    </li>
  )
}
