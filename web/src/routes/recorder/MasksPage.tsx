import { useEffect, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { Reading } from '@/components/ui/Reading'
import { Notice } from '@/components/ui/Notice'
import { Button } from '@/components/ui/Button'
import { Select } from '@/components/ui/Select'
import { Input } from '@/components/ui/Input'
import { Label } from '@/components/ui/Label'
import { formatCount } from '@/lib/format'
import { cn } from '@/lib/cn'
import { useRecorderCameraAction, useRecorderCameras, useRecorderMask } from '@/recorder/queries'
import {
  recorderCalibrationFrameUrl,
  saveRecorderMask,
  validateRecorderMask,
} from '@/recorder/recorderClient'
import { displayPointToFramePoint, polygonPointsAttr } from '@/recorder/maskGeometry'
import type { RecorderMaskRegion, RecorderMaskValidation } from '@/recorder/recorder.types'

/** `14.561631944444445` -> "14.6%". These are already percentages off the wire, not ratios — `formatPercent` in `lib/format.ts` expects a 0..1 ratio, so it is not reused here. */
function formatMaskPercent(pct: number): string {
  return `${pct.toFixed(1)}%`
}

export function MasksPage() {
  const cameras = useRecorderCameras()
  const cameraList = cameras.data?.cameras ?? []
  const [cameraId, setCameraId] = useState('')
  const firstCameraId = cameraList[0]?.id

  // Default to the first camera once the registry loads. A later reload that
  // returns a different first camera must not silently switch the operator's
  // selection out from under them, so this only fires while nothing is picked.
  // Depends on `firstCameraId` rather than the whole (recreated-on-every-render
  // while pending) `cameraList` array, so it does not re-run every render.
  useEffect(() => {
    if (cameraId === '' && firstCameraId !== undefined) {
      setCameraId(firstCameraId)
    }
  }, [cameraId, firstCameraId])

  return (
    <>
      <PageHeader
        eyebrow="privacy"
        title="Masks"
        lede="Regions of a camera's frame that are never observed and never recorded. Drawn over the camera's own calibration frame — never over the live snapshot — and checked against the recorder's own rules before anything is saved."
      />

      {cameras.isError ? (
        <Notice tone="breach">
          {cameras.error.message} The camera list could not be read, so no camera can be chosen
          below.
        </Notice>
      ) : null}

      <Panel className="mt-3.5">
        <Label htmlFor="mask-camera">Camera</Label>
        <Select
          id="mask-camera"
          value={cameraId}
          onChange={(event) => setCameraId(event.target.value)}
          disabled={cameraList.length === 0}
        >
          {cameraList.length === 0 ? <option value="">No cameras reported</option> : null}
          {cameraList.map((camera) => (
            <option key={camera.id} value={camera.id}>
              {camera.name} ({camera.id})
            </option>
          ))}
        </Select>
      </Panel>

      {/* Keyed on cameraId so switching cameras resets every piece of local
          editor state (draft points, in-flight validation result, ...) instead
          of a stale draft from one camera bleeding into another's. */}
      {cameraId ? <MaskEditor key={cameraId} cameraId={cameraId} /> : null}
    </>
  )
}

function MaskEditor({ cameraId }: { cameraId: string }) {
  const mask = useRecorderMask(cameraId)
  const cameraAction = useRecorderCameraAction()
  const data = mask.data

  const [editedRegions, setEditedRegions] = useState<RecorderMaskRegion[]>([])
  const [seeded, setSeeded] = useState(false)
  const [drawing, setDrawing] = useState(false)
  const [draftPoints, setDraftPoints] = useState<[number, number][]>([])
  const [draftPointX, setDraftPointX] = useState('')
  const [draftPointY, setDraftPointY] = useState('')
  const [draftName, setDraftName] = useState('')
  const [result, setResult] = useState<{ kind: 'validate' | 'save'; body: RecorderMaskValidation } | null>(
    null,
  )
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState<'validate' | 'save' | null>(null)

  // Seed the draft from the saved regions exactly once per camera (this
  // component remounts on camera change — see the `key` above) rather than on
  // every background refetch, so a poll landing mid-edit cannot wipe an
  // operator's unsaved work.
  useEffect(() => {
    if (!seeded && data) {
      setEditedRegions(data.regions)
      setSeeded(true)
    }
  }, [seeded, data])

  if (mask.isError) {
    return (
      <Notice tone="breach" className="mt-3.5">
        {mask.error.message} Nothing below reflects this camera's actual mask configuration until
        the recorder answers again.
      </Notice>
    )
  }

  if (mask.isPending || !data) {
    return <p className="muted mt-3.5">Loading the mask editor…</p>
  }

  const canSeeFrame = data.calibration_available
  const canEdit = data.calibration_available && data.editable
  const dirty = JSON.stringify(editedRegions) !== JSON.stringify(data.regions)

  function addPoint(point: [number, number]) {
    setDraftPoints((current) => [...current, point])
  }

  function startDrawing() {
    setDrawing(true)
    setDraftPoints([])
    setDraftName('')
  }

  function cancelDrawing() {
    setDrawing(false)
    setDraftPoints([])
    setDraftName('')
  }

  function commitRegion() {
    const name = draftName.trim()
    if (name === '' || draftPoints.length < 3) return
    setEditedRegions((current) => [...current, { region_id: name, polygon: draftPoints }])
    cancelDrawing()
  }

  function removeRegion(index: number) {
    setEditedRegions((current) => current.filter((_, i) => i !== index))
  }

  async function handleValidate() {
    setSubmitting('validate')
    setSubmitError(null)
    try {
      const body = await validateRecorderMask(cameraId, editedRegions)
      setResult({ kind: 'validate', body })
    } catch (error) {
      setResult(null)
      setSubmitError(error instanceof Error ? error.message : 'The check could not be completed.')
    } finally {
      setSubmitting(null)
    }
  }

  async function handleSave() {
    setSubmitting('save')
    setSubmitError(null)
    try {
      const body = await saveRecorderMask(cameraId, editedRegions)
      setResult({ kind: 'save', body })
      if (body.valid) {
        await mask.refetch()
      }
    } catch (error) {
      setResult(null)
      setSubmitError(error instanceof Error ? error.message : 'Saving failed.')
    } finally {
      setSubmitting(null)
    }
  }

  return (
    <>
      <div className="mt-3.5 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Reading label="saved regions" value={formatCount(data.regions.length)} />
        <Reading
          label="masked, saved"
          value={data.inventory ? formatMaskPercent(data.inventory.total_pct) : '0.0%'}
        />
        <Reading label="frame size" value={`${formatCount(data.frame_width)}×${formatCount(data.frame_height)}`} />
        <Reading label="regions overlap" value={data.inventory ? (data.inventory.may_overlap ? 'Yes' : 'No') : '—'} />
      </div>

      <Notice tone="inert" className="mt-3.5">
        {data.note}
      </Notice>

      {!canSeeFrame ? (
        <>
          <Notice tone="breach" className="mt-3.5">
            <strong className="block font-mono text-[12.5px] tracking-[0.1em] uppercase">
              No calibration frame available
            </strong>
            <span className="mt-1.5 block">
              {data.calibration_refusal ?? 'This camera cannot serve a frame to draw against right now.'}
            </span>
          </Notice>
          <p className="muted mt-2">
            This screen never draws over the live snapshot to work around that: the snapshot is
            already masked, so drawing against it would either mislead you about what you are
            covering or expose exactly what the mask exists to hide. What is shown below is the
            saved mask only, over a neutral placeholder — nothing can be added or removed until a
            calibration frame is available.
          </p>
          {data.recording ? <StopCameraRemedy cameraId={cameraId} cameraAction={cameraAction} onStopped={() => void mask.refetch()} /> : null}
        </>
      ) : !data.editable ? (
        <Notice tone="caution" className="mt-3.5">
          This camera's mask is not editable. What is shown below is the saved mask only.
        </Notice>
      ) : null}

      <div className="mt-3.5 grid gap-3 lg:grid-cols-[minmax(0,1fr)_280px]">
        <MaskCanvas
          frameWidth={data.frame_width}
          frameHeight={data.frame_height}
          imageUrl={canSeeFrame ? recorderCalibrationFrameUrl(cameraId) : undefined}
          regions={editedRegions}
          draftPoints={draftPoints}
          interactive={canEdit && drawing}
          onAddPoint={addPoint}
        />

        <div className="flex flex-col gap-3">
          <Panel>
            <div className="eyebrow">regions{dirty ? ' · draft, not yet saved' : ''}</div>
            {editedRegions.length === 0 ? (
              <p className="muted mt-2 mb-0">No regions in this draft.</p>
            ) : (
              <ul className="m-0 mt-2 list-none space-y-1.5 p-0">
                {editedRegions.map((region, index) => (
                  <li key={`${region.region_id}-${index}`} className="flex items-center justify-between gap-2 text-[12.5px]">
                    <span className="readout">
                      {region.region_id} <span className="text-dim">· {formatCount(region.polygon.length)} pts</span>
                    </span>
                    {canEdit ? (
                      <Button variant="quiet" size="small" onClick={() => removeRegion(index)}>
                        Remove
                      </Button>
                    ) : null}
                  </li>
                ))}
              </ul>
            )}
          </Panel>

          {canEdit ? (
            <Panel>
              <div className="eyebrow">draw a region</div>
              {!drawing ? (
                <Button className="mt-2" size="small" onClick={startDrawing}>
                  Start new region
                </Button>
              ) : (
                <div className="mt-2">
                  <p className="muted mb-2">
                    Click the frame to add a point, or enter coordinates below. At least 3 points
                    are needed. {formatCount(draftPoints.length)} so far.
                  </p>
                  <div className="flex items-end gap-2">
                    <div className="flex-1">
                      <Label htmlFor="draft-x" className="mt-0">
                        X
                      </Label>
                      <Input
                        id="draft-x"
                        type="number"
                        min={0}
                        max={data.frame_width}
                        value={draftPointX}
                        onChange={(event) => setDraftPointX(event.target.value)}
                      />
                    </div>
                    <div className="flex-1">
                      <Label htmlFor="draft-y" className="mt-0">
                        Y
                      </Label>
                      <Input
                        id="draft-y"
                        type="number"
                        min={0}
                        max={data.frame_height}
                        value={draftPointY}
                        onChange={(event) => setDraftPointY(event.target.value)}
                      />
                    </div>
                    <Button
                      variant="quiet"
                      size="small"
                      onClick={() => {
                        const x = Number(draftPointX)
                        const y = Number(draftPointY)
                        if (!Number.isFinite(x) || !Number.isFinite(y)) return
                        addPoint([x, y])
                        setDraftPointX('')
                        setDraftPointY('')
                      }}
                    >
                      Add point
                    </Button>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button
                      variant="quiet"
                      size="small"
                      disabled={draftPoints.length === 0}
                      onClick={() => setDraftPoints((current) => current.slice(0, -1))}
                    >
                      Undo last point
                    </Button>
                    <Button variant="quiet" size="small" onClick={cancelDrawing}>
                      Cancel
                    </Button>
                  </div>

                  {draftPoints.length >= 3 ? (
                    <div className="mt-3">
                      <Label htmlFor="draft-name" className="mt-0">
                        Region name
                      </Label>
                      <Input
                        id="draft-name"
                        value={draftName}
                        onChange={(event) => setDraftName(event.target.value)}
                        placeholder="e.g. toilet"
                      />
                      <Button
                        className="mt-2"
                        size="small"
                        disabled={draftName.trim() === ''}
                        onClick={commitRegion}
                      >
                        Add region
                      </Button>
                    </div>
                  ) : null}
                </div>
              )}
            </Panel>
          ) : null}

          {canEdit ? (
            <Panel>
              <div className="eyebrow">check &amp; save</div>
              <p className="muted mt-2">
                Checking and saving both run through the recorder's own polygon rules — nothing
                here re-derives or second-guesses them.
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                <Button
                  variant="quiet"
                  size="small"
                  disabled={editedRegions.length === 0 || submitting !== null}
                  onClick={() => void handleValidate()}
                >
                  {submitting === 'validate' ? 'Checking…' : 'Check'}
                </Button>
                <Button
                  size="small"
                  disabled={editedRegions.length === 0 || submitting !== null}
                  onClick={() => void handleSave()}
                >
                  {submitting === 'save' ? 'Saving…' : 'Save'}
                </Button>
              </div>
              {editedRegions.length === 0 ? (
                <p className="muted mt-2 mb-0">
                  At least one region is required — this recorder does not support saving an empty
                  mask set.
                </p>
              ) : null}
              {submitError ? (
                <p className="err mt-2 mb-0" role="alert">
                  {submitError}
                </p>
              ) : null}
              {result ? <ValidationResult kind={result.kind} body={result.body} /> : null}
            </Panel>
          ) : null}
        </div>
      </div>
    </>
  )
}

function ValidationResult({ kind, body }: { kind: 'validate' | 'save'; body: RecorderMaskValidation }) {
  const failing = body.regions.filter((region) => !region.valid)
  return (
    <Notice tone={body.valid ? 'inert' : 'breach'} className="mt-3">
      <strong className="block font-mono text-[12.5px] tracking-[0.1em] uppercase">
        {kind === 'save'
          ? body.valid
            ? 'Saved'
            : 'Not saved'
          : body.valid
            ? 'Valid'
            : 'Not valid'}
      </strong>
      {body.error ? <span className="mt-1.5 block">{body.error}</span> : null}
      {!body.valid && failing.length > 0 ? (
        <ul className="mt-2 mb-0 list-disc pl-4">
          {failing.map((region) => (
            <li key={region.region_id}>
              {region.region_id}: {region.error ?? 'invalid'}
            </li>
          ))}
        </ul>
      ) : null}
      {body.valid && body.inventory ? (
        <span className="mt-1.5 block">
          {formatCount(body.inventory.regions.length)} region
          {body.inventory.regions.length === 1 ? '' : 's'} ·{' '}
          {formatMaskPercent(body.inventory.total_pct)} of the frame masked
        </span>
      ) : null}
    </Notice>
  )
}

function StopCameraRemedy({
  cameraId,
  cameraAction,
  onStopped,
}: {
  cameraId: string
  cameraAction: ReturnType<typeof useRecorderCameraAction>
  onStopped: () => void
}) {
  const isThisCamera = cameraAction.variables?.cameraId === cameraId
  const pending = isThisCamera && cameraAction.isPending
  return (
    <div className="mt-2.5 flex flex-wrap items-center gap-3">
      <Button
        variant="live"
        size="small"
        disabled={cameraAction.isPending}
        onClick={() =>
          cameraAction.mutate({ cameraId, action: 'stop' }, { onSuccess: onStopped })
        }
      >
        {pending ? 'Stopping…' : 'Stop this camera'}
      </Button>
      {isThisCamera && cameraAction.isSuccess ? (
        <p className="ok-text m-0">Sent. The mask editor will refresh.</p>
      ) : null}
      {isThisCamera && cameraAction.isError ? (
        <p className="err m-0" role="alert">
          {cameraAction.error.message}
        </p>
      ) : null}
    </div>
  )
}

function MaskCanvas({
  frameWidth,
  frameHeight,
  imageUrl,
  regions,
  draftPoints,
  interactive,
  onAddPoint,
}: {
  frameWidth: number
  frameHeight: number
  imageUrl?: string
  regions: RecorderMaskRegion[]
  draftPoints: [number, number][]
  interactive: boolean
  onAddPoint: (point: [number, number]) => void
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [imageFailed, setImageFailed] = useState(false)

  useEffect(() => {
    setImageFailed(false)
  }, [imageUrl])

  function handleClick(event: ReactMouseEvent<SVGSVGElement>) {
    if (!interactive || !svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    const framePoint = displayPointToFramePoint(
      [event.clientX - rect.left, event.clientY - rect.top],
      { width: rect.width, height: rect.height },
      { frameWidth, frameHeight },
    )
    onAddPoint(framePoint)
  }

  const showImage = imageUrl !== undefined && !imageFailed

  return (
    <div
      className="relative w-full overflow-hidden rounded-sm border border-line-2 bg-void"
      style={{ aspectRatio: `${frameWidth} / ${frameHeight}` }}
    >
      {showImage ? (
        <img
          src={imageUrl}
          alt="Calibration frame for drawing privacy masks"
          className="absolute inset-0 h-full w-full object-contain"
          onError={() => setImageFailed(true)}
        />
      ) : (
        <div
          className="absolute inset-0 [background:repeating-linear-gradient(45deg,#0a0e11_0,#0a0e11_6px,#0d1216_6px,#0d1216_12px)]"
          aria-hidden="true"
        />
      )}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${frameWidth} ${frameHeight}`}
        preserveAspectRatio="xMidYMid meet"
        className={cn('absolute inset-0 h-full w-full', interactive && 'cursor-crosshair')}
        onClick={handleClick}
        role="img"
        aria-label={`${formatCount(regions.length)} mask region${regions.length === 1 ? '' : 's'} over a ${frameWidth} by ${frameHeight} pixel frame`}
      >
        {regions.map((region, index) => (
          <polygon
            key={`${region.region_id}-${index}`}
            points={polygonPointsAttr(region.polygon)}
            fill="var(--color-signal-d)"
            fillOpacity={0.35}
            stroke="var(--color-signal-d)"
            strokeWidth={3}
          />
        ))}
        {draftPoints.length > 0 ? (
          <polyline
            points={polygonPointsAttr(draftPoints)}
            fill="none"
            stroke="var(--color-caution)"
            strokeWidth={3}
          />
        ) : null}
        {draftPoints.map(([x, y], index) => (
          <circle key={index} cx={x} cy={y} r={6} fill="var(--color-caution)" />
        ))}
      </svg>
    </div>
  )
}
