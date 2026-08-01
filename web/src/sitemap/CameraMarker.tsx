import type { KeyboardEvent, PointerEvent } from 'react'
import { Link } from 'react-router-dom'
import { cn } from '@/lib/cn'
import type { Tone } from '@/lib/severity'
import { humanizeEnum } from '@/lib/format'
import {
  nudgeDirectionForKey,
  pointFromClientPosition,
  NUDGE_STEP,
  NUDGE_STEP_LARGE,
  type NormalizedPoint,
  type NudgeDirection,
  type StageRect,
} from '@/sitemap/placement'

const dotTone: Record<Tone, string> = {
  nominal: 'border-signal-d bg-signal',
  caution: 'border-[#574000] bg-caution',
  breach: 'border-[#5c231b] bg-breach',
  inert: 'border-line-2 bg-panel-2',
}

const toneLabel: Record<Tone, string> = {
  nominal: 'nominal',
  caution: 'caution',
  breach: 'breach',
  inert: 'no events yet',
}

export interface CameraMarkerProps {
  cameraId: string
  zone: string | null
  point: NormalizedPoint
  tone: Tone
  /** True when this camera has never been explicitly placed — `point` is the schematic default, not something the operator dragged. */
  isDefaultPosition: boolean
  /**
   * Whether this render is a "just changed" one, so the marker can announce
   * the transition. Purely presentational — driven by remounting this
   * component under a `tone`-suffixed `key` in `FloorplanStage`, so the CSS
   * rise-in animation replays exactly on a tone change and never on an
   * unrelated re-render (data refetch, drag of a different marker, etc).
   * `prefers-reduced-motion` is honoured globally (see `index.css`).
   */
  editable: boolean
  onNudge: (direction: NudgeDirection, step: number) => void
  onDragTo: (point: NormalizedPoint) => void
  getStageRect: () => StageRect | null
}

/**
 * One camera's marker on the site map. In `editable` mode it is a `button`
 * that can be dragged (pointer events, captured on the marker itself so a
 * fast drag past its edge does not lose the pointer) or nudged with the
 * arrow keys — drag-only would leave keyboard-only operators unable to place
 * a camera at all. Outside edit mode it is a plain link to that camera's own
 * page, exactly like every other camera reference in this console.
 */
export function CameraMarker({
  cameraId,
  zone,
  point,
  tone,
  isDefaultPosition,
  editable,
  onNudge,
  onDragTo,
  getStageRect,
}: CameraMarkerProps) {
  const style = {
    left: `${point.x * 100}%`,
    top: `${point.y * 100}%`,
  }

  const zoneWords = zone ? humanizeEnum(zone) : 'ungrouped'
  const label = `${cameraId}, ${zoneWords}, ${toneLabel[tone]}${
    isDefaultPosition ? ', not yet placed' : ''
  }`

  function handlePointerDown(event: PointerEvent<HTMLButtonElement>) {
    if (!editable) return
    // Not implemented in every test environment (jsdom has no pointer
    // capture support) and genuinely optional here: capturing just keeps a
    // fast drag past the marker's own edge from losing the pointer, it is
    // not required for a drag to work at all.
    event.currentTarget.setPointerCapture?.(event.pointerId)
  }

  function handlePointerMove(event: PointerEvent<HTMLButtonElement>) {
    if (!editable || event.buttons === 0) return
    const rect = getStageRect()
    if (!rect) return
    onDragTo(pointFromClientPosition(event.clientX, event.clientY, rect))
  }

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (!editable) return
    const direction = nudgeDirectionForKey(event.key)
    if (!direction) return
    event.preventDefault()
    onNudge(direction, event.shiftKey ? NUDGE_STEP_LARGE : NUDGE_STEP)
  }

  const dotClass = cn(
    'block h-4 w-4 rounded-full border-2 animate-rise',
    dotTone[tone],
    isDefaultPosition && 'border-dashed',
  )

  const wrapperClass = cn(
    'group absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center gap-1',
    'focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal focus-visible:outline-offset-2',
    editable ? 'cursor-grab touch-none' : 'cursor-pointer',
  )

  const labelClass =
    'readout rounded-sm bg-void/80 px-1 py-px text-[10px] text-dim group-hover:text-fg group-focus-visible:text-fg'

  if (editable) {
    return (
      <button
        type="button"
        style={style}
        className={wrapperClass}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onKeyDown={handleKeyDown}
        aria-label={`${label}. Use arrow keys to move, hold shift to move further.`}
        data-camera-id={cameraId}
        data-testid={`marker-${cameraId}`}
      >
        <span className={dotClass} />
        <span className={labelClass}>{cameraId}</span>
      </button>
    )
  }

  return (
    <Link
      to={`/cameras/${encodeURIComponent(cameraId)}`}
      style={style}
      className={wrapperClass}
      aria-label={label}
      data-camera-id={cameraId}
      data-testid={`marker-${cameraId}`}
    >
      <span className={dotClass} />
      <span className={labelClass}>{cameraId}</span>
    </Link>
  )
}
