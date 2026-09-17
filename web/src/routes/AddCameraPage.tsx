import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Capability, CameraCreateRequest } from '@/api/engineClient'
import { EngineHttpError } from '@/api/engineClient'
import { useCameras, useCreateCamera, useProbeSource } from '@/api/queries'
import { Notice } from '@/components/ui/Notice'
import { Panel } from '@/components/ui/Panel'
import { CAPABILITY_COST, CAPABILITY_LABELS, CAPABILITY_ORDER } from '@/lib/alerts'
import { ZONES } from '@/lib/cameraEdit'
import { humanizeEnum } from '@/lib/format'

/**
 * Stand up a camera without editing a file or restarting the engine.
 *
 * The order of the form is the order of the decisions, deliberately: **the stream first,
 * and looked at, before anything else is asked.** A camera added with a wrong URL fails
 * quietly — it appears in the list, its runner retries forever behind exponential
 * backoff, and the only symptom is a `frames_seen` that never moves. Every other field
 * is cheap to change afterwards; this is the one worth proving first, and the only proof
 * that means anything is a picture.
 *
 * So the rest of the form is unreachable until a frame has been decoded, and what the
 * operator sees is that frame.
 */
export function AddCameraPage() {
  const navigate = useNavigate()
  const cameras = useCameras()
  const probe = useProbeSource()
  const create = useCreateCamera()

  const [url, setUrl] = useState('')
  const [cameraId, setCameraId] = useState('')
  const [label, setLabel] = useState('')
  const [zone, setZone] = useState('')
  const [capabilities, setCapabilities] = useState<Capability[]>([
    'anomaly_detection',
    'camera_tamper',
  ])

  const preview = probe.data
  const playable = preview?.ok === true
  const writable = cameras.data?.config_writable ?? false
  const taken = new Set((cameras.data?.cameras ?? []).map((camera) => camera.camera_id))
  const idClash = cameraId.trim().length > 0 && taken.has(cameraId.trim())

  function toggle(capability: Capability) {
    setCapabilities((previous) =>
      previous.includes(capability)
        ? previous.filter((item) => item !== capability)
        : [...previous, capability],
    )
  }

  function onProbe() {
    const trimmed = url.trim()
    if (!trimmed) return
    probe.mutate(trimmed, {
      onSuccess: (result) => {
        // The camera's own name for itself beats anything a person would type, and an
        // EarthCam page supplies one. A default only — still editable below.
        if (result.ok && result.title && !label) setLabel(result.title)
      },
    })
  }

  function onCreate() {
    create.mutate(
      {
        camera_id: cameraId.trim(),
        url: url.trim(),
        label: label.trim(),
        zone: zone === '' ? null : (zone as CameraCreateRequest['zone']),
        capabilities,
      },
      { onSuccess: (record) => navigate(`/cameras/${encodeURIComponent(record.camera_id)}`) },
    )
  }

  const refusal =
    create.error instanceof EngineHttpError ? create.error.detail : create.error?.message

  return (
    <section>
      <h1>Add a camera</h1>
      <p className="lede">
        The engine starts watching as soon as this is saved — no restart. The record is
        written to <code>cameras.json</code>, so it survives one.
      </p>

      {!writable ? (
        <Notice tone="breach" className="mb-4">
          Camera writes are disabled on this engine, so this form cannot save. Set{' '}
          <code className="font-mono text-[11.5px]">SENTINEL_ENABLE_CAMERA_WRITES=true</code>{' '}
          only where everything that can reach the port is permitted to reconfigure
          cameras — there is no authentication in front of it.
        </Notice>
      ) : null}

      <Panel>
        <div className="eyebrow mb-1">1 · The stream</div>
        <p className="muted mt-0 mb-3 max-w-[70ch]">
          An <code>rtsp://</code> URL, an <strong className="text-fg">earthcam.com page</strong>{' '}
          (the page itself, not a playlist), or a path to a file. Look at it before
          saving: a wrong URL produces a camera that looks present and never delivers a
          frame.
        </p>

        <div className="flex flex-wrap gap-2">
          <input
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="rtsp://… or https://www.earthcam.com/…"
            aria-label="Stream URL"
            className="min-w-[280px] flex-1 rounded-sm border border-line-2 bg-panel-2 px-[9px] py-[6px] font-mono text-[12px] text-fg"
          />
          <button
            type="button"
            onClick={onProbe}
            disabled={!url.trim() || probe.isPending}
            className="rounded-sm border border-signal-d bg-[#0f1610] px-[13px] py-[6px] font-mono text-[11.5px] uppercase tracking-[0.08em] text-signal disabled:opacity-40"
          >
            {probe.isPending ? 'Looking…' : 'Preview'}
          </button>
        </div>

        {probe.isError ? (
          <Notice tone="breach" className="mt-3">
            {probe.error.message}
          </Notice>
        ) : null}

        {preview ? (
          <div className="mt-3">
            {preview.ok ? (
              <div className="flex flex-wrap items-start gap-3">
                {preview.thumbnail ? (
                  <img
                    src={preview.thumbnail}
                    alt="One frame decoded from this stream"
                    className="w-[280px] rounded-sm border border-line"
                  />
                ) : null}
                <dl className="readout m-0 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[12px]">
                  <dt className="text-dimmer">Source</dt>
                  <dd className="m-0 text-fg">{preview.source_kind}</dd>
                  {preview.title ? (
                    <>
                      <dt className="text-dimmer">Name</dt>
                      <dd className="m-0 text-fg">{preview.title}</dd>
                    </>
                  ) : null}
                  <dt className="text-dimmer">Size</dt>
                  <dd className="m-0 text-fg">
                    {preview.width}×{preview.height}
                  </dd>
                  <dt className="text-dimmer">Codec</dt>
                  <dd className="m-0 text-fg">{preview.codec}</dd>
                  <dt className="text-dimmer">Rate</dt>
                  <dd className="m-0 text-fg">{preview.fps.toFixed(0)} fps</dd>
                </dl>
              </div>
            ) : (
              <Notice tone="caution">This URL did not play: {preview.detail}</Notice>
            )}
          </div>
        ) : null}
      </Panel>

      {/* Gated on a decoded frame, not on a filled-in field. Everything below is cheap
          to change later; the stream is the one thing worth proving first. */}
      {playable ? (
        <>
          <Panel className="mt-4">
            <div className="eyebrow mb-1">2 · Name and group</div>
            <p className="muted mt-0 mb-3 max-w-[70ch]">
              The id is how every event, alert and clip refers to this camera, and it
              cannot be changed afterwards. The group is how the console arranges the
              wall — it changes nothing about how the engine watches.
            </p>

            <div className="grid max-w-[560px] gap-3">
              <label className="grid gap-1">
                <span className="text-[13px] text-fg">Camera id</span>
                <input
                  value={cameraId}
                  onChange={(event) => setCameraId(event.target.value)}
                  placeholder="loading-bay"
                  className="rounded-sm border border-line-2 bg-panel-2 px-[9px] py-[6px] font-mono text-[12px] text-fg"
                />
                {idClash ? (
                  <span className="text-[11.5px] text-breach">
                    A camera with this id already exists.
                  </span>
                ) : null}
              </label>

              <label className="grid gap-1">
                <span className="text-[13px] text-fg">Label</span>
                <input
                  value={label}
                  onChange={(event) => setLabel(event.target.value)}
                  placeholder="Loading bay"
                  className="rounded-sm border border-line-2 bg-panel-2 px-[9px] py-[6px] text-[13px] text-fg"
                />
              </label>

              <label className="grid gap-1">
                <span className="text-[13px] text-fg">Group</span>
                <select
                  value={zone}
                  onChange={(event) => setZone(event.target.value)}
                  className="rounded-sm border border-line-2 bg-panel-2 px-[9px] py-[6px] text-[13px] text-fg"
                >
                  <option value="">Ungrouped</option>
                  {ZONES.map((value) => (
                    <option key={value} value={value}>
                      {humanizeEnum(value)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </Panel>

          <Panel className="mt-4">
            <div className="eyebrow mb-1">3 · What it watches for</div>
            <p className="muted mt-0 mb-3 max-w-[70ch]">
              A capability that is off loads no model and costs nothing. One whose model
              this engine did not load at startup is refused when you save, naming the
              restart — which models exist is decided when the process starts.
            </p>

            <ul className="m-0 flex list-none flex-col gap-px overflow-hidden rounded-sm border border-line p-0">
              {CAPABILITY_ORDER.map((capability) => (
                <li key={capability} className="bg-panel-2">
                  <label className="flex cursor-pointer items-center gap-3 px-[11px] py-[9px] hover:bg-[#161f26]">
                    <input
                      type="checkbox"
                      checked={capabilities.includes(capability)}
                      onChange={() => toggle(capability)}
                      className="size-[14px] shrink-0 accent-[#76b900]"
                    />
                    <span className="min-w-0 flex-1">
                      <span className="block text-[13px] text-fg">
                        {CAPABILITY_LABELS[capability]}
                      </span>
                      <span className="block font-mono text-[10.5px] text-dimmer">
                        {CAPABILITY_COST[capability]}
                      </span>
                    </span>
                  </label>
                </li>
              ))}
            </ul>
          </Panel>

          {create.isError ? (
            <Notice tone="breach" className="mt-4">
              {refusal}
            </Notice>
          ) : null}

          <div className="mt-4 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={onCreate}
              disabled={!writable || !cameraId.trim() || idClash || create.isPending}
              className="rounded-sm border border-signal-d bg-[#0f1610] px-[15px] py-[7px] font-mono text-[12px] uppercase tracking-[0.08em] text-signal disabled:opacity-40"
            >
              {create.isPending ? 'Adding…' : 'Add camera'}
            </button>
            <button
              type="button"
              onClick={() => navigate('/dashboard')}
              className="font-mono text-[11px] uppercase tracking-[0.07em] text-dim underline-offset-2 hover:text-fg hover:underline"
            >
              Cancel
            </button>
          </div>
        </>
      ) : null}
    </section>
  )
}
