import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { Reading } from '@/components/ui/Reading'
import { Notice } from '@/components/ui/Notice'
import { Pill } from '@/components/ui/Pill'
import { formatBytes, formatCount, formatNanosDateTime } from '@/lib/format'
import { useRecorderUntrackedStorage } from '@/recorder/queries'
import type {
  RecorderUntrackedCamera,
  RecorderUntrackedResponse,
} from '@/recorder/recorder.types'

export function StoragePage() {
  const storage = useRecorderUntrackedStorage()
  const data: RecorderUntrackedResponse | undefined = storage.data
  const camerasWithFootage = data?.cameras.filter((camera) => camera.count > 0) ?? []
  const camerasWithSample = data?.cameras.filter((camera) => (camera.sample?.length ?? 0) > 0) ?? []

  return (
    <>
      <PageHeader
        eyebrow="disk"
        title="Storage"
        lede="Footage on disk that the segment index has no record of, per camera. None of it is deleted automatically — reclaiming the space is a deliberate act, not a background job."
      />

      {storage.isError ? (
        <Notice tone="breach">
          {storage.error.message} Nothing below reflects what is actually on disk until the
          recorder answers again.
        </Notice>
      ) : null}

      {storage.isPending ? <p className="muted">Loading untracked storage…</p> : null}

      {data ? (
        <>
          <Notice tone="inert">{data.note}</Notice>

          <div className="mt-3.5 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Reading label="untracked, total" value={formatBytes(data.total_bytes)} />
            <Reading label="untracked files" value={formatCount(data.total_count)} />
            <Reading
              label="cameras with untracked footage"
              value={formatCount(camerasWithFootage.length)}
              alarm={camerasWithFootage.length > 0}
            />
            <Reading label="cameras reporting" value={formatCount(data.cameras.length)} />
          </div>

          <Panel className="mt-3.5 p-0">
            <div className="table-scroll">
              <table className="data-table">
                <caption className="sr-only">Untracked footage per camera</caption>
                <thead>
                  <tr>
                    <th scope="col">Camera</th>
                    <th scope="col">Directory</th>
                    <th scope="col">Files</th>
                    <th scope="col">Bytes</th>
                    <th scope="col">Sample</th>
                  </tr>
                </thead>
                <tbody>
                  {data.cameras.map((camera) => (
                    <tr key={camera.camera_id}>
                      <td className="readout whitespace-nowrap">{camera.camera_id}</td>
                      <td className="readout break-all">{camera.dir}</td>
                      <td className="readout">{formatCount(camera.count)}</td>
                      <td className="readout whitespace-nowrap">
                        {formatBytes(camera.total_bytes)}
                      </td>
                      <td>
                        {camera.count === 0 ? (
                          <Pill tone="inert">none</Pill>
                        ) : camera.truncated ? (
                          <Pill tone="caution">partial — more not shown</Pill>
                        ) : (
                          <Pill tone="inert">
                            complete · {formatCount(camera.sample?.length ?? 0)}
                          </Pill>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          {camerasWithSample.length > 0 ? <SampleFiles cameras={camerasWithSample} /> : null}
        </>
      ) : null}
    </>
  )
}

function SampleFiles({ cameras }: { cameras: RecorderUntrackedCamera[] }) {
  return (
    <Panel className="mt-3.5">
      <div className="eyebrow">sample files</div>
      <p className="muted mt-2">
        A partial listing per camera — not an inventory of everything on disk.
      </p>
      {cameras.map((camera) => (
        <details
          key={camera.camera_id}
          className="mt-2 rounded-sm border border-line bg-panel-2 px-3 py-2"
        >
          <summary className="cursor-pointer text-[13px] text-fg">
            <span className="readout">{camera.camera_id}</span>
            <span className="muted ml-2">
              {formatCount(camera.sample?.length ?? 0)} of {formatCount(camera.count)} shown
              {camera.truncated ? ' · more exist' : ''}
            </span>
          </summary>
          <ul className="m-0 mt-2 list-none space-y-1 p-0">
            {(camera.sample ?? []).map((file) => (
              <li key={file.path} className="flex flex-wrap justify-between gap-2 text-[12px]">
                <span className="readout break-all">{file.path}</span>
                <span className="readout whitespace-nowrap text-dim">
                  {formatBytes(file.size_bytes)} · {formatNanosDateTime(file.modified_ns)}
                </span>
              </li>
            ))}
          </ul>
        </details>
      ))}
    </Panel>
  )
}
