import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { Reading } from '@/components/ui/Reading'
import { Notice } from '@/components/ui/Notice'
import { Pill } from '@/components/ui/Pill'
import { KvList } from '@/components/ui/Kv'
import { formatBytes, formatCount, formatUsd, humanizeEnum } from '@/lib/format'
import { useRecorderModels } from '@/recorder/queries'
import { toneForTierStatus } from '@/recorder/tone'
import type {
  RecorderCatalogEntry,
  RecorderModelsResponse,
  RecorderModelStatus,
} from '@/recorder/recorder.types'

/** `off` -> "Only when asked". The wire tokens are never rendered raw as prose. */
function describeCadence(cadence: string): string {
  switch (cadence) {
    case 'off':
      return 'Only when asked'
    case 'interval':
      return 'On a timer'
    case 'event':
      return 'When something happens'
    case 'both':
      return 'On a timer, and on an event'
    default:
      return humanizeEnum(cadence)
  }
}

export function ModelsPage() {
  const models = useRecorderModels()
  const data: RecorderModelsResponse | undefined = models.data

  return (
    <>
      <PageHeader
        eyebrow="perception"
        title="Models"
        lede="Which model reads the footage, where it runs, and what it costs. Everything on this page describes the recorder appliance, not the AI engine."
      />

      {models.isError ? (
        <Notice tone="breach">
          {models.error.message} This page cannot tell you whether a model is running, so do not
          read the absence of a warning below as an all-clear.
        </Notice>
      ) : null}

      {models.isPending ? <p className="muted">Loading the model configuration…</p> : null}

      {data ? (
        <>
          <StatusBlock status={data.status} />

          {data.config_error ? (
            <Notice tone="breach" className="mt-3.5">
              The saved model is not usable, so nothing is being described: {data.config_error}
            </Notice>
          ) : null}

          <WhereItRuns data={data} />

          <LocalStack data={data} />
          <FrontierStack data={data} />
          <PerceptionTiers data={data} />
        </>
      ) : null}
    </>
  )
}

/**
 * Where inference would run. The whole block is skipped rather than guessed at
 * if the recorder does not report it — see the note on `perception` in
 * recorder.types.ts.
 */
function WhereItRuns({ data }: { data: RecorderModelsResponse }) {
  const { perception, selected } = data
  if (!perception && !selected) return null
  return (
    <Panel className="mt-3.5">
      <div className="eyebrow">where inference runs</div>
      <KvList
        rows={[
          ...(selected
            ? [
                {
                  key: 'mode',
                  label: 'Selected mode',
                  value: selected.mode === 'online' ? 'Offsite' : 'On this machine',
                },
              ]
            : []),
          ...(perception
            ? [
                {
                  key: 'remote',
                  label: 'Runs in a separate process',
                  value: perception.remote ? 'Yes' : 'No',
                },
                {
                  key: 'reachable',
                  label: 'Runtime reachable',
                  value: perception.reachable ? 'Yes' : 'No',
                },
                {
                  key: 'offsite',
                  label: 'Frames leave this machine',
                  value: perception.offsite ? 'Yes' : 'No',
                },
              ]
            : []),
        ]}
      />
      {perception ? <p className="muted mt-3 mb-0">{perception.note}</p> : null}
    </Panel>
  )
}

/**
 * The single most important thing on the page: whether anything is reading the
 * footage at all. It goes first, at full width, and it is never softened to
 * protect the layout.
 */
function StatusBlock({ status }: { status: RecorderModelStatus }) {
  const cadenceInUse = status.cadence === 'interval' || status.cadence === 'both'

  return (
    <section aria-labelledby="model-status-heading">
      <h2 id="model-status-heading" className="sr-only">
        Description status
      </h2>

      {status.ready ? (
        <Notice tone="inert">
          A model is configured and footage is being described. Descriptions are of a single still
          frame; they assert nothing about welfare, and no alert is derived from them.
        </Notice>
      ) : (
        <Notice tone="breach">
          <strong className="block font-mono text-[12.5px] tracking-[0.1em] uppercase">
            Nothing is reading the footage
          </strong>
          <span className="mt-1.5 block">{status.detail}</span>
          <span className="mt-1.5 block text-[11.5px] text-dim">
            Reported reason: <code className="readout">{status.reason}</code> —{' '}
            {humanizeEnum(status.reason).toLowerCase()}. Recording continues; only description is
            off.
          </span>
        </Notice>
      )}

      <div className="mt-3.5 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Reading label="description" value={status.ready ? 'Running' : 'Off'} alarm={!status.ready} />
        <Reading label="when to describe" value={describeCadence(status.cadence)} />
        <Reading
          label="interval"
          value={formatCount(status.interval_sec)}
          unit={cadenceInUse ? 's' : 's · not in use'}
        />
        <Reading
          label="frames leave this machine"
          value={status.offsite ? 'Yes' : 'No'}
          alarm={status.offsite}
        />
      </div>

      {!status.vision ? (
        <Notice tone="caution" className="mt-3.5">
          The configured model cannot accept images. A text-only model asked to describe a camera
          answers with a plausible scene it has never seen, and that text would be filed against a
          real person — so descriptions are refused rather than generated.
        </Notice>
      ) : null}
    </section>
  )
}

function LocalStack({ data }: { data: RecorderModelsResponse }) {
  const catalog = data.catalog ?? []
  const catalogFor = (name: string): RecorderCatalogEntry | undefined =>
    catalog.find((entry) => entry.local_id === name)

  return (
    <Panel className="mt-3.5">
      <div className="eyebrow">local model stack</div>
      <p className="muted mt-2">
        Installed on the recorder itself. Frames handled by one of these never leave the machine.
      </p>

      {data.local_error ? (
        <Notice tone="breach" className="mt-3">
          No local model runtime is reachable: {data.local_error}
        </Notice>
      ) : null}

      {data.local.length === 0 ? (
        <p className="muted mt-3 mb-0">The recorder reports no local model installed.</p>
      ) : (
        <div className="table-scroll mt-3">
          <table className="data-table">
            <caption className="sr-only">Local models installed on the recorder</caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Download</th>
                <th scope="col">Images</th>
                <th scope="col">Family</th>
                <th scope="col">In this build&rsquo;s catalog</th>
              </tr>
            </thead>
            <tbody>
              {data.local.map((model) => {
                const entry = catalogFor(model.name)
                return (
                  <tr key={model.name}>
                    <td className="readout">{model.name}</td>
                    <td className="readout whitespace-nowrap">{formatBytes(model.size_bytes)}</td>
                    <td>
                      {model.vision ? (
                        <Pill tone="nominal">sees images</Pill>
                      ) : (
                        <Pill tone="breach">text only</Pill>
                      )}
                    </td>
                    <td className="readout">{model.family}</td>
                    <td>
                      {entry?.declared ? (
                        <Pill tone="nominal">declared</Pill>
                      ) : (
                        <Pill tone="inert">not declared</Pill>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {catalog.length > 0 ? (
        <div className="mt-4">
          <h3 className="mb-1.5 text-[13px]">What the recorder has measured</h3>
          {data.catalog_note ? <p className="muted mb-2">{data.catalog_note}</p> : null}
          {catalog.map((entry) => (
            <details
              key={entry.local_id}
              className="mt-2 rounded-sm border border-line bg-panel-2 px-3 py-2"
            >
              <summary className="cursor-pointer text-[13px] text-fg">
                <span className="readout">{entry.local_id}</span>
                {entry.parameters ? (
                  <span className="muted ml-2">{entry.parameters} parameters</span>
                ) : null}
              </summary>
              <p className="muted mt-2 whitespace-pre-line">{entry.runtime_note}</p>
              {entry.online_equivalent ? (
                <>
                  <p className="muted mt-2">
                    <span className="eyebrow mr-2">offsite counterpart</span>
                    <span className="readout">{entry.online_equivalent}</span> —{' '}
                    {entry.why_this_pairing}
                  </p>
                  <p className="muted mt-2">{entry.not_equivalent_because}</p>
                </>
              ) : (
                <p className="muted mt-2">{entry.not_equivalent_because}</p>
              )}
            </details>
          ))}
        </div>
      ) : null}
    </Panel>
  )
}

function FrontierStack({ data }: { data: RecorderModelsResponse }) {
  return (
    <Panel className="mt-3.5">
      <div className="eyebrow">offsite models</div>
      <p className="muted mt-2">
        Priced per million tokens. Every frame sent to one of these leaves the machine.
      </p>

      {data.frontier_credentialed ? (
        <Notice tone="caution" className="mt-3">
          An API key is present, so the offsite path can run. Selecting it sends images of the
          people in front of these cameras to a third party.
        </Notice>
      ) : (
        <Notice tone="breach" className="mt-3">
          No API key is present in the recorder&rsquo;s environment, so none of these can be used
          today. The prices below are what they would cost, not what is being spent.
        </Notice>
      )}

      <div className="table-scroll mt-3">
        <table className="data-table">
          <caption className="sr-only">Offsite models and their per-million-token prices</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Identifier</th>
              <th scope="col">Input / MTok</th>
              <th scope="col">Output / MTok</th>
              <th scope="col">Note</th>
            </tr>
          </thead>
          <tbody>
            {data.frontier.map((model) => (
              <tr key={model.id}>
                <td>
                  {model.label}
                  {data.frontier_credentialed ? null : (
                    <Pill tone="inert" className="ml-2">
                      no key
                    </Pill>
                  )}
                </td>
                <td className="readout">{model.id}</td>
                <td className="readout whitespace-nowrap">{formatUsd(model.input_per_mtok)}</td>
                <td className="readout whitespace-nowrap">{formatUsd(model.output_per_mtok)}</td>
                <td className="muted">{model.note || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="muted mt-3 mb-0">{data.frontier_note}</p>
    </Panel>
  )
}

function PerceptionTiers({ data }: { data: RecorderModelsResponse }) {
  const tiers = data.tiers ?? []
  if (tiers.length === 0) return null
  return (
    <Panel className="mt-3.5">
      <div className="eyebrow">perception tiers</div>
      <p className="muted mt-2">
        What the design specifies for each tier, and whether it exists. A tier that is not built
        detects nothing — its absence is not evidence that nothing happened.
      </p>
      <div className="table-scroll mt-3">
        <table className="data-table">
          <caption className="sr-only">Perception tiers and their build status</caption>
          <thead>
            <tr>
              <th scope="col">Tier</th>
              <th scope="col">Specified</th>
              <th scope="col">Status</th>
              <th scope="col">What that means</th>
            </tr>
          </thead>
          <tbody>
            {tiers.map((tier) => (
              <tr key={tier.tier}>
                <td className="readout">{tier.tier}</td>
                <td className="readout">{tier.model}</td>
                <td>
                  <Pill tone={toneForTierStatus(tier.status)}>{tier.status}</Pill>
                </td>
                <td className="muted">{tier.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
