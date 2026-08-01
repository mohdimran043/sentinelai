import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { Notice } from '@/components/ui/Notice'
import { KvList } from '@/components/ui/Kv'
import { useRecorderSettings } from '@/recorder/queries'
import type { RecorderSettingsResponse } from '@/recorder/recorder.types'

export function ConfigurePage() {
  const settings = useRecorderSettings()
  const data: RecorderSettingsResponse | undefined = settings.data

  return (
    <>
      <PageHeader
        eyebrow="inference policy"
        title="Configure"
        lede="Whether inference is allowed to run on this machine or offsite, and the recorder’s own warning about what that choice means. This page describes the recorder appliance, not the AI engine."
      />

      {settings.isError ? (
        <Notice tone="breach">
          {settings.error.message} Nothing below reflects the recorder’s actual configuration
          until it answers again.
        </Notice>
      ) : null}

      {settings.isPending ? <p className="muted">Loading the recorder’s settings…</p> : null}

      {data ? (
        <>
          <WiredNotice wired={data.wired} />

          <Notice tone="breach" className="mt-3.5">
            <strong className="block font-mono text-[12.5px] tracking-[0.1em] uppercase">
              Offsite inference warning
            </strong>
            <span className="mt-1.5 block">{data.online_warning}</span>
          </Notice>

          <Panel className="mt-3.5">
            <div className="eyebrow">recorded setting</div>
            <KvList
              rows={[
                {
                  key: 'mode',
                  label: 'Inference mode',
                  value:
                    data.settings.inference_mode === 'online' ? 'Offsite' : 'On this machine',
                },
                {
                  key: 'ack',
                  label: 'Offsite use acknowledged',
                  value: data.settings.online_acknowledged ? 'Yes' : 'No',
                },
                {
                  key: 'wired',
                  label: 'Drives anything today',
                  value: data.wired ? 'Yes' : 'No',
                },
              ]}
            />
            <p className="muted mt-3 mb-0">{data.note}</p>
          </Panel>
        </>
      ) : null}
    </>
  )
}

/**
 * The single most important fact on the page: whether this setting does
 * anything at all. `wired: false` means it does not, and that must be said as
 * plainly as ModelsPage says "nothing is reading the footage" — not folded
 * into a key/value row where it reads as just another field.
 */
function WiredNotice({ wired }: { wired: boolean }) {
  if (!wired) {
    return (
      <Notice tone="breach">
        <strong className="block font-mono text-[12.5px] tracking-[0.1em] uppercase">
          This setting decides nothing today
        </strong>
        <span className="mt-1.5 block">
          <code className="readout">wired: false</code> — no component reads{' '}
          <code className="readout">inference_mode</code> yet. Choosing offsite here does not
          send a single frame anywhere, and choosing local does not stop one. The choice is
          recorded so it is explicit and auditable for when something is wired to it.
        </span>
      </Notice>
    )
  }

  return (
    <Notice tone="inert">
      <strong className="block font-mono text-[12.5px] tracking-[0.1em] uppercase">
        This setting is wired
      </strong>
      <span className="mt-1.5 block">
        The recorder acts on this value: it decides where inference actually runs.
      </span>
    </Notice>
  )
}
