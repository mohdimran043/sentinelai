import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { Reading } from '@/components/ui/Reading'
import { Notice } from '@/components/ui/Notice'
import { Pill } from '@/components/ui/Pill'
import { formatCount } from '@/lib/format'
import { useRecorderCapabilities } from '@/recorder/queries'
import type { RecorderCapabilitiesResponse } from '@/recorder/recorder.types'

export function CapabilitiesPage() {
  const capabilities = useRecorderCapabilities()
  const data: RecorderCapabilitiesResponse | undefined = capabilities.data
  const available = data?.capabilities.filter((capability) => capability.available) ?? []
  const unavailable = data?.capabilities.filter((capability) => !capability.available) ?? []

  return (
    <>
      <PageHeader
        eyebrow="capability matrix"
        title="Capabilities"
        lede="What this recorder can and cannot do today, in its own words. A capability the recorder cannot deliver is rendered as unavailable, not hidden — the split between the two is the information this page exists to show."
      />

      {capabilities.isError ? (
        <Notice tone="breach">
          {capabilities.error.message} Nothing below reflects what the recorder can actually do
          until it answers again.
        </Notice>
      ) : null}

      {capabilities.isPending ? <p className="muted">Loading the capability matrix…</p> : null}

      {data ? (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Reading label="declared" value={formatCount(data.capabilities.length)} />
            <Reading label="available" value={formatCount(available.length)} />
            <Reading
              label="not available"
              value={formatCount(unavailable.length)}
              alarm={unavailable.length > 0}
            />
          </div>

          <div className="mt-3.5 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {data.capabilities.map((capability) => (
              <Panel
                key={capability.key}
                className={
                  capability.available ? undefined : 'border-l-2 border-l-breach'
                }
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="eyebrow">{capability.key}</div>
                    <h3 className="mt-1 mb-0 text-[14px]">{capability.label}</h3>
                  </div>
                  <Pill tone={capability.available ? 'nominal' : 'breach'}>
                    {capability.available ? 'available' : 'not available'}
                  </Pill>
                </div>
                {capability.reason ? (
                  <p className="muted mt-2 mb-0">{capability.reason}</p>
                ) : null}
              </Panel>
            ))}
          </div>
        </>
      ) : null}
    </>
  )
}
