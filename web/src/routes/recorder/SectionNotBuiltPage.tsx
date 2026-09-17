import { Navigate, useParams } from 'react-router-dom'
import { PageHeader } from '@/components/ui/PageHeader'
import { Absent } from '@/components/ui/Absent'
import { Pill } from '@/components/ui/Pill'
import { RECORDER_LANDING_PATH, findRecorderSection } from '@/routes/recorder/sections'

/**
 * The screen behind a recorder section this console has not built yet.
 *
 * It exists because the rail lists every section — and a nav item that leads to
 * a blank page, a spinner, or worse an empty table that looks like "no data" is
 * a lie about the system's state. This page says plainly that the screen is not
 * built, names what it will show, and names the endpoint it will read, so
 * nobody mistakes an unbuilt console for a silent recorder.
 */
export function SectionNotBuiltPage() {
  const { section: sectionPath } = useParams()
  const section = sectionPath === undefined ? undefined : findRecorderSection(sectionPath)

  // A section this console has never heard of — a stale bookmark, or one of the
  // sections that has since been removed. Send the operator somewhere real rather
  // than rendering an empty frame. Sections that ARE built never reach this
  // component: React Router ranks their static route above `:section`.
  //
  // The target must be a section that still exists, and it is the same one
  // `/recorder` itself redirects to. Pointing at a removed section instead — this
  // said `/recorder/alerts` until the engine's alert queue superseded it — sends
  // the operator to a URL that lands back here and redirects again, forever.
  if (section === undefined) return <Navigate to={`/recorder/${RECORDER_LANDING_PATH}`} replace />

  return (
    <>
      <PageHeader
        eyebrow="recorder"
        title={section.label}
        lede="This screen is not built in this console yet. The recorder itself is answering — nothing below is a report that the recorder is empty or down."
      />
      <Absent
        title={
          <>
            <Pill tone="inert">not built here</Pill>
            {section.label}
          </>
        }
      >
        {section.summary}
      </Absent>
      <p className="muted mt-3.5">
        Will read{' '}
        {section.endpoints.map((endpoint, index) => (
          <span key={endpoint}>
            {index > 0 ? ' and ' : ''}
            <code className="readout text-[12px] text-[#a9d367]">{endpoint}</code>
          </span>
        ))}
        .
      </p>
    </>
  )
}
