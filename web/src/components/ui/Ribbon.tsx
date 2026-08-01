import type { Tone } from '@/lib/severity'
import { cn } from '@/lib/cn'

export interface RibbonCell {
  tone: Tone
  /** Human-readable description of this bucket, used for the title tooltip and the
   * screen-reader-only summary table. */
  description: string
}

const cellTone: Record<Tone, string> = {
  nominal: 'bg-signal',
  caution: 'bg-caution',
  breach:
    '[background:repeating-linear-gradient(45deg,var(--color-breach)_0_2px,#7a2318_2px_4px)]',
  inert:
    'opacity-50 [background:repeating-linear-gradient(45deg,transparent_0_3px,var(--color-unknown)_3px_4px)]',
}

/**
 * The reference's signature device: a 46px strip of 1px-separated cells. Here it
 * carries threat-over-time for one camera, one cell per time bucket, coloured by
 * the worst severity observed in that bucket. This is the page's boldest visual
 * moment on purpose — everything around it stays quiet.
 */
export function Ribbon({ cells, ariaLabel }: { cells: RibbonCell[]; ariaLabel: string }) {
  return (
    <div>
      <div
        className="grid h-[46px] min-w-[520px] gap-px rounded-sm border border-line bg-void p-[3px]"
        style={{ gridAutoFlow: 'column', gridAutoColumns: 'minmax(1px, 1fr)' }}
        role="img"
        aria-label={ariaLabel}
      >
        {cells.map((cell, index) => (
          <div
            key={index}
            title={cell.description}
            className={cn('animate-rise rounded-[1px]', cellTone[cell.tone])}
          />
        ))}
      </div>
      <table className="sr-only">
        <caption>{ariaLabel}</caption>
        <thead>
          <tr>
            <th scope="col">Bucket</th>
            <th scope="col">Reading</th>
          </tr>
        </thead>
        <tbody>
          {cells.map((cell, index) => (
            <tr key={index}>
              <td>{index + 1}</td>
              <td>{cell.description}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
