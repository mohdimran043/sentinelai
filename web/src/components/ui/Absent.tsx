import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/cn'

export interface AbsentProps extends Omit<HTMLAttributes<HTMLDivElement>, 'title'> {
  title: ReactNode
  children?: ReactNode
}

/**
 * A tile that honestly says "there is nothing here yet" — used for data this
 * slice cannot answer (Phase 1C tiles) and for genuinely empty lists. Never a
 * silent blank; never a fabricated zero.
 */
export function Absent({ title, children, className, ...props }: AbsentProps) {
  return (
    <div
      className={cn(
        'rounded-md border border-dashed border-line-2 p-[15px]',
        '[background:repeating-linear-gradient(45deg,#0a0e11_0,#0a0e11_6px,#0d1216_6px,#0d1216_12px)]',
        className,
      )}
      {...props}
    >
      <h3 className="mb-[7px] flex items-center gap-2 text-[13px] font-semibold text-dim">
        {title}
      </h3>
      {children ? <p className="m-0 text-[12.5px] text-dimmer">{children}</p> : null}
    </div>
  )
}
