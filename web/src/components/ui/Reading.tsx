import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/cn'

export interface ReadingProps extends HTMLAttributes<HTMLDivElement> {
  label: ReactNode
  value: ReactNode
  unit?: ReactNode
  alarm?: boolean
}

/** A single telemetry readout: eyebrow label + large tabular-mono value. */
export function Reading({ label, value, unit, alarm, className, ...props }: ReadingProps) {
  return (
    <div
      className={cn(
        'rounded-md border border-line bg-panel p-[14px_15px]',
        alarm && 'border-[#5c231b] bg-[#150e0d]',
        className,
      )}
      {...props}
    >
      <div className="eyebrow">{label}</div>
      <div
        className={cn(
          'readout mt-2 text-[24px] font-normal leading-[1.15]',
          alarm ? 'text-breach' : 'text-fg',
        )}
      >
        {value}
        {unit ? <small className="ml-[3px] text-[13px] text-dim">{unit}</small> : null}
      </div>
    </div>
  )
}
