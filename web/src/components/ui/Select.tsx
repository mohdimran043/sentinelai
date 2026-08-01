import type { SelectHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

/** Matches the reference's `select` rule: 2px radius, #090d10 well, green focus ring. */
export function Select({ className, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        'w-full rounded-sm border border-line-2 bg-[#090d10] px-2.5 py-2 font-sans text-[13px] text-fg',
        'focus-visible:border-signal-d focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal-d focus-visible:outline-offset-[1px]',
        className,
      )}
      {...props}
    />
  )
}
