import type { InputHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        'w-full rounded-sm border border-line-2 bg-[#090d10] px-2.5 py-2 font-sans text-[13px] text-fg',
        'focus-visible:border-signal-d focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal-d focus-visible:outline-offset-[1px]',
        className,
      )}
      {...props}
    />
  )
}
