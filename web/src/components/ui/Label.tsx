import type { LabelHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

export function Label({ className, ...props }: LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={cn('mt-[13px] mb-[5px] block text-[13px] text-fg', className)} {...props} />
}
