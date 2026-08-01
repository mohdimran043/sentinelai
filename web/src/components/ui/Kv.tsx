import type { ReactNode } from 'react'

export interface KvRow {
  key: string
  label: ReactNode
  value: ReactNode
}

/** Key/value list matching the reference's `.kv` rows — label left, tabular value right. */
export function KvList({ rows }: { rows: KvRow[] }) {
  return (
    <dl className="m-0">
      {rows.map((row) => (
        <div
          key={row.key}
          className="flex justify-between gap-3 border-b border-[#151d24] py-[5px] text-[12.5px] last:border-b-0"
        >
          <dt className="text-dim">{row.label}</dt>
          <dd className="readout m-0">{row.value}</dd>
        </div>
      ))}
    </dl>
  )
}
