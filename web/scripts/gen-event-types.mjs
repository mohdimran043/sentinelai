import { compileFromFile } from 'json-schema-to-typescript'
import { writeFileSync } from 'node:fs'

const ts = await compileFromFile('../contracts/events/anomaly_event.schema.json', {
  bannerComment:
    '/**\n * Generated from contracts/events/anomaly_event.schema.json via json-schema-to-typescript.\n * Run `npm run gen:events` to regenerate after the schema changes. Do not hand-edit.\n */',
  style: { semi: true, singleQuote: true },
})
writeFileSync(new URL('../src/events/anomalyEvent.types.ts', import.meta.url), ts)
