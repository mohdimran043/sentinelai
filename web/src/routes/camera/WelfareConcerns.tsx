import type { Confidence, WelfareConcernEntry } from '@/api/engineClient'
import { Pill } from '@/components/ui/Pill'
import type { Tone } from '@/lib/severity'
import { humanizeEnum } from '@/lib/format'

/**
 * `likely` is the tier that notifies by default, so it reads on the caution
 * band; `possible` stays inert. Deliberately not `nominal` for either — green is
 * this console's "everything is fine" colour and a welfare concern is never
 * that. Deliberately not `breach` for `likely` either: a single frame cannot
 * earn the throbbing red the console spends on a critical threat score, and
 * spending it here would make the two indistinguishable to someone scanning.
 */
function toneForConfidence(confidence: Confidence): Tone {
  return confidence === 'likely' ? 'caution' : 'inert'
}

/**
 * What the vision model said about a person's wellbeing in one keyframe.
 *
 * Shows **every** concern the event carried, including kinds this camera's
 * `notify_on` excludes: that policy decides who gets paged, not what an operator
 * looking straight at the camera page is allowed to see. Muting a camera must
 * silence its notifications without also blinding its console.
 */
export function WelfareConcerns({ concerns }: { concerns: readonly WelfareConcernEntry[] }) {
  // Nothing at all rather than an empty section: a heading over no rows reads as
  // "not assessed", which is a different claim from "reported nothing".
  if (concerns.length === 0) return null

  return (
    <div className="mt-2">
      <ul className="m-0 list-none p-0">
        {concerns.map((concern, index) => (
          <li
            // `kind` is unique per assessment — duplicates collapse in the domain
            // — but the index keeps this stable if that ever stops being true.
            key={`${concern.kind}-${index}`}
            className="mt-1 flex flex-wrap items-baseline gap-2 first:mt-0"
            data-testid="welfare-concern"
            data-confidence={concern.confidence}
          >
            <Pill tone={toneForConfidence(concern.confidence)}>{humanizeEnum(concern.kind)}</Pill>
            <span className="text-[12px] text-dim">{concern.confidence}</span>
            {concern.evidence_stated ? (
              <span className="text-[13px] text-fg">{concern.evidence}</span>
            ) : (
              // The `evidence` string is a fixed placeholder here, not something
              // the model said. Rendering it as prose would attribute an
              // observation to a model that made none.
              <span className="muted italic">
                No reason given — the model named this concern but described nothing.
              </span>
            )}
          </li>
        ))}
      </ul>
      <p className="muted mt-1" data-testid="welfare-concerns-note">
        A vision-language model's opinion about a single frame — not a detector's
        finding. It has no memory of the frame before, and it misses things.
      </p>
    </div>
  )
}
