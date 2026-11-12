"use client";

import { IconMatches, IconNoMatch, IconUnsure } from "@/components/Icons";

/** The user's own judgement of one candidate; kept in the session only. */
export type Verdict = "matches" | "no-match" | "unsure";

const OPTIONS: { value: Verdict; label: string; Icon: () => React.JSX.Element; on: string }[] = [
  { value: "matches", label: "Matches", Icon: IconMatches, on: "aria-pressed:bg-sure aria-pressed:text-surface" },
  { value: "no-match", label: "Doesn't match", Icon: IconNoMatch, on: "aria-pressed:bg-danger aria-pressed:text-surface" },
  { value: "unsure", label: "Unsure", Icon: IconUnsure, on: "aria-pressed:bg-ink aria-pressed:text-surface" },
];

export const verdictLabel = (v: Verdict) => OPTIONS.find((o) => o.value === v)?.label ?? "";

/** Three-way segmented control; pressing the active option clears it. */
export default function VerdictControl({
  value,
  onChange,
  subject,
  withText,
}: {
  value: Verdict | undefined;
  onChange: (v: Verdict | undefined) => void;
  subject: string; // e.g. "candidate 2", for screen readers.
  withText?: boolean;
}) {
  return (
    <div role="group" aria-label={`Your verdict on ${subject}`} className="flex shrink-0 overflow-hidden rounded-control border border-rule-control">
      {OPTIONS.map(({ value: v, label, Icon, on }) => (
        <button
          key={v}
          type="button"
          title={label}
          aria-label={label}
          aria-pressed={value === v}
          onClick={(e) => {
            e.stopPropagation();
            onChange(value === v ? undefined : v);
          }}
          className={`flex h-7 items-center gap-1.5 border-r border-rule-control px-1.5 text-xs text-ink-2 last:border-r-0 hover:bg-paper hover:text-ink ${on}`}
        >
          <Icon />
          {withText && <span>{label}</span>}
        </button>
      ))}
    </div>
  );
}
