// Inline stroke icons (1.4 px stroke from the --icon-stroke token). Decorative: every
// button that uses one carries its own aria-label.
import type { SVGProps } from "react";

function Svg(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="var(--icon-stroke)"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="size-icon"
      {...props}
    />
  );
}

export const IconNew = () => (
  <Svg>
    <rect x="3.5" y="3.5" width="13" height="13" />
    <path d="M10 7v6M7 10h6" />
  </Svg>
);

export const IconRecent = () => (
  <Svg>
    <circle cx="10" cy="10" r="6.5" />
    <path d="M10 6.5V10l2.5 1.8" />
  </Svg>
);

export const IconCoverage = () => (
  <Svg>
    <path d="M3 13.5c3-1 4.5-5 7-5s4 3 7 2" strokeDasharray="0.1 2.6" />
    <path d="M3 7.5c3-1 4.5-3 7-3s4 1.5 7 1" />
  </Svg>
);

export const IconHelp = () => (
  <Svg>
    <circle cx="10" cy="10" r="6.5" />
    <path d="M8.2 8.2a1.9 1.9 0 1 1 2.6 1.8c-.6.3-.8.6-.8 1.2M10 13.6v.1" />
  </Svg>
);

export const IconBack = () => (
  <Svg>
    <path d="M12 4.5 6.5 10l5.5 5.5" />
  </Svg>
);

export const IconSwap = () => (
  <Svg>
    <path d="M7 6.5 3.5 10 7 13.5M13 6.5l3.5 3.5-3.5 3.5M3.5 10h13" />
  </Svg>
);

export const IconClose = () => (
  <Svg>
    <path d="M5 5l10 10M15 5 5 15" />
  </Svg>
);

export const IconPlus = () => (
  <Svg>
    <path d="M10 4.5v11M4.5 10h11" />
  </Svg>
);

export const IconMinus = () => (
  <Svg>
    <path d="M4.5 10h11" />
  </Svg>
);

/** Two concentric ink circles with a cobalt centre: rings around a located point. */
export function LogoMark() {
  return (
    <svg viewBox="0 0 28 28" aria-hidden="true" className="size-7">
      <circle cx="14" cy="14" r="11.5" fill="none" stroke="var(--color-ink)" strokeWidth="1.4" />
      <circle cx="14" cy="14" r="7" fill="none" stroke="var(--color-ink)" strokeWidth="1.4" />
      <circle cx="14" cy="14" r="2.5" fill="var(--color-pin)" />
    </svg>
  );
}
