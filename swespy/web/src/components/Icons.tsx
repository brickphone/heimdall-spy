// Icons: IBM Carbon (@carbon/icons-react, Apache-2.0; see docs/LICENSES.md), re-exported
// under the names the app uses, at 16 px. Decorative: every button that uses one carries
// its own aria-label. Only the logo mark below is our own drawing.
import {
  Add,
  ArrowLeft,
  ArrowsHorizontal,
  CheckmarkOutline,
  Close,
  CloseOutline,
  Compass,
  Copy,
  CubeView,
  Help,
  RecentlyViewed,
  SplitScreen,
  Subtract,
  Upload,
  Information,
} from "@carbon/icons-react";

const icon = (C: typeof Add) =>
  function CarbonIcon() {
    return <C size={16} aria-hidden="true" />;
  };

export const IconNew = icon(Add);
export const IconRecent = icon(RecentlyViewed);
export const IconHelp = icon(Information);
export const IconBack = icon(ArrowLeft);
export const IconSwap = icon(ArrowsHorizontal);
export const IconClose = icon(Close);
export const IconPlus = icon(Add);
export const IconMinus = icon(Subtract);
export const IconCompass = icon(Compass);
export const IconCopy = icon(Copy);
export const Icon3d = icon(CubeView);
export const IconSideBySide = icon(SplitScreen);
export const IconUpload = icon(Upload);
export const IconMatches = icon(CheckmarkOutline);
export const IconNoMatch = icon(CloseOutline);
export const IconUnsure = icon(Help);

/** Two concentric ink circles with a cobalt centre: rings around a located point. */
export function LogoMark() {
  return (
    <svg viewBox="0 0 28 28" aria-hidden="true" className="size-6">
      <circle cx="14" cy="14" r="11.5" fill="none" stroke="var(--color-ink)" strokeWidth="1.4" />
      <circle cx="14" cy="14" r="7" fill="none" stroke="var(--color-ink)" strokeWidth="1.4" />
      <circle cx="14" cy="14" r="2.5" fill="var(--color-pin)" />
    </svg>
  );
}
