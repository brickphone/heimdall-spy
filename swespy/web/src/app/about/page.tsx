/* eslint-disable @next/next/no-img-element -- a single static, credited example photo */
import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import { LogoMark } from "@/components/Icons";
import about from "@/generated/about.json";
import crop from "@/generated/crop_example.json";

export const metadata: Metadata = { title: "How it works · Heimdall" };

interface Rate {
  count: number;
  n: number;
  rate: number;
  ci95: number[];
}

const pct = (v: number) => `${Math.round(v * 100)}%`;
const rate = (r: Rate) => `${r.count} of ${r.n} (${pct(r.rate)}, 95% interval ${pct(r.ci95[0])} to ${pct(r.ci95[1])})`;
const num = (n: number) => n.toLocaleString("en-US");
const km = (m: number) => (m < 1000 ? `${Math.round(m / 10) * 10} m` : `${(m / 1000).toFixed(1)} km`);

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-3 border-t border-rule py-8">
      <h2 className="text-lg font-semibold text-ink">{title}</h2>
      {children}
    </section>
  );
}

const P = ({ children }: { children: ReactNode }) => <p className="max-w-prose text-body text-ink">{children}</p>;

export default function About() {
  const shown = crop.attempts.filter((a) => a.image_id === crop.shown);
  const whole = shown.find((a) => !a.region);
  const cropped = shown.find((a) => a.region);
  const region = cropped?.region;
  const sources = about.bySource as Record<string, number>;
  const ft = about.failureTypes as Record<string, Rate>;

  return (
    <main className="min-h-dvh bg-paper">
      <div className="mx-auto flex max-w-3xl flex-col bg-surface px-6 py-8 desk:border-x desk:border-rule desk:px-10">
        <header className="flex items-center justify-between gap-4 pb-6">
          <span className="flex items-center gap-3">
            <LogoMark />
            <span className="text-md font-semibold text-ink">Heimdall</span>
          </span>
          <Link href="/" className="rounded-control border border-rule-control px-3 py-2 text-control text-ink no-underline hover:bg-paper">
            Back to the map
          </Link>
        </header>

        <h1 className="pb-4 text-xl font-semibold tracking-[var(--text-xl--letter-spacing)] text-ink">How it works</h1>
        <P>
          Heimdall estimates where an outdoor photo taken in Linköping was taken. It compares the
          photo with a library of street-level photos whose positions are known, and tells you
          how sure it is.
        </P>

        <Section title="What it does, step by step">
          <P>
            <strong className="font-semibold">A reference library.</strong> {num(about.references)} photos
            of Linköping&apos;s streets with known positions: {num(sources.mapillary ?? 0)} from
            Mapillary, {num(sources.wikimedia_commons ?? 0)} from Wikimedia Commons and{" "}
            {num(sources.kartaview ?? 0)} from KartaView.
          </P>
          <P>
            <strong className="font-semibold">Retrieval.</strong> A neural network (MegaLoc) turns
            your photo into a compact description of what it shows and finds the reference photos
            with the most similar descriptions.
          </P>
          <P>
            <strong className="font-semibold">Re-ranking.</strong> The closest references are
            checked in detail: it looks for small, distinctive details (corners,
            window edges, signs) that appear in both photos in a geometrically consistent way. Each
            such pair is a &quot;matching detail&quot;. The answer is the place of the best match,
            refined by nearby candidates that agree.
          </P>
          <P>
            <strong className="font-semibold">Confidence.</strong> The answer is marked{" "}
            <span className="font-medium text-sure">Confident</span> when the best match shares at
            least {about.minInliers} details with your photo and no other well-matching candidate
            points somewhere else. Otherwise it is marked{" "}
            <span className="font-medium text-unsure">Low confidence</span>: a lead to check, not an
            answer.
          </P>
        </Section>

        <Section title="How well it works">
          <P>
            Measured on {about.photos} development photos from Wikimedia Commons, taken by people
            in Linköping and never part of the reference library. These numbers come straight from
            the evaluation results ({about.source}).
          </P>
          <table className="w-full max-w-prose border-collapse text-body">
            <tbody>
              {[
                ["Within 100 m of the true place", rate(about.within100m)],
                ["Within 500 m", rate(about.within500m)],
                ["Typical error (median)", km(about.medianErrorM)],
                ["Marked Confident", `${about.confident.photos} photos, of which ${rate(about.confident.correct)} within 500 m`],
                ["Marked Low confidence", `${about.low.photos} photos, of which ${rate(about.low.correct)} within 500 m`],
                ["For comparison: always answering the city centre", `${rate(about.baselineWithin500m)} within 500 m`],
              ].map(([k, v]) => (
                <tr key={k} className="border-b border-rule-list align-top">
                  <th scope="row" className="py-2 pr-4 text-left font-normal text-ink-2">{k}</th>
                  <td className="py-2 text-ink">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <P>
            The confidence state is the most useful number: when it says Confident it is usually
            right, and when it says Low confidence it is usually wrong. With only {about.photos}{" "}
            photos the intervals are wide, so treat these as estimates.
          </P>
        </Section>

        <Section title="When it fails">
          <P>
            The most common failure is that the right place is not among the closest matches at
            all ({rate(ft.retrieval)} of the development photos), usually because no reference
            photo shows the same scene. Typical cases:
          </P>
          <ul className="flex max-w-prose list-disc flex-col gap-2 pl-5 text-body text-ink">
            <li>People or vehicles in the foreground hide the buildings behind them.</li>
            <li>Distant subjects, such as a skyline or a church tower far away, have few details in common with street-level photos.</li>
            <li>Elevated angles, from a balcony or a hill, look unlike the street-level references.</li>
            <li>Close-ups of artworks, plaques or statues that no street photo captures.</li>
          </ul>
          <P>
            Selecting the part of the photo that shows the place removes what covers the scene, but
            it can only help if the library has a photo of that place from a similar angle. We have
            not yet measured how often it helps. Here is a real example where it does not:
          </P>
          {region && whole && cropped && (
            <figure className="flex max-w-prose flex-col gap-3">
              <div className="relative w-full">
                <img src={crop.photo.file} alt="A white van parked in front of a brick building with an arched window" className="block w-full" />
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full" aria-hidden="true">
                  <path
                    fillRule="evenodd"
                    fill="var(--color-scrim)"
                    d={`M0 0H100V100H0Z M${region.x * 100} ${region.y * 100}h${region.width * 100}v${region.height * 100}h${-region.width * 100}Z`}
                  />
                </svg>
                <span
                  className="absolute border-[1.5px] border-surface"
                  style={{
                    left: `${region.x * 100}%`,
                    top: `${region.y * 100}%`,
                    width: `${region.width * 100}%`,
                    height: `${region.height * 100}%`,
                  }}
                  aria-hidden="true"
                />
              </div>
              <figcaption className="text-sm text-ink-2">
                Photo by {crop.photo.creator} on{" "}
                <a href={crop.photo.source_url} target="_blank" rel="noopener noreferrer">Wikimedia Commons</a>,{" "}
                <a href={crop.photo.license_url} target="_blank" rel="noopener noreferrer">CC BY-SA 4.0</a>, resized.
              </figcaption>
              <table className="w-full border-collapse text-body">
                <thead>
                  <tr className="border-b border-rule text-left text-sm text-ink-2">
                    <th className="py-2 pr-4 font-normal">Searched</th>
                    <th className="py-2 pr-4 font-normal">Error</th>
                    <th className="py-2 font-normal">State</th>
                  </tr>
                </thead>
                <tbody>
                  {[whole, cropped].map((a) => (
                    <tr key={a.label} className="border-b border-rule-list">
                      <td className="py-2 pr-4 text-ink">{a.label === "whole photo" ? "Whole photo" : "Selected region"}</td>
                      <td className="py-2 pr-4 text-ink">{km(a.error_m)}</td>
                      <td className={`py-2 ${a.state === "ok" ? "text-sure" : "text-unsure"}`}>
                        {a.state === "ok" ? "Confident" : "Low confidence"}, {a.top_inliers} details
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </figure>
          )}
          <P>
            The van covers most of the frame, so selecting the brick building behind it seems
            sensible. It still fails, because none of the library&apos;s photos shows that façade
            from this side. Both answers are correctly marked Low confidence. We tried{" "}
            {crop.attempts.filter((a) => a.region).length} hand-drawn regions on{" "}
            {new Set(crop.attempts.map((a) => a.image_id)).size} such photos, and none moved the answer
            closer; every attempt is recorded, none is hidden.
          </P>
        </Section>

        <Section title="Data and licences">
          <ul className="flex max-w-prose list-disc flex-col gap-2 pl-5 text-body text-ink">
            <li>Reference photos: Mapillary and KartaView (CC BY-SA 4.0) and Wikimedia Commons (CC0, CC BY or CC BY-SA, per photo). Each match is credited with its author and licence.</li>
            <li>Map: OpenStreetMap data (ODbL), © OpenStreetMap contributors, served by OpenFreeMap and drawn with MapLibre GL JS (BSD-3-Clause).</li>
            <li>Street and district names: OpenStreetMap (ODbL).</li>
            <li>Models: MegaLoc (MIT) for retrieval; ALIKED (BSD-3-Clause) and LightGlue (Apache-2.0) for matching details.</li>
            <li>Type: Familjen Grotesk (SIL Open Font License).</li>
            <li>No Google Street View imagery is shown anywhere in this demo.</li>
          </ul>
        </Section>

        <Section title="What it deliberately does not do">
          <ul className="flex max-w-prose list-disc flex-col gap-2 pl-5 text-body text-ink">
            <li>It is not publicly deployed. It runs on this computer only and is not offered as a public upload service.</li>
            <li>It does not identify or track people. It locates places, not persons, and never looks for faces.</li>
            <li>It does not read location data such as GPS from your file, and it does not store your photo.</li>
            <li>It does not work outside Linköping: the library only covers the city.</li>
          </ul>
        </Section>
      </div>
    </main>
  );
}
