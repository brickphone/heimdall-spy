import Demo from "@/components/Demo";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-6 sm:py-10">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Where was this photo taken?</h1>
        <p className="mt-1 text-muted">
          Linköping only. Your photo is processed in memory and not stored unless you tick the
          box. Location metadata (EXIF/GPS) in the file is ignored.
        </p>
      </header>
      <Demo />
    </main>
  );
}
