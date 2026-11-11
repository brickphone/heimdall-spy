import Workspace from "@/components/Workspace";
import about from "@/generated/about.json";

export default function Home() {
  return (
    <main className="h-dvh">
      <Workspace references={about.references} area={about.areaPolygon as [number, number][]} />
    </main>
  );
}
