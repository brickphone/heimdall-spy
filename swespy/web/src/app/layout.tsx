import type { Metadata, Viewport } from "next";
import { Familjen_Grotesk } from "next/font/google";
import "./globals.css";

// One family, three weights. latin covers å, ä and ö; latin-ext covers other European names.
const familjen = Familjen_Grotesk({
  variable: "--font-familjen",
  subsets: ["latin", "latin-ext"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "SweSpy",
  description: "Find where a photo taken in Linköping was taken, with an honest confidence state.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${familjen.variable} h-full`}>
      <body className="h-full">{children}</body>
    </html>
  );
}
