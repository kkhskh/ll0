import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HEPHAELION EARTH",
  description: "Hephaelion Earth. An interactive 3D globe with streamed imagery and terrain.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
