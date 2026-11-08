import { NextResponse } from "next/server";
import { parseNominatim } from "@/geo/nominatim";

const USER_AGENT =
  "HephaelionEarth/0.1 (foundation globe; +https://github.com/kkhskh/ll0)";

let lastRequestAt = 0;

export async function GET(request: Request) {
  const query = new URL(request.url).searchParams.get("q")?.trim() ?? "";
  if (query.length < 2 || query.length > 200) {
    return NextResponse.json({ results: [] });
  }

  const now = Date.now();
  if (now - lastRequestAt < 1000) {
    return NextResponse.json(
      {
        results: [],
        error: "Search is pausing to respect the gazetteer rate limit. Try again in a moment.",
      },
      { status: 429 },
    );
  }
  lastRequestAt = now;

  try {
    const url = new URL("https://nominatim.openstreetmap.org/search");
    url.searchParams.set("q", query);
    url.searchParams.set("format", "jsonv2");
    url.searchParams.set("limit", "5");
    url.searchParams.set("addressdetails", "0");
    const response = await fetch(url, {
      headers: {
        Accept: "application/json",
        "Accept-Language": "en",
        "User-Agent": USER_AGENT,
      },
      cache: "no-store",
    });
    if (!response.ok) {
      return NextResponse.json(
        {
          results: [],
          error: "The public gazetteer is unavailable. The globe is unaffected.",
        },
        { status: 502 },
      );
    }
    const payload: unknown = await response.json();
    return NextResponse.json({ results: parseNominatim(payload) });
  } catch {
    return NextResponse.json(
      {
        results: [],
        error: "The public gazetteer is unavailable. The globe is unaffected.",
      },
      { status: 502 },
    );
  }
}
