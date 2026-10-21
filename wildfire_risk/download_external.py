"""Download the external wildfire predictor sources listed in `data_sources.py`.

Every NASA source here sits behind Earthdata Login, and FIRMS needs its own map key.
Run `check-credentials` first to confirm the machine is set up, then run one subcommand
per dataset. Downloads resume and skip files that are already complete, so re-running
after an interruption is cheap.
"""

from __future__ import annotations

import argparse
import logging
import math
import netrc
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter, sleep
from typing import Iterable, Iterator

import requests

LOGGER = logging.getLogger(__name__)

CMR_GRANULE_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
EARTHDATA_LOGIN_HOST = "urs.earthdata.nasa.gov"
EARTHDATA_AUTH_HOST_SUFFIXES = (".earthdata.nasa.gov", ".nasa.gov", ".usgs.gov")
FIRMS_AREA_API_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# The FIRMS area API rejects anything outside a 1 to 5 day range per request.
FIRMS_MAX_DAY_SPAN = 5
FIRMS_RATE_LIMIT_MESSAGE = "Exceeding allowed transaction limit"
# The transaction budget refills on a rolling 10 minute window.
FIRMS_RATE_LIMIT_WAIT_SECONDS = 90

# MODIS sinusoidal tile layout: 36 columns by 18 rows of 1111950.52 m squares.
MODIS_SPHERE_RADIUS_M = 6371007.181
MODIS_TILE_SIZE_M = 1111950.5196669999
MODIS_GRID_X_MIN = -20015109.354
MODIS_GRID_Y_MAX = 10007554.677


@dataclass(frozen=True)
class CollectionSpec:
    """A CMR collection plus the defaults we want for wildfire modeling."""

    key: str
    concept_id: str
    short_name: str
    version: str
    description: str
    granule_estimate: str


COLLECTIONS: dict[str, CollectionSpec] = {
    "mcd12q1": CollectionSpec(
        key="mcd12q1",
        concept_id="C2484079608-LPCLOUD",
        short_name="MCD12Q1",
        version="061",
        description="Annual 500 m land cover type, one HDF per MODIS sine tile per year",
        granule_estimate="about 330 tiles per year, roughly 2 GB per year",
    ),
    "nasadem": CollectionSpec(
        key="nasadem",
        concept_id="C2763264762-LPCLOUD",
        short_name="NASADEM_HGT",
        version="001",
        description="Merged 1 arc second DEM, one zip per 1 degree tile",
        granule_estimate="about 14000 tiles globally, roughly 180 GB",
    ),
    "nasadem-slope": CollectionSpec(
        key="nasadem-slope",
        concept_id="C2763264770-LPCLOUD",
        short_name="NASADEM_SC",
        version="001",
        description="Precomputed 1 arc second slope and curvature, one zip per 1 degree tile",
        granule_estimate="about 14000 tiles globally, roughly 250 GB",
    ),
    "gpw": CollectionSpec(
        key="gpw",
        concept_id="C3540910651-ESDIS",
        short_name="CIESIN_SEDAC_GPWv4_POPDENS_R11",
        version="4.11",
        description="GPWv4.11 population density, five-year snapshots from 2000 to 2020",
        granule_estimate="one zip per year and resolution, roughly 50 MB at 2.5 arc minutes",
    ),
    "groads": CollectionSpec(
        key="groads",
        concept_id="C3550192909-ESDIS",
        short_name="CIESIN_SEDAC_GROADS_V1",
        version="1.0",
        description="gROADSv1 road vectors, one geodatabase zip per region",
        granule_estimate="about 700 MB for the global geodatabase",
    ),
}


class MissingCredentialsError(RuntimeError):
    pass


class EarthdataSession(requests.Session):
    """A session that keeps Earthdata credentials across the URS redirect chain.

    `requests` drops the Authorization header when a redirect crosses hosts, which is
    exactly what Earthdata Login does on every data request, so the header has to be
    re-attached for NASA-operated hosts.
    """

    def __init__(self, username: str | None = None, password: str | None = None, token: str | None = None) -> None:
        super().__init__()
        self.headers["User-Agent"] = "hephaelion-wildfire-downloader"
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        elif username and password:
            self.auth = (username, password)
        else:
            raise MissingCredentialsError("EarthdataSession needs either a bearer token or a username and password")

    def rebuild_auth(self, prepared_request: requests.PreparedRequest, response: requests.Response) -> None:
        if self.auth is None:
            return  # Bearer tokens live in self.headers and survive redirects on their own.
        hostname = requests.utils.urlparse(prepared_request.url).hostname or ""
        if hostname.endswith(EARTHDATA_AUTH_HOST_SUFFIXES) or hostname == EARTHDATA_LOGIN_HOST:
            return
        super().rebuild_auth(prepared_request, response)


def resolve_earthdata_credentials() -> tuple[str | None, str | None, str | None]:
    """Return (username, password, token) from the environment or ~/.netrc."""
    token = os.environ.get("EARTHDATA_TOKEN")
    if token:
        return None, None, token.strip()

    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if username and password:
        return username, password, None

    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            entry = netrc.netrc(str(netrc_path)).authenticators(EARTHDATA_LOGIN_HOST)
        except netrc.NetrcParseError as exc:
            raise MissingCredentialsError(f"Could not parse {netrc_path}: {exc}") from exc
        if entry:
            return entry[0], entry[2], None

    return None, None, None


def create_earthdata_session() -> EarthdataSession:
    username, password, token = resolve_earthdata_credentials()
    if not token and not (username and password):
        raise MissingCredentialsError(
            "No Earthdata credentials found. Register at https://urs.earthdata.nasa.gov/users/new "
            "and then do one of:\n"
            "  1. export EARTHDATA_TOKEN=... (generate one at https://urs.earthdata.nasa.gov/profile)\n"
            "  2. export EARTHDATA_USERNAME=... and export EARTHDATA_PASSWORD=...\n"
            "  3. add `machine urs.earthdata.nasa.gov login USER password PASS` to ~/.netrc "
            "and run `chmod 600 ~/.netrc`"
        )
    return EarthdataSession(username=username, password=password, token=token)


def resolve_firms_map_key() -> str:
    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        raise MissingCredentialsError(
            "No FIRMS map key found. Request a free key at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/ and then "
            "export FIRMS_MAP_KEY=..."
        )
    return map_key.strip()


def search_cmr_granules(
    concept_id: str,
    temporal: str | None = None,
    bounding_box: str | None = None,
    granule_filter: str | None = None,
    page_size: int = 2000,
) -> list[dict]:
    """Page through CMR and return granule metadata, newest API path (search-after)."""
    params: dict[str, object] = {"collection_concept_id": concept_id, "page_size": page_size}
    if temporal:
        params["temporal"] = temporal
    if bounding_box:
        params["bounding_box"] = bounding_box
    if granule_filter:
        params["readable_granule_name"] = granule_filter
        params["options[readable_granule_name][pattern]"] = "true"

    granules: list[dict] = []
    search_after: str | None = None
    while True:
        headers = {"CMR-Search-After": search_after} if search_after else {}
        response = requests.get(CMR_GRANULE_SEARCH_URL, params=params, headers=headers, timeout=120)
        response.raise_for_status()
        items = response.json().get("items", [])
        granules.extend(items)
        search_after = response.headers.get("CMR-Search-After")
        if not search_after or not items:
            return granules


def granule_download_urls(granule: dict) -> list[str]:
    urls = []
    for related_url in granule.get("umm", {}).get("RelatedUrls", []):
        if not related_url.get("Type", "").startswith("GET DATA"):
            continue
        url = related_url.get("URL", "")
        if url.startswith("http") and not url.endswith((".xml", ".jpg", ".png", ".met")):
            urls.append(url)
    return urls


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    max_attempts: int = 4,
) -> Path:
    """Stream a URL to disk, resuming a partial `.part` file when the server allows it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, max_attempts + 1):
        already_downloaded = partial_path.stat().st_size if partial_path.exists() else 0
        headers = {"Range": f"bytes={already_downloaded}-"} if already_downloaded else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                if response.status_code in {401, 403}:
                    raise MissingCredentialsError(
                        f"Earthdata rejected the request for {url} with HTTP {response.status_code}. "
                        "Confirm the credentials are valid and that the data set is authorised for your "
                        "account at https://urs.earthdata.nasa.gov/profile (some LP DAAC and SEDAC "
                        "collections need a one-time EULA acceptance)."
                    )
                response.raise_for_status()
                # A server that ignores Range restarts the body, so the partial file must be replaced.
                mode = "ab" if response.status_code == 206 and already_downloaded else "wb"
                with open(partial_path, mode) as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            partial_path.replace(destination)
            return destination
        except MissingCredentialsError:
            raise
        except (requests.RequestException, OSError) as exc:
            if attempt == max_attempts:
                raise
            backoff_seconds = 2**attempt
            LOGGER.warning(
                "Attempt %s/%s failed for %s (%s). Retrying in %ss",
                attempt,
                max_attempts,
                destination.name,
                exc,
                backoff_seconds,
            )
            sleep(backoff_seconds)
    return destination


def download_all(
    session: requests.Session,
    urls: Iterable[str],
    dest_dir: Path,
    label: str,
) -> list[Path]:
    urls = list(dict.fromkeys(urls))
    dest_dir.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()
    written: list[Path] = []

    for index, url in enumerate(urls, start=1):
        destination = dest_dir / url.rsplit("/", 1)[-1].split("?")[0]
        if destination.exists() and destination.stat().st_size > 0:
            written.append(destination)
            continue
        LOGGER.info("[%s] %s/%s downloading %s", label, index, len(urls), destination.name)
        written.append(download_file(session, url, destination))
        if index % 25 == 0:
            LOGGER.info(
                "[%s] %s/%s complete after %.1fm",
                label,
                index,
                len(urls),
                (perf_counter() - started_at) / 60.0,
            )

    LOGGER.info(
        "[%s] finished %s files in %s after %.1fm",
        label,
        len(written),
        dest_dir,
        (perf_counter() - started_at) / 60.0,
    )
    return written


def month_starts(start_month: str, end_month: str) -> list[date]:
    start = date(int(start_month[:4]), int(start_month[5:7]), 1)
    end = date(int(end_month[:4]), int(end_month[5:7]), 1)
    if start > end:
        raise ValueError("start_month must not be after end_month")
    months = []
    cursor = start
    while cursor <= end:
        months.append(cursor)
        cursor = date(cursor.year + (cursor.month == 12), (cursor.month % 12) + 1, 1)
    return months


def firms_date_windows(start_month: str, end_month: str) -> Iterator[tuple[date, int]]:
    """Yield (window_start, day_count) pairs no longer than the FIRMS per-request cap."""
    months = month_starts(start_month, end_month)
    period_start = months[0]
    last_month = months[-1]
    next_month = date(last_month.year + (last_month.month == 12), (last_month.month % 12) + 1, 1)
    period_end = next_month - timedelta(days=1)

    cursor = period_start
    while cursor <= period_end:
        day_count = min(FIRMS_MAX_DAY_SPAN, (period_end - cursor).days + 1)
        yield cursor, day_count
        cursor += timedelta(days=day_count)


def _fetch_firms_window(
    session: requests.Session,
    url: str,
    window_start: date,
    max_attempts: int = 12,
) -> str:
    """Fetch one FIRMS window, waiting out the transaction allowance when it runs dry.

    FIRMS meters by data volume rather than request count, and a global request is worth
    roughly 180 transactions against a 5000 per 10 minute budget. Hitting the ceiling
    returns HTTP 400 with a plain-text message, so the body has to be inspected before
    the status code is treated as fatal.
    """
    for attempt in range(1, max_attempts + 1):
        response = session.get(url, timeout=(30, 600))
        body = response.text

        if response.status_code == 401 or "invalid map key" in body.lower():
            raise MissingCredentialsError(
                "FIRMS rejected the map key. Request a new one and re-export FIRMS_MAP_KEY."
            )

        if FIRMS_RATE_LIMIT_MESSAGE in body:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"FIRMS transaction limit still exhausted for {window_start} after "
                    f"{max_attempts} attempts. Wait 10 minutes and re-run to resume."
                )
            LOGGER.info(
                "[firms] transaction allowance exhausted at %s, waiting %ss for the window to reset",
                window_start,
                FIRMS_RATE_LIMIT_WAIT_SECONDS,
            )
            sleep(FIRMS_RATE_LIMIT_WAIT_SECONDS)
            continue

        response.raise_for_status()
        if body.lstrip().lower().startswith("invalid"):
            raise RuntimeError(f"FIRMS returned an error for {window_start}: {body[:200]}")
        return body

    raise RuntimeError(f"FIRMS never returned data for {window_start}")


def download_firms_archive(
    dest_dir: Path,
    start_month: str,
    end_month: str,
    source: str = "MODIS_SP",
    area: str = "world",
    sleep_seconds: float = 1.0,
) -> list[Path]:
    """Pull FIRMS active-fire detections in ten-day windows via the area API."""
    map_key = resolve_firms_map_key()
    dest_dir.mkdir(parents=True, exist_ok=True)
    windows = list(firms_date_windows(start_month, end_month))
    LOGGER.info(
        "Downloading FIRMS %s detections for %s in %s windows of up to %s days",
        source,
        area,
        len(windows),
        FIRMS_MAX_DAY_SPAN,
    )

    session = requests.Session()
    session.headers["User-Agent"] = "hephaelion-wildfire-downloader"
    written: list[Path] = []
    for index, (window_start, day_count) in enumerate(windows, start=1):
        destination = dest_dir / f"firms_{source.lower()}_{window_start.isoformat()}_{day_count}d.csv"
        if destination.exists() and destination.stat().st_size > 0:
            written.append(destination)
            continue

        url = f"{FIRMS_AREA_API_URL}/{map_key}/{source}/{area}/{day_count}/{window_start.isoformat()}"
        body = _fetch_firms_window(session, url, window_start)
        destination.write_text(body, encoding="utf-8")
        written.append(destination)
        LOGGER.info(
            "[firms] %s/%s wrote %s (%s detection rows)",
            index,
            len(windows),
            destination.name,
            max(body.count("\n") - 1, 0),
        )
        sleep(sleep_seconds)

    LOGGER.info("[firms] finished %s window files in %s", len(written), dest_dir)
    return written


def modis_tiles_for_bbox(bounding_box: str) -> list[str]:
    """Return the MODIS sine tiles that intersect a `west,south,east,north` box.

    MODIS tiles are a regular 36x18 layout in sinusoidal projection metres, not in
    degrees, so a tile spans a wider band of longitude the further it sits from the
    equator. The horizontal span is therefore measured at whichever latitude in the box
    is closest to the equator, which is where the projected x extent is widest.
    """
    west, south, east, north = (float(value) for value in bounding_box.split(","))
    if west > east or south > north:
        raise ValueError(f"Bounding box must be west,south,east,north: got {bounding_box}")

    widest_latitude = 0.0 if south <= 0.0 <= north else min((south, north), key=abs)
    scale = math.cos(math.radians(widest_latitude))

    def h_index(longitude: float) -> int:
        x = MODIS_SPHERE_RADIUS_M * math.radians(longitude) * scale
        return int((x - MODIS_GRID_X_MIN) // MODIS_TILE_SIZE_M)

    def v_index(latitude: float) -> int:
        y = MODIS_SPHERE_RADIUS_M * math.radians(latitude)
        return int((MODIS_GRID_Y_MAX - y) // MODIS_TILE_SIZE_M)

    h_start = max(min(h_index(west), 35), 0)
    h_end = max(min(h_index(east), 35), 0)
    v_start = max(min(v_index(north), 17), 0)
    v_end = max(min(v_index(south), 17), 0)
    return [
        f"h{h:02d}v{v:02d}"
        for v in range(v_start, v_end + 1)
        for h in range(h_start, h_end + 1)
    ]


def command_check_credentials(args: argparse.Namespace) -> int:
    exit_code = 0

    username, _, token = resolve_earthdata_credentials()
    if token:
        print("Earthdata: bearer token found in EARTHDATA_TOKEN")
    elif username:
        print(f"Earthdata: username/password found for `{username}`")
    else:
        print("Earthdata: MISSING - needed for mcd12q1, nasadem, gpw, and groads")
        exit_code = 1

    if token or username:
        try:
            session = create_earthdata_session()
            response = session.get("https://urs.earthdata.nasa.gov/profile", timeout=60, allow_redirects=True)
            verdict = "accepted" if response.status_code < 400 else f"rejected with HTTP {response.status_code}"
            print(f"Earthdata: credentials {verdict}")
            if response.status_code >= 400:
                exit_code = 1
        except (requests.RequestException, MissingCredentialsError) as exc:
            print(f"Earthdata: could not verify credentials ({exc})")
            exit_code = 1

    if os.environ.get("FIRMS_MAP_KEY"):
        print("FIRMS: map key found in FIRMS_MAP_KEY")
    else:
        print("FIRMS: MISSING - needed for the firms subcommand")
        exit_code = 1

    return exit_code


def command_list(args: argparse.Namespace) -> int:
    print(f"{'dataset':<15} {'short name':<32} {'version':<8} size estimate")
    print("-" * 110)
    for spec in COLLECTIONS.values():
        print(f"{spec.key:<15} {spec.short_name:<32} {spec.version:<8} {spec.granule_estimate}")
        print(f"{'':<15} {spec.description}")
    print(f"{'firms':<15} {'MODIS/VIIRS active fire':<32} {'-':<8} about 20 million detections for 2020-2025 global")
    print(f"{'':<15} Active-fire point detections pulled from the FIRMS area API")
    return 0


def command_mcd12q1(args: argparse.Namespace) -> int:
    spec = COLLECTIONS["mcd12q1"]
    temporal = f"{args.start_year}-01-01T00:00:00Z,{args.end_year}-12-31T23:59:59Z"
    granules = search_cmr_granules(
        concept_id=spec.concept_id,
        temporal=temporal,
        bounding_box=args.bbox,
    )
    tile_filter = set(args.tiles or [])
    if args.bbox and not tile_filter:
        tile_filter = set(modis_tiles_for_bbox(args.bbox))

    urls = []
    for granule in granules:
        granule_ur = granule.get("umm", {}).get("GranuleUR", "")
        if tile_filter and not any(tile in granule_ur for tile in tile_filter):
            continue
        urls.extend(granule_download_urls(granule))

    LOGGER.info("MCD12Q1: %s granules matched, %s files to fetch", len(granules), len(urls))
    if not urls:
        LOGGER.error("No MCD12Q1 granules matched. Widen --bbox, --tiles, or the year range.")
        return 1
    download_all(create_earthdata_session(), urls, Path(args.dest), label="mcd12q1")
    return 0


def command_nasadem(args: argparse.Namespace) -> int:
    spec = COLLECTIONS["nasadem-slope" if args.product == "slope" else "nasadem"]
    if not args.bbox and not args.allow_global:
        LOGGER.error(
            "NASADEM is %s. Pass --bbox west,south,east,north to limit the download, "
            "or --allow-global to accept the full volume.",
            spec.granule_estimate,
        )
        return 1

    granules = search_cmr_granules(concept_id=spec.concept_id, bounding_box=args.bbox)
    urls = [url for granule in granules for url in granule_download_urls(granule)]
    LOGGER.info("%s: %s granules matched, %s files to fetch", spec.short_name, len(granules), len(urls))
    if not urls:
        LOGGER.error("No %s granules matched the requested area.", spec.short_name)
        return 1
    download_all(create_earthdata_session(), urls, Path(args.dest), label=spec.key)
    return 0


def command_gpw(args: argparse.Namespace) -> int:
    spec = COLLECTIONS["gpw"]
    granules = search_cmr_granules(concept_id=spec.concept_id)
    wanted_years = {str(year) for year in args.years}

    urls = []
    for granule in granules:
        granule_ur = granule.get("umm", {}).get("GranuleUR", "")
        if args.resolution not in granule_ur or args.file_format not in granule_ur:
            continue
        if wanted_years and not any(year in granule_ur for year in wanted_years):
            continue
        urls.extend(granule_download_urls(granule))

    LOGGER.info("GPWv4: %s granules available, %s files to fetch", len(granules), len(urls))
    if not urls:
        available = sorted(granule.get("umm", {}).get("GranuleUR", "") for granule in granules)
        LOGGER.error(
            "No GPWv4 granules matched resolution `%s` and format `%s`. Available granules:\n  %s",
            args.resolution,
            args.file_format,
            "\n  ".join(available),
        )
        return 1
    download_all(create_earthdata_session(), urls, Path(args.dest), label="gpw")
    return 0


def command_groads(args: argparse.Namespace) -> int:
    spec = COLLECTIONS["groads"]
    granules = search_cmr_granules(concept_id=spec.concept_id)
    urls = []
    for granule in granules:
        granule_ur = granule.get("umm", {}).get("GranuleUR", "")
        if args.region not in granule_ur or args.file_format not in granule_ur:
            continue
        urls.extend(granule_download_urls(granule))

    LOGGER.info("gROADSv1: %s granules available, %s files to fetch", len(granules), len(urls))
    if not urls:
        available = sorted(granule.get("umm", {}).get("GranuleUR", "") for granule in granules)
        LOGGER.error(
            "No gROADS granules matched region `%s` and format `%s`. Available granules:\n  %s",
            args.region,
            args.file_format,
            "\n  ".join(available),
        )
        return 1
    download_all(create_earthdata_session(), urls, Path(args.dest), label="groads")
    return 0


def command_firms(args: argparse.Namespace) -> int:
    download_firms_archive(
        dest_dir=Path(args.dest),
        start_month=args.start_month,
        end_month=args.end_month,
        source=args.source,
        area=args.area,
        sleep_seconds=args.sleep_seconds,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download external wildfire predictor sources from NASA Earthdata and FIRMS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="Show the known datasets and their size estimates").set_defaults(
        handler=command_list
    )
    subparsers.add_parser(
        "check-credentials", help="Verify that Earthdata and FIRMS credentials are usable"
    ).set_defaults(handler=command_check_credentials)

    firms = subparsers.add_parser("firms", help="Download FIRMS active-fire detections")
    firms.add_argument("--dest", default="datasets/wildfires/firms")
    firms.add_argument("--start-month", default="2020-01")
    firms.add_argument("--end-month", default="2025-12")
    firms.add_argument(
        "--source",
        default="MODIS_SP",
        help="FIRMS source. `MODIS_SP` and `VIIRS_SNPP_SP` are the archive-quality feeds.",
    )
    firms.add_argument("--area", default="world", help="`world` or a west,south,east,north box")
    firms.add_argument("--sleep-seconds", type=float, default=1.0, help="Pause between requests to respect rate limits")
    firms.set_defaults(handler=command_firms)

    mcd12q1 = subparsers.add_parser("mcd12q1", help="Download MCD12Q1 annual land cover tiles")
    mcd12q1.add_argument("--dest", default="datasets/wildfires/mcd12q1")
    mcd12q1.add_argument("--start-year", type=int, default=2019)
    mcd12q1.add_argument("--end-year", type=int, default=2024)
    mcd12q1.add_argument("--bbox", help="Limit to west,south,east,north")
    mcd12q1.add_argument("--tiles", nargs="*", help="Explicit MODIS tiles such as h08v05")
    mcd12q1.set_defaults(handler=command_mcd12q1)

    nasadem = subparsers.add_parser("nasadem", help="Download NASADEM terrain tiles")
    nasadem.add_argument("--dest", default="datasets/wildfires/nasadem")
    nasadem.add_argument("--product", choices=("hgt", "slope"), default="hgt")
    nasadem.add_argument("--bbox", help="Limit to west,south,east,north")
    nasadem.add_argument("--allow-global", action="store_true", help="Accept the full global volume")
    nasadem.set_defaults(handler=command_nasadem)

    gpw = subparsers.add_parser("gpw", help="Download GPWv4.11 population density")
    gpw.add_argument("--dest", default="datasets/wildfires/gpw")
    gpw.add_argument("--years", nargs="*", type=int, default=[2020])
    gpw.add_argument(
        "--resolution",
        default="2pt5_min",
        choices=("30_sec", "2pt5_min", "15_min", "30_min", "1_deg"),
        help="2.5 arc minutes is already finer than the 0.5 degree wildfire grid",
    )
    gpw.add_argument("--file-format", default="tif", choices=("tif", "asc", "nc"))
    gpw.set_defaults(handler=command_gpw)

    groads = subparsers.add_parser("groads", help="Download gROADSv1 road vectors")
    groads.add_argument("--dest", default="datasets/wildfires/groads")
    groads.add_argument("--region", default="global", help="`global`, `africa`, `asia`, `americas`, `europe`, ...")
    groads.add_argument("--file-format", default="gdb", choices=("gdb", "shp", "csv"))
    groads.set_defaults(handler=command_groads)

    return parser


def glue_negative_option_values(argv: list[str]) -> list[str]:
    """Keep `--bbox -130,25,-100,52` from being parsed as an unknown flag."""
    glued: list[str] = []
    skip_next = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token in {"--bbox", "--area"} and index + 1 < len(argv):
            nxt = argv[index + 1]
            if nxt.startswith("-") and not nxt.startswith("--"):
                glued.append(f"{token}={nxt}")
                skip_next = True
                continue
        glued.append(token)
    return glued


def main(argv: list[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(glue_negative_option_values(raw))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        return args.handler(args)
    except MissingCredentialsError as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
