import argparse
import json
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from curl_cffi import requests
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

BASE_URL = "https://steamrip.com"
LIST_SUFFIX = "/games-list-page"
SITE_NAME = "SteamRip"
OUTPUT_PATH = "steamrip.json"
LOG_PATH = "errors.log"

MAX_WORKERS = 10
BATCH_SIZE = 10
BATCH_DELAY_RANGE = (2, 5)

IMPERSONATE = "chrome110"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2.0

FAST_MAX_WORKERS = 30
FAST_BATCH_SIZE = 30

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

RELATIVE_UNIT_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2629800,
    "year": 31557600,
}

RELATIVE_AGO_RE = re.compile(
    r"^(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago$", re.IGNORECASE
)


def build_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("scraper")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)

    return logger


def batch_delay() -> float:
    return round(random.uniform(*BATCH_DELAY_RANGE), 2)


def fetch(url: str, logger: logging.Logger) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                impersonate=IMPERSONATE,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            logger.warning("Attempt %d/%d failed for %s — %s", attempt, MAX_RETRIES, url, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    logger.warning("All %d attempts failed for %s. Skipping.", MAX_RETRIES, url)
    return None


def resolve_listing_url(href: str, base_url: str) -> str:
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    if href.startswith(("http://", "https://")):
        return href
    return base_url.rstrip("/") + "/" + href.lstrip("/")


def normalise_download_url(href: str) -> str:
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    return href


BRACKETED_RE = re.compile(r"[\(\[\{][^\(\)\[\]\{\}]*[\)\]\}]")
FREE_DOWNLOAD_RE = re.compile(r"(?i)\bfree\s+download\b")


def clean_title(raw_title: str) -> str:
    text = raw_title

    previous = None
    while previous != text:
        previous = text
        text = BRACKETED_RE.sub("", text)

    text = FREE_DOWNLOAD_RE.sub("", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_listing(html: str, base_url: str, logger: logging.Logger) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = soup.find_all("li", class_="az-list-item")

    if not items:
        logger.warning("No <li class='az-list-item'> elements found on the listing page.")
        return []

    results = []
    for li in items:
        anchor = li.find("a")
        if not anchor:
            logger.warning("Found an <li class='az-list-item'> with no <a> child. Skipping.")
            continue

        href = anchor.get("href", "").strip()
        title = clean_title(anchor.get_text(strip=True))

        if not href:
            logger.warning("Empty href for item '%s'. Skipping.", title)
            continue

        results.append({"title": title, "url": resolve_listing_url(href, base_url)})

    logger.info("Found %d items on the listing page.", len(results))
    return results


def parse_relative_date(raw: str) -> datetime | None:
    text = raw.strip().lower()
    now = datetime.now(timezone.utc)

    if text in ("today", "just now", "moments ago", "a moment ago"):
        return now
    if text in ("yesterday",):
        return now - timedelta(days=1)
    if text in ("an hour ago", "a hour ago"):
        return now - timedelta(hours=1)
    if text in ("a day ago",):
        return now - timedelta(days=1)
    if text in ("a week ago",):
        return now - timedelta(weeks=1)
    if text in ("a month ago",):
        return now - timedelta(seconds=RELATIVE_UNIT_SECONDS["month"])
    if text in ("a year ago",):
        return now - timedelta(seconds=RELATIVE_UNIT_SECONDS["year"])

    match = RELATIVE_AGO_RE.match(text)
    if not match:
        return None

    amount = int(match.group(1))
    unit = match.group(2).lower()
    return now - timedelta(seconds=amount * RELATIVE_UNIT_SECONDS[unit])


def parse_date(raw: str, url: str, logger: logging.Logger) -> str | None:
    raw = raw.strip()

    match = re.match(r"^(\w+)\s+(\d{1,2}),\s+(\d{4})$", raw, re.IGNORECASE)
    if match:
        month_name, day, year = match.group(1).lower(), int(match.group(2)), int(match.group(3))
        month = MONTH_MAP.get(month_name)
        if month is None:
            logger.warning("Unknown month name '%s' in date '%s' on %s.", month_name, raw, url)
            return None
        try:
            return datetime(year, month, day, tzinfo=timezone.utc).isoformat()
        except ValueError as exc:
            logger.warning("Invalid date values in '%s' on %s — %s.", raw, url, exc)
            return None

    relative = parse_relative_date(raw)
    if relative is not None:
        return relative.isoformat()

    logger.warning("Could not parse date '%s' on %s — unrecognized format (absolute or relative).", raw, url)
    return None


def parse_file_size(soup: BeautifulSoup) -> str | None:
    for li in soup.find_all("li"):
        strong = li.find("strong")
        if strong and "game size" in strong.get_text(strip=True).lower():
            size_text = li.get_text(separator=" ", strip=True)
            size_value = re.sub(r"(?i)game\s*size\s*:?\s*", "", size_text).strip()
            return size_value or None
    return None


def parse_download_uris(soup: BeautifulSoup) -> list[str]:
    anchors = soup.find_all(
        "a",
        class_=re.compile(r"shortc-button"),
        string=re.compile(r"download\s+here", re.IGNORECASE),
    )
    if not anchors:
        anchors = [
            a for a in soup.find_all("a")
            if re.search(r"download\s+here", a.get_text(strip=True), re.IGNORECASE)
        ]
    return [normalise_download_url(a.get("href", "")) for a in anchors if a.get("href", "").strip()]


def parse_detail(html: str, page_url: str, logger: logging.Logger) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    result = {"uploadDate": None, "fileSize": None, "uris": []}

    date_span = soup.select_one("div.post-meta.clearfix span.date.meta-item.tie-icon")
    if date_span:
        result["uploadDate"] = parse_date(date_span.get_text(strip=True), page_url, logger)
    else:
        logger.warning(
            "Date element not found on %s — looked for: div.post-meta.clearfix > span.date.meta-item.tie-icon",
            page_url,
        )

    result["fileSize"] = parse_file_size(soup)
    if result["fileSize"] is None:
        logger.warning(
            "File size element not found on %s — looked for: <li><strong>Game Size: </strong>...</li>",
            page_url,
        )
        result["fileSize"] = "0 GB"

    result["uris"] = parse_download_uris(soup)
    if not result["uris"]:
        logger.warning(
            "No download URLs found on %s — looked for <a class='shortc-button ...'>DOWNLOAD HERE</a>",
            page_url,
        )

    return result


def scrape_item(item: dict, logger: logging.Logger) -> dict:
    logger.debug("Fetching item: %s (%s)", item["title"], item["url"])

    html = fetch(item["url"], logger)
    if html is None:
        return {"title": item["title"], "uploadDate": None, "fileSize": 0, "uris": []}

    detail = parse_detail(html, item["url"], logger)
    return {"title": item["title"], **detail}


def scrape_all(items: list[dict], logger: logging.Logger) -> list[dict]:
    downloads = []
    with logging_redirect_tqdm(loggers=[logger]):
        with tqdm(
            total=len(items),
            desc="Scraping items",
            unit="item",
            dynamic_ncols=True,
            leave=True,
        ) as progress_bar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for batch_start in range(0, len(items), BATCH_SIZE):
                    batch = items[batch_start:batch_start + BATCH_SIZE]
                    for result in executor.map(lambda item: scrape_item(item, logger), batch):
                        if result["uris"]:
                            downloads.append(result)
                        progress_bar.update(1)

                    if batch_start + BATCH_SIZE < len(items):
                        time.sleep(batch_delay())

    return downloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape a games listing site and dump download info to JSON.",
    )

    parser.add_argument("--impersonate", default=IMPERSONATE, help=f"Browser fingerprint to impersonate (default: {IMPERSONATE})")
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT, help=f"Per-request timeout in seconds (default: {REQUEST_TIMEOUT})")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help=f"Retry attempts per request (default: {MAX_RETRIES})")

    parser.add_argument("--max-workers", type=int, default=None, help=f"Concurrent request threads (default: {MAX_WORKERS}, {FAST_MAX_WORKERS} with --fast)")
    parser.add_argument("--batch-size", type=int, default=None, help=f"Items per batch (default: {BATCH_SIZE}, {FAST_BATCH_SIZE} with --fast)")
    parser.add_argument("--batch-delay-min", type=float, default=None, help=f"Minimum pause between batches, seconds (default: {BATCH_DELAY_RANGE[0]}, 0 with --fast)")
    parser.add_argument("--batch-delay-max", type=float, default=None, help=f"Maximum pause between batches, seconds (default: {BATCH_DELAY_RANGE[1]}, 0 with --fast)")
    parser.add_argument("--retry-delay", type=float, default=None, help=f"Delay between retries, seconds (default: {RETRY_DELAY}, 0 with --fast)")

    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip pauses between batches and retries, and raise worker/batch-size "
            "defaults, to scrape as fast as possible. Passing --max-workers, "
            "--batch-size, --batch-delay-min/max, or --retry-delay explicitly "
            "still overrides the --fast value for that parameter."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global MAX_WORKERS, BATCH_SIZE, BATCH_DELAY_RANGE
    global IMPERSONATE, REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY

    IMPERSONATE = args.impersonate
    REQUEST_TIMEOUT = args.timeout
    MAX_RETRIES = args.max_retries

    MAX_WORKERS = args.max_workers if args.max_workers is not None else (FAST_MAX_WORKERS if args.fast else MAX_WORKERS)
    BATCH_SIZE = args.batch_size if args.batch_size is not None else (FAST_BATCH_SIZE if args.fast else BATCH_SIZE)

    if args.fast and args.batch_delay_min is None and args.batch_delay_max is None:
        BATCH_DELAY_RANGE = (0.0, 0.0)
    else:
        delay_min = args.batch_delay_min if args.batch_delay_min is not None else BATCH_DELAY_RANGE[0]
        delay_max = args.batch_delay_max if args.batch_delay_max is not None else BATCH_DELAY_RANGE[1]
        BATCH_DELAY_RANGE = (delay_min, delay_max)

    RETRY_DELAY = args.retry_delay if args.retry_delay is not None else (0.0 if args.fast else RETRY_DELAY)

    logger = build_logger(LOG_PATH)
    logger.info(
        "Config: base_url=%s max_workers=%d batch_size=%d batch_delay=%s retry_delay=%.2f fast=%s",
        BASE_URL, MAX_WORKERS, BATCH_SIZE, BATCH_DELAY_RANGE, RETRY_DELAY, args.fast,
    )

    listing_url = BASE_URL.rstrip("/") + LIST_SUFFIX
    logger.info("Fetching listing page: %s", listing_url)

    listing_html = fetch(listing_url, logger)
    if listing_html is None:
        logger.error("Cannot fetch listing page. Aborting.")
        sys.exit(1)

    items = parse_listing(listing_html, BASE_URL, logger)
    if not items:
        logger.error("No items found on listing page. Aborting.")
        sys.exit(1)

    downloads = scrape_all(items, logger)

    output = {"name": SITE_NAME, "downloads": downloads}
    Path(OUTPUT_PATH).write_text(json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8")

    logger.info("Done. %d items scraped → %s | Warnings log → %s", len(downloads), OUTPUT_PATH, LOG_PATH)


if __name__ == "__main__":
    main()
