from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

from common import ARTIFACT_DIR


# Edit these values if you want different locations or a slower request rate.
OUTPUT_CSV = ARTIFACT_DIR / "repositories" / "fdroid_github_only.csv"
REQUEST_DELAY_SECONDS = 0.25

FDROID_BASE = "https://f-droid.org"
PACKAGE_PATH = re.compile(r"^/(?:[a-z]{2}/)?packages/[^/]+(?:/index\.html|/)?$")


def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "FineDroid repository discovery"})
    return session


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def canonicalize_package_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if path.endswith("/index.html"):
        path = path[: -len("index.html")]
    elif not path.endswith("/"):
        path += "/"
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


def get_category_links(session: requests.Session) -> list[str]:
    soup = get_soup(session, urljoin(FDROID_BASE, "/en/packages/"))
    content = soup.find("div", class_="post-content")
    if content is None:
        raise RuntimeError("The F-Droid package page has no post-content block")
    return sorted(
        {
            urljoin(FDROID_BASE, anchor["href"])
            for anchor in content.find_all("a", href=True)
            if anchor["href"].startswith("/en/categories/")
        }
    )


def package_links_on_page(soup: BeautifulSoup, page_url: str) -> set[str]:
    host = urlparse(FDROID_BASE).netloc
    links: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        candidate = urljoin(page_url, anchor["href"])
        parsed = urlparse(candidate)
        if parsed.netloc == host and PACKAGE_PATH.match(parsed.path):
            links.add(canonicalize_package_url(candidate))
    return links


def crawl_category(session: requests.Session, category_url: str) -> set[str]:
    base = category_url.rstrip("/") + "/"
    results: set[str] = set()
    page_number = 1
    while True:
        page_url = base if page_number == 1 else f"{base}{page_number}/index.html"
        try:
            soup = get_soup(session, page_url)
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code == 404:
                break
            raise
        page_results = package_links_on_page(soup, page_url)
        results.update(page_results)
        print(f"[F-Droid] {page_url}: {len(page_results)} packages")
        page_number += 1
        time.sleep(REQUEST_DELAY_SECONDS)
    return results


def extract_source_url(session: requests.Session, app_url: str) -> str | None:
    soup = get_soup(session, app_url)
    direct = soup.find(
        "a",
        string=lambda value: isinstance(value, str)
        and value.strip().lower() == "source code",
    )
    if direct and direct.get("href"):
        return urljoin(app_url, direct["href"].strip())
    for anchor in soup.find_all("a", href=True):
        if "github.com" in anchor["href"]:
            return urljoin(app_url, anchor["href"].strip())
    return None


def normalize_github_repository(url: str) -> str | None:
    parsed = urlsplit(url.strip())
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    repository = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    return urlunsplit(("https", "github.com", f"/{parts[0]}/{repository}", "", ""))


def main() -> None:
    session = build_session()
    apps: set[str] = set()
    categories = get_category_links(session)
    for index, category in enumerate(categories, start=1):
        print(f"[F-Droid] Category {index}/{len(categories)}: {category}")
        apps.update(crawl_category(session, category))

    repositories: set[str] = set()
    for index, app_url in enumerate(sorted(apps), start=1):
        try:
            repository = normalize_github_repository(extract_source_url(session, app_url) or "")
            if repository:
                repositories.add(repository)
        except requests.RequestException as error:
            print(f"[F-Droid] Failed to inspect {app_url}: {error}")
        if index % 100 == 0:
            print(f"[F-Droid] Inspected {index}/{len(apps)} package pages")
        time.sleep(REQUEST_DELAY_SECONDS)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_code_url"])
        writer.writerows([[url] for url in sorted(repositories)])
    print(f"Saved {len(repositories)} unique GitHub repositories to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
