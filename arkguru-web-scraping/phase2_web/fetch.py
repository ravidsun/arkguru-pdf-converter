"""HTTP fetch, robots.txt, same-site BFS, path denylist."""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

log = logging.getLogger("phase2.fetch")

USER_AGENT = "arkguru-phase2/1.0 (+https://github.com/ravidsun/arkguru-web-scraping)"

DEFAULT_PATH_DENY = (
    "/login", "/signup", "/sign-in", "/sign-up", "/cart", "/checkout",
    "/chat", "/call", "/account", "/auth", "/wp-admin", "/wp-login",
)

BINARY_SUFFIXES = (
    ".exe", ".zip", ".msi", ".dmg", ".pkg", ".gz", ".rar", ".7z",
    ".iso", ".deb", ".rpm", ".apk",
)

ASSET_SUFFIXES = BINARY_SUFFIXES + (
    ".css", ".js", ".mjs", ".map",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".avi", ".mov",
)

GENERATOR_QUERY_KEYS = frozenset({
    "dob", "dateofbirth", "birthdate", "birth_date", "latitude", "longitude",
    "lat", "lon", "boy", "girl", "name", "pname",
})


def normalize_url(url: str) -> str:
    url, _frag = urldefrag(url.strip())
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
        parsed = urlparse(url)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        # Keep www as-is on the wire; registrable matching strips it separately.
        pass
    return parsed._replace(path=path, netloc=netloc).geturl()


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().strip("[]")


def registrable_host(host: str) -> str:
    return host.lower().strip("[]").removeprefix("www.")


def same_site(url: str, seed_url: str) -> bool:
    h = registrable_host(host_of(url))
    s = registrable_host(host_of(seed_url))
    if not h or not s:
        return False
    return h == s or h.endswith("." + s)


def path_denied(url: str, deny: Iterable[str] = DEFAULT_PATH_DENY) -> bool:
    path = (urlparse(url).path or "/").lower()
    for prefix in deny:
        p = prefix.lower().rstrip("/") or "/"
        if path == p or path.startswith(p + "/"):
            return True
    return False


def is_binary_url(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    name = path.rsplit("/", 1)[-1]
    if "jhora" in name and not name.endswith(".pdf"):
        return True
    return any(path.endswith(suf) for suf in BINARY_SUFFIXES)


def is_pdf_url(url: str, content_type: str = "") -> bool:
    path = (urlparse(url).path or "").lower()
    if path.endswith(".pdf"):
        return True
    return "application/pdf" in (content_type or "").lower()


def is_kundli_generator(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.query:
        return False
    keys = {p.split("=", 1)[0].lower() for p in parsed.query.split("&") if p}
    return bool(keys & GENERATOR_QUERY_KEYS)


class RobotsCache:
    def __init__(self, user_agent: str = USER_AGENT, timeout: float = 10.0):
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, Optional[RobotFileParser]] = {}

    def allowed(self, url: str, fetch_bytes) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._parsers:
            robots_url = origin + "/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                body, _, _ = fetch_bytes(robots_url)
                rp.parse(body.decode("utf-8", "replace").splitlines())
            except Exception as e:
                log.info("robots.txt missing/unreadable for %s (%s); allow",
                         origin, e)
                rp.parse([])
            self._parsers[origin] = rp
        rp = self._parsers[origin]
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True


@dataclass
class FetchedPage:
    url: str
    status: int
    content_type: str
    body: bytes
    html: str = ""
    is_pdf: bool = False


@dataclass
class CrawlResult:
    pages: list[FetchedPage] = field(default_factory=list)
    pdf_urls: list[str] = field(default_factory=list)
    skipped: int = 0


def is_asset_url(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    return any(path.endswith(suf) for suf in ASSET_SUFFIXES)


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"


def resolve_href(href: str, base_url: str) -> Optional[str]:
    """Resolve a single href. Slash-containing relatives are site-root paths
    (vedicastrologer.org frameset menus live under ``/modules/``)."""
    href = href.strip()
    if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
        return None
    if href.startswith(("http://", "https://", "//", "/")):
        return urljoin(base_url, href)
    stripped = href.lstrip("./")
    if href.startswith("."):
        return urljoin(base_url, href)
    if "/" in stripped:
        return urljoin(_origin(base_url), stripped)
    return urljoin(base_url, href)


def extract_links(html: str, base_url: str) -> list[str]:
    """Collect navigable URLs. ``href`` only — ``src`` is images/scripts."""
    import re
    found: list[str] = []
    for m in re.finditer(r"""href\s*=\s*['"]([^'"]+)['"]""", html, re.I):
        resolved = resolve_href(m.group(1), base_url)
        if resolved:
            found.append(resolved)
    # JS-driven menus stash page paths in quoted strings.
    origin = _origin(base_url)
    for m in re.finditer(
            r"""['"]([A-Za-z0-9_./-]+\.(?:html?|pdf|php))['"]""", html, re.I):
        rel = m.group(1)
        found.append(urljoin(origin, rel.lstrip("./")))
    # de-dupe preserve order
    out, seen = [], set()
    for u in found:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fetch_url(client, url: str, timeout: float = 20.0) -> FetchedPage:
    r = client.get(url, follow_redirects=True, timeout=timeout)
    ctype = r.headers.get("content-type", "")
    page = FetchedPage(
        url=str(r.url) or url,
        status=r.status_code,
        content_type=ctype,
        body=bytes(r.content or b""),
        is_pdf=is_pdf_url(str(r.url) or url, ctype),
    )
    if page.is_pdf:
        return page
    head = page.body[:400].lower().lstrip()
    if "html" in ctype.lower() or head.startswith(b"<!doctype") or b"<html" in head:
        page.html = r.text if "html" in ctype.lower() else page.body.decode("utf-8", "replace")
    return page


def crawl(
    seeds: list[str],
    *,
    max_pages: int = 320,
    max_pages_per_seed: int = 80,
    delay_seconds: float = 0.75,
    same_domain_only: bool = True,
    path_deny: Iterable[str] = DEFAULT_PATH_DENY,
    timeout: float = 20.0,
    client=None,
) -> CrawlResult:
    """BFS per seed. HTML pages go to ``pages``; PDF URLs are listed, not fetched
    as HTML (``pdfs.download_pdfs`` pulls the bytes later)."""
    import httpx

    own_client = client is None
    if own_client:
        client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=timeout,
        )

    def fetch_bytes(url: str) -> tuple[bytes, int, str]:
        r = client.get(url, follow_redirects=True, timeout=timeout)
        return bytes(r.content or b""), r.status_code, r.headers.get("content-type", "")

    robots = RobotsCache()
    out = CrawlResult()
    seen: set[str] = set()

    try:
        for seed in seeds:
            seed = normalize_url(seed)
            per_seed = 0
            q: deque[str] = deque([seed])
            while q and len(out.pages) + len(out.pdf_urls) < max_pages and per_seed < max_pages_per_seed:
                url = normalize_url(q.popleft())
                if url in seen:
                    continue
                seen.add(url)
                if same_domain_only and not same_site(url, seed):
                    out.skipped += 1
                    continue
                if path_denied(url, path_deny) or is_kundli_generator(url) or is_binary_url(url) or is_asset_url(url):
                    out.skipped += 1
                    continue
                if not robots.allowed(url, fetch_bytes):
                    log.info("robots deny %s", url)
                    out.skipped += 1
                    continue
                if is_pdf_url(url):
                    if url not in out.pdf_urls:
                        out.pdf_urls.append(url)
                        per_seed += 1
                    continue
                if delay_seconds > 0 and (out.pages or out.pdf_urls or per_seed > 0):
                    time.sleep(delay_seconds)
                try:
                    page = fetch_url(client, url, timeout=timeout)
                except Exception as e:
                    log.warning("fetch failed %s: %s", url, e)
                    out.skipped += 1
                    per_seed += 1
                    continue
                # 404/5xx still consume the per-seed budget so broken sitemaps
                # cannot drain thousands of links after a handful of real pages.
                if page.status >= 400:
                    out.skipped += 1
                    per_seed += 1
                    continue
                if page.is_pdf:
                    if page.url not in out.pdf_urls:
                        out.pdf_urls.append(page.url)
                        per_seed += 1
                    continue
                if not page.html:
                    out.skipped += 1
                    per_seed += 1
                    continue
                out.pages.append(page)
                per_seed += 1
                for href in extract_links(page.html, page.url):
                    try:
                        n = normalize_url(href)
                    except Exception:
                        continue
                    if n not in seen:
                        q.append(n)
            log.info("seed %s -> %d pages this seed", seed, per_seed)
    finally:
        if own_client:
            client.close()

    log.info("crawl done: %d html, %d pdf urls, %d skipped",
             len(out.pages), len(out.pdf_urls), out.skipped)
    return out
