"""JRDB data downloader.

Standalone class (does NOT inherit BaseScraper) because:
- BaseScraper is async (httpx.AsyncClient + aiolimiter) for real-time scraping
- JRDB download is a batch process where sync httpx.Client is simpler and sufficient
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import httpx
import lhafile
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from providence.config import Settings

logger = structlog.get_logger()


class JrdbDownloader:
    """Downloads and extracts JRDB fixed-length data files."""

    def __init__(self, settings: Settings | None = None) -> None:
        if settings is None:
            from providence.config import get_settings
            settings = get_settings()
        self._user_id = settings.jrdb_user_id
        self._password = settings.jrdb_password
        self._base_url = settings.jrdb_base_url
        self._interval = settings.scrape_interval_sec
        self._timeout = settings.scrape_timeout_sec
        self._log = logger.bind(component="JrdbDownloader")

        self._raw_dir = Path("data/jrdb/raw")
        self._extracted_dir = Path("data/jrdb/extracted")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    def _download(self, url: str) -> bytes:
        with httpx.Client(
            auth=(self._user_id, self._password),
            timeout=httpx.Timeout(self._timeout),
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content

    def download_yearly(self, jrdb_dir: str, prefix: str, year: int) -> Path | None:
        """Download a yearly bulk archive (e.g. SED_2024.lzh).

        Args:
            jrdb_dir: Directory on JRDB server (e.g. "Sed", "Ks")
            prefix: Filename prefix (e.g. "SED", "KZA")
            year: 4-digit year

        Returns path to the extracted directory, or None if the file doesn't exist.
        """
        filename = f"{prefix}_{year}.lzh"
        url = f"{self._base_url}/member/data/{jrdb_dir}/{filename}"
        return self._download_and_extract(url, jrdb_dir.lower(), filename)

    def download_daily(self, jrdb_dir: str, prefix: str, date_str: str) -> Path | None:
        """Download a single-day file (e.g. KYI260404.lzh).

        Args:
            jrdb_dir: Directory on JRDB server (e.g. "Kyi", "Ks")
            prefix: Filename prefix (e.g. "KYI", "KSA")
            date_str: yyMMdd format

        Returns path to the extracted directory, or None if not found.
        """
        filename = f"{prefix}{date_str}.lzh"
        url = f"{self._base_url}/member/data/{jrdb_dir}/{filename}"
        return self._download_and_extract(url, jrdb_dir.lower(), filename)

    def _download_and_extract(self, url: str, file_type: str, filename: str) -> Path | None:
        raw_dir = self._raw_dir / file_type.lower()
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / filename

        extract_dir = self._extracted_dir / file_type.lower()
        extract_dir.mkdir(parents=True, exist_ok=True)

        self._log.info("downloading", url=url)
        try:
            data = self._download(url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (403, 404):
                self._log.warning("not_found", url=url, status=e.response.status_code)
                return None
            raise

        raw_path.write_bytes(data)
        time.sleep(self._interval)

        extracted_files = self._extract(data, extract_dir, filename)
        self._log.info("extracted", count=len(extracted_files), dest=str(extract_dir))
        return extract_dir

    def _extract(self, data: bytes, dest: Path, filename: str) -> list[Path]:
        if filename.endswith(".zip"):
            return self._extract_zip(data, dest)
        return self._extract_lzh(data, dest)

    @staticmethod
    def _extract_lzh(data: bytes, dest: Path) -> list[Path]:
        lzh = lhafile.Lhafile(io.BytesIO(data))
        paths: list[Path] = []
        for info in lzh.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue
            out_path = dest / Path(name).name
            out_path.write_bytes(lzh.read(name))
            paths.append(out_path)
        return paths

    @staticmethod
    def _extract_zip(data: bytes, dest: Path) -> list[Path]:
        paths: list[Path] = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                out_path = dest / Path(info.filename).name
                out_path.write_bytes(zf.read(info.filename))
                paths.append(out_path)
        return paths

    def read_extracted_files(self, file_type: str, pattern: str = "*.txt") -> list[tuple[str, bytes]]:
        """Read all extracted text files for a file type.

        Returns list of (filename, content_bytes) tuples.
        """
        extract_dir = self._extracted_dir / file_type.lower()
        if not extract_dir.exists():
            return []
        results = []
        for path in sorted(extract_dir.glob(pattern)):
            results.append((path.name, path.read_bytes()))
        return results
