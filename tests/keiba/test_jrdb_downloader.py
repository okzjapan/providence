"""Tests for JrdbDownloader."""

import io
import zipfile
from unittest.mock import patch

import httpx
import pytest

from providence.keiba.scraper.jrdb import JrdbDownloader


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


@pytest.fixture()
def downloader(tmp_path):
    dl = JrdbDownloader.__new__(JrdbDownloader)
    dl._user_id = "test"
    dl._password = "test"
    dl._base_url = "http://test.jrdb.com"
    dl._interval = 0
    dl._timeout = 5.0
    dl._log = __import__("structlog").get_logger().bind(component="test")
    dl._raw_dir = tmp_path / "raw"
    dl._extracted_dir = tmp_path / "extracted"
    return dl


class TestExtractZip:
    def test_extracts_files(self, tmp_path, downloader):
        content = b"hello world"
        data = _make_zip({"test.txt": content})
        dest = tmp_path / "out"
        dest.mkdir()
        paths = downloader._extract_zip(data, dest)
        assert len(paths) == 1
        assert paths[0].read_bytes() == content

    def test_skips_directories(self, tmp_path, downloader):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("dir/", "")
            zf.writestr("dir/file.txt", b"data")
        dest = tmp_path / "out"
        dest.mkdir()
        paths = downloader._extract_zip(buf.getvalue(), dest)
        assert len(paths) == 1
        assert paths[0].name == "file.txt"


class TestDownloadAndExtract:
    def test_successful_download(self, downloader, tmp_path):
        content = b"fixed-length-record-data"
        zip_data = _make_zip({"SED260101.txt": content})

        with patch.object(downloader, "_download", return_value=zip_data):
            result = downloader._download_and_extract(
                "http://test/file.zip", "sed", "SED_2026.zip"
            )
        assert result is not None
        extracted = list(result.glob("*.txt"))
        assert len(extracted) == 1
        assert extracted[0].read_bytes() == content

    def test_404_returns_none(self, downloader):
        resp = httpx.Response(404, request=httpx.Request("GET", "http://test/"))
        error = httpx.HTTPStatusError("not found", request=resp.request, response=resp)

        with patch.object(downloader, "_download", side_effect=error):
            result = downloader._download_and_extract(
                "http://test/missing.lzh", "sed", "SED_2099.lzh"
            )
        assert result is None


class TestReadExtractedFiles:
    def test_reads_sorted_files(self, downloader):
        d = downloader._extracted_dir / "sed"
        d.mkdir(parents=True)
        (d / "SED260101.txt").write_bytes(b"aaa")
        (d / "SED260102.txt").write_bytes(b"bbb")

        files = downloader.read_extracted_files("sed")
        assert len(files) == 2
        assert files[0] == ("SED260101.txt", b"aaa")
        assert files[1] == ("SED260102.txt", b"bbb")

    def test_empty_when_no_dir(self, downloader):
        files = downloader.read_extracted_files("nonexistent")
        assert files == []
