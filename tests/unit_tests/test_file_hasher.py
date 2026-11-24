"""FileHasher fetch error handling: mid-download failures must not abort the hashing phase."""

from __future__ import annotations

import unittest
from unittest import mock

import requests

from hydrusvideodeduplicator import dedup


class _FailingBodyResponse:
    def __init__(self, exc: Exception):
        self._exc = exc
        self.closed = False

    @property
    def content(self):
        raise self._exc

    def close(self):
        self.closed = True


class TestFileHasherFetch(unittest.TestCase):
    def _hasher(self, response):
        client = mock.MagicMock()
        client.client.get_file.return_value = response
        return dedup.FileHasher(client)

    def test_fetch_file_chunked_encoding_error_raises_and_closes(self):
        response = _FailingBodyResponse(requests.exceptions.ChunkedEncodingError("truncated"))
        hasher = self._hasher(response)
        with self.assertRaises(dedup.HydrusApiException) as ctx:
            hasher._fetch_file("a" * 64)
        self.assertIsInstance(ctx.exception.args[0], requests.exceptions.ChunkedEncodingError)
        self.assertTrue(response.closed)

    def test_fetch_file_connection_error_raises_and_closes(self):
        response = _FailingBodyResponse(requests.exceptions.ConnectionError("dropped"))
        hasher = self._hasher(response)
        with self.assertRaises(dedup.HydrusApiException):
            hasher._fetch_file("a" * 64)
        self.assertTrue(response.closed)

    def test_fetch_file_or_error_returns_exception_for_body_read_failure(self):
        response = _FailingBodyResponse(requests.exceptions.ChunkedEncodingError("truncated"))
        hasher = self._hasher(response)
        with mock.patch.object(dedup.time, "sleep"):
            result = hasher.fetch_file_or_error("a" * 64)
        self.assertIsInstance(result, dedup.HydrusApiException)
        self.assertTrue(response.closed)

    def test_fetch_file_success_reads_content_and_closes(self):
        response = mock.MagicMock()
        response.content = b"video-bytes"
        hasher = self._hasher(response)
        self.assertEqual(hasher._fetch_file("a" * 64), b"video-bytes")
        response.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
