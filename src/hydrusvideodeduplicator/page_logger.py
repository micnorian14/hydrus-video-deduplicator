from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from .client import HVDClient

import logging


def get_page_key(client: HVDClient, page_name: str) -> str | None:
    """Takes the provided page name, and searches through all the pages in the Hydrus client for an appropriate
    page with that name. If there are multiple pages with the same name, one of those pages is chosen randomly."""
    response = client.client.get_pages()
    page_key = find_page_key_from_name(response["pages"], page_name)
    return page_key


def find_page_key_from_name(page: dict[str, Any], page_name: str) -> str | None:
    """Recursive function to search the response JSON provided by the Hydrus API's get_pages call. Because every
    page can potentially contain other pages, a recursive search through the object is necessary. As soon as a
    page is found with the correct page name and page type, that page's page_key is returned."""
    if page["name"].lower() == page_name.lower() and page["page_type"] == 6:
        return page["page_key"]
    elif "pages" in page:
        for subpage in page["pages"]:
            result = find_page_key_from_name(subpage, page_name)
            if result is not None:
                return result
    return None


DEFAULT_FAILED_PAGE_NAME = "failed"


def resolve_failed_page_name(name: str | None) -> str:
    """'failed' unless a custom --failed-page-name (or equivalent) was given."""
    if name is None or not str(name).strip():
        return DEFAULT_FAILED_PAGE_NAME
    return str(name).strip()


class HydrusPageLogger:
    """Add failed files to a Hydrus page, if one with this name exists."""

    _log = logging.getLogger("HydrusPageLogger")
    _log.setLevel(logging.INFO)

    def __init__(self, client: HVDClient, page_name: str):
        self.client = client
        self.page_name = page_name
        self.sent = 0
        self.page_missing = False
        self._page_key: str | None = None
        self._resolved = False

    def _page_key_or_none(self) -> str | None:
        """Look up the page once. A missing page or lookup error is silent and is not retried."""
        if self._resolved:
            return self._page_key
        self._resolved = True
        try:
            self._page_key = get_page_key(self.client, self.page_name)
        except Exception as exc:
            self._log.debug(f"Could not look up Hydrus page '{self.page_name}': {exc}")
            return None
        if self._page_key is None:
            self.page_missing = True
            self._log.debug(f"No Hydrus page named '{self.page_name}'")
        return self._page_key

    def add_failed_video(self, video_hash: str) -> None:
        """Add a failed video to the Hydrus page. Does nothing if the page is missing."""
        page_key = self._page_key_or_none()
        if page_key is None:
            return
        try:
            self.client.client.add_files_to_page(page_key=page_key, hashes=[video_hash])
            self.sent += 1
        except Exception as exc:
            self._log.debug(
                f"Could not add file {video_hash} to Hydrus page '{self.page_name}': {exc}"
            )
