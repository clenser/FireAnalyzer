"""48-hour analysis result cache keyed by the SHA-256 of the uploaded file.

* The key is the SHA-256 of the **original uploaded bytes** (the image file or
  the whole video file - never extracted frames), namespaced by kind
  (``image`` / ``video``).
* An entry lives exactly :data:`CACHE_TTL_SECONDS` (48 h) from the moment it is
  stored.  Expired entries are ignored and deleted when encountered.
* Entries are JSON files in one directory, written atomically, so every worker
  process shares them and they survive a restart.
* Corrupt, unreadable or foreign-version entries are deleted and treated as a
  miss.
* Every operation is best-effort: a cache failure is logged and never raised,
  so it can never break an analysis.
* Only the JSON analysis result is stored - no request headers, no settings,
  no API keys.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

__all__ = ["CACHE_TTL_SECONDS", "AnalysisCache", "sha256_hex"]

logger = logging.getLogger(__name__)

#: Exactly 30 minutes.
CACHE_TTL_SECONDS = 30 * 60

#: Bumped whenever the cached result structure changes, so results produced by
#: an older pipeline are never served after a deploy.
CACHE_FORMAT_VERSION = 2

_PREFERRED_DIR = Path("/var/lib/flame-analyzer/cache")
_KINDS = ("image", "video")


def sha256_hex(data: bytes) -> str:
    """SHA-256 of the original uploaded bytes, as lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def _usable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


class AnalysisCache:
    """File-backed 48-hour cache.  Never raises."""

    def __init__(self, directory: str | Path | None = None, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.directory: Path | None = None
        if not self.enabled:
            return
        candidates = (
            [Path(directory)]
            if directory
            else ([_PREFERRED_DIR] if os.name == "posix" else [])
            + [Path(tempfile.gettempdir()) / "flame-analyzer-cache"]
        )
        for candidate in candidates:
            if _usable_dir(candidate):
                self.directory = candidate
                break
        if self.directory is None:
            logger.warning("No writable cache directory; the analysis cache is disabled")
            self.enabled = False
        else:
            logger.info("Analysis cache directory: %s", self.directory)

    # ------------------------------------------------------------------
    def _path(self, kind: str, digest: str) -> Path | None:
        if not self.enabled or self.directory is None or kind not in _KINDS:
            return None
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            return None
        return self.directory / f"{kind}-{digest}.json"

    @staticmethod
    def _delete(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete cache entry %s", path.name)

    def get(self, kind: str, digest: str) -> dict[str, Any] | None:
        """Return the cached result, or ``None`` on a miss/expiry/corruption."""
        path = self._path(kind, digest)
        if path is None:
            return None
        try:
            if not path.is_file():
                return None
            with path.open("r", encoding="utf-8") as handle:
                entry = json.load(handle)
        except (OSError, ValueError):
            logger.warning("Corrupt cache entry %s; deleting it", path.name)
            self._delete(path)
            return None

        try:
            valid = (
                isinstance(entry, dict)
                and entry.get("version") == CACHE_FORMAT_VERSION
                and entry.get("kind") == kind
                and entry.get("sha256") == digest
                and isinstance(entry.get("expires_at"), (int, float))
                and isinstance(entry.get("result"), dict)
            )
        except Exception:  # noqa: BLE001 - defensive: any odd payload is corrupt
            valid = False
        if not valid:
            logger.warning("Invalid cache entry %s; deleting it", path.name)
            self._delete(path)
            return None

        if time.time() >= float(entry["expires_at"]):
            logger.info("Expired cache entry %s; deleting it", path.name)
            self._delete(path)
            return None

        logger.info("Cache hit (%s)", kind)
        return entry["result"]

    def set(self, kind: str, digest: str, result: dict[str, Any]) -> bool:
        """Store (or replace) a result; the 48-hour expiry restarts now."""
        path = self._path(kind, digest)
        if path is None or not isinstance(result, dict):
            return False
        stored_at = time.time()
        entry = {
            "version": CACHE_FORMAT_VERSION,
            "kind": kind,
            "sha256": digest,
            "stored_at": stored_at,
            "expires_at": stored_at + CACHE_TTL_SECONDS,
            "result": result,
        }
        tmp_name: str | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{kind}-", suffix=".tmp", dir=str(path.parent)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(entry, handle, separators=(",", ":"))
            os.replace(tmp_name, path)
            tmp_name = None
        except (OSError, TypeError, ValueError):
            logger.warning("Could not write cache entry for %s", kind)
            return False
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        self._purge_expired()
        return True

    def _purge_expired(self) -> None:
        """Delete expired/corrupt entries.  Cheap: only reads file mtimes."""
        if self.directory is None:
            return
        cutoff = time.time() - CACHE_TTL_SECONDS
        try:
            for item in self.directory.glob("*.json"):
                try:
                    if item.stat().st_mtime < cutoff:
                        item.unlink(missing_ok=True)
                except OSError:
                    continue
        except OSError:
            pass
