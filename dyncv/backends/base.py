"""What the orchestrator needs from a compute backend. A new backend (Modal, a VM, ...) implements
these methods and runs `python -m dyncv.session run ...` inside its container; nothing else in
dyncv changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

FINAL = ("complete", "error", "cancelled")


class Backend(Protocol):
    name: str

    def submit(self, queue: Path, only: list[str] | None, budget_h: float) -> str:
        """Start one session on the queue (resuming the queue's previous session). Returns its id."""

    def status(self, sid: str) -> str:
        """queued | running | complete | error | cancelled | unknown."""

    def logs(self, sid: str) -> str:
        """Session stdout so far ('' if the backend cannot show it yet)."""

    def fetch(self, sid: str, dst: Path) -> Path:
        """Copy the session's runs/remote (job workspaces) into dst. Returns dst."""

    def active(self, queue: Path) -> str | None:
        """Id of the queue's session that is still queued or running, if any."""

    def mark(self, sid: str, status: str | None = None, **info) -> None:
        """Remember a session's final status and facts such as fetched=True / no_output=True."""

    def failures(self, queue: Path) -> int:
        """Number of failed sessions at the end of the queue's history (0 after a complete one)."""

    def unfetched(self, queue: Path) -> str | None:
        """Latest ended session whose output was never downloaded."""
