"""Host-side, per-task recovery journal. Never stores HTTP credentials."""

import hashlib
import json
import os
import posixpath
import uuid
from pathlib import Path


class RecoveryJournal:
    def __init__(self, root):
        self.directory = Path(root) / uuid.uuid4().hex
        self.directory.mkdir(parents=True, mode=0o700)
        self.events = []

    def record(self, event):
        # Persist before making the event available for recovery.
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)

    def file(self, path, content):
        normalized = posixpath.normpath(path)
        if normalized.startswith(("/", "../")) or normalized in {".", ".."}:
            raise ValueError("Recovery requires a relative workspace path")
        digest = hashlib.sha256(content).hexdigest()
        blob = self.directory / digest
        if not blob.exists():
            with blob.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        self.record({"kind": "file", "path": normalized, "sha256": digest})

    def read(self, event):
        content = (self.directory / event["sha256"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != event["sha256"]:
            raise ValueError("Recovery backup checksum mismatch")
        return content


def result_digest(result):
    return hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
