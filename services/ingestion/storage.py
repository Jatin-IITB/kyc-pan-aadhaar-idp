from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname
@dataclass(frozen=True)
class StoredObject:
    uri: str

class Storage(Protocol):
    def put_bytes(self, *, job_id: str, blob: bytes) -> StoredObject: ...
    def get_bytes(self, *, uri: str) -> bytes: ...
    def put_json_atomic(self, *, job_id: str, obj: dict, name: str) -> StoredObject: ...
    def get_json_if_exists(self, *, job_id: str, name: str) -> dict | None: ...

class LocalStorage:
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def put_bytes(self, *, job_id: str, blob: bytes) -> StoredObject:
        p = self._job_dir(job_id) / "input.bin"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
        return StoredObject(uri=p.resolve().as_uri())

    def get_bytes(self, *, uri: str) -> bytes:
        u = urlparse(uri)
        if u.scheme != "file":
            raise ValueError(f"unsupported uri scheme: {u.scheme}")
        path =  url2pathname(unquote(u.path))
        if len(path) >= 3 and (path[0] in ("\\", "/")) and path[2] == ":":
            path = path[1:]
        if u.netloc:
            path = f"\\\\{u.netloc}{path}"    
        return Path(path).read_bytes()

    def put_json_atomic(self, *, job_id: str, obj: dict, name: str) -> StoredObject:
        out = self._job_dir(job_id) / name
        out.parent.mkdir(parents=True, exist_ok=True)

        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        tmp.replace(out)  # atomic on same filesystem

        return StoredObject(uri=out.resolve().as_uri())

    def get_json_if_exists(self, *, job_id: str, name: str) -> dict | None:
        p = self._job_dir(job_id) / name
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
