from __future__ import annotations

import io
import json
from dataclasses import dataclass

from minio import Minio

from services.ingestion.storage import StoredObject


class MinIOStorage:
    """Storage backend using MinIO / S3-compatible object storage.

    Implements the same interface as LocalStorage so callers can switch
    backends via configuration without code changes.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> None:
        self._client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )
        self._bucket = bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def _key(self, job_id: str, name: str) -> str:
        return f"{job_id}/{name}"

    def _uri(self, key: str) -> str:
        return f"minio://{self._bucket}/{key}"

    def _parse_uri(self, uri: str) -> tuple[str, str]:
        if not uri.startswith("minio://"):
            raise ValueError(f"unsupported uri scheme: {uri}")
        rest = uri[len("minio://"):]
        bucket, _, key = rest.partition("/")
        return bucket, key

    def put_bytes(self, *, job_id: str, blob: bytes) -> StoredObject:
        key = self._key(job_id, "input.bin")
        self._client.put_object(
            self._bucket, key, io.BytesIO(blob), length=len(blob)
        )
        return StoredObject(uri=self._uri(key))

    def get_bytes(self, *, uri: str) -> bytes:
        bucket, key = self._parse_uri(uri)
        response = self._client.get_object(bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def put_json_atomic(
        self, *, job_id: str, obj: dict, name: str
    ) -> StoredObject:
        key = self._key(job_id, name)
        data = json.dumps(obj, indent=2).encode("utf-8")
        self._client.put_object(
            self._bucket,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type="application/json",
        )
        return StoredObject(uri=self._uri(key))

    def get_json_if_exists(self, *, job_id: str, name: str) -> dict | None:
        key = self._key(job_id, name)
        try:
            response = self._client.get_object(self._bucket, key)
            try:
                return json.loads(response.read())
            finally:
                response.close()
                response.release_conn()
        except Exception:
            return None

    def get_presigned_url(self, *, uri: str, expiry_s: int = 3600) -> str:
        from datetime import timedelta

        bucket, key = self._parse_uri(uri)
        return self._client.presigned_get_object(
            bucket, key, expires=timedelta(seconds=expiry_s)
        )
