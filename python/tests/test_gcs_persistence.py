"""Opt-in GCS integration tests.

Skipped by default. To run locally against a real (or emulated) GCS:

    # Option A: real GCS
    export LANCE_CONTEXT_GCS_BUCKET=my-test-bucket
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
    uv run pytest python/tests/test_gcs_persistence.py -v

    # Option B: against fake-gcs-server or another emulator, point the
    # relevant storage_options at the emulator endpoint (e.g. via
    # `use_opendal=true`, `endpoint=http://...`, `allow_anonymous=true`).
    export LANCE_CONTEXT_GCS_BUCKET=test-bucket
    export LANCE_CONTEXT_GCS_ENDPOINT=http://127.0.0.1:4443
    uv run pytest python/tests/test_gcs_persistence.py -v

These tests intentionally do not bring up their own emulator in CI because
there is no pure-Python GCS emulator that is both (a) maintained on modern
Python and (b) fully compatible with the lance-io GCS backend. The S3
suite uses moto, which has no GCS counterpart of equivalent quality.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python" / "python"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lance_context.api import Context  # noqa: E402

lance = pytest.importorskip("lance")

GCS_BUCKET = os.environ.get("LANCE_CONTEXT_GCS_BUCKET")
GCS_ENDPOINT = os.environ.get("LANCE_CONTEXT_GCS_ENDPOINT")
GCS_SA_JSON = os.environ.get("LANCE_CONTEXT_GCS_SERVICE_ACCOUNT_KEY")
GCS_ADC = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

_has_gcs_config = bool(GCS_BUCKET) and (
    bool(GCS_SA_JSON) or bool(GCS_ADC) or bool(GCS_ENDPOINT)
)

pytestmark = pytest.mark.skipif(
    not _has_gcs_config,
    reason=(
        "Set LANCE_CONTEXT_GCS_BUCKET plus one of "
        "LANCE_CONTEXT_GCS_SERVICE_ACCOUNT_KEY / "
        "GOOGLE_APPLICATION_CREDENTIALS / "
        "LANCE_CONTEXT_GCS_ENDPOINT to run GCS integration tests."
    ),
)


def _gcs_storage_options() -> dict[str, str]:
    options: dict[str, str] = {}
    if GCS_SA_JSON is not None:
        options["google_service_account_key"] = GCS_SA_JSON
    if GCS_ADC is not None:
        options["google_application_credentials"] = GCS_ADC
    if GCS_ENDPOINT is not None:
        # Emulator path: OpenDAL backend supports a custom endpoint and
        # anonymous auth, which is how fake-gcs-server is typically driven.
        options["use_opendal"] = "true"
        options["endpoint"] = GCS_ENDPOINT
        options.setdefault("allow_anonymous", "true")
    return options


def test_gcs_round_trip_via_storage_options() -> None:
    """End-to-end: Context.create(gs://...) with generic storage_options."""
    assert GCS_BUCKET is not None
    key = f"contexts/{uuid.uuid4().hex}/context.lance"
    uri = f"gs://{GCS_BUCKET}/{key}"
    options = _gcs_storage_options()

    ctx = Context.create(uri, storage_options=options)

    ctx.add("user", "gcs-hello")
    ctx.add("assistant", "gcs-response")
    assert ctx.entries() == 2

    dataset = lance.dataset(uri, storage_options=options)
    rows = dataset.to_table().to_pylist()
    assert [row["text_payload"] for row in rows] == ["gcs-hello", "gcs-response"]
