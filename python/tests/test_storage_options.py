"""Unit tests for generic storage_options handling.

These exercise the Python-side plumbing that prepares the storage_options
dict before handing it to the Rust/PyO3 layer. Backend-specific semantics
(S3, GCS, Azure) are validated by the underlying lance/object_store crates
and by the per-backend integration tests in this package.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python" / "python"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lance_context.api import _merge_storage_options  # noqa: E402


def test_none_returns_empty_dict() -> None:
    options = _merge_storage_options(
        None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
        region=None,
        endpoint_url=None,
        allow_http=False,
    )
    assert options == {}


def test_generic_storage_options_passes_through_unchanged() -> None:
    incoming = {
        "google_service_account_key": "sa-json",
        "azure_storage_account_key": "az-key",
        "custom_provider_token": "token",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        merged = _merge_storage_options(
            incoming,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            region=None,
            endpoint_url=None,
            allow_http=False,
        )

    assert merged == incoming
    assert merged is not incoming  # defensive copy, caller dict untouched


def test_aws_kwargs_emit_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="storage_options"):
        merged = _merge_storage_options(
            None,
            aws_access_key_id="AKIA...",
            aws_secret_access_key="secret",
            aws_session_token=None,
            region="us-west-2",
            endpoint_url=None,
            allow_http=False,
        )

    assert merged == {
        "aws_access_key_id": "AKIA...",
        "aws_secret_access_key": "secret",
        "aws_region": "us-west-2",
    }


def test_aws_kwargs_translate_to_correct_option_keys() -> None:
    with pytest.warns(DeprecationWarning):
        merged = _merge_storage_options(
            None,
            aws_access_key_id="id",
            aws_secret_access_key="secret",
            aws_session_token="token",
            region="us-east-1",
            endpoint_url="http://minio:9000",
            allow_http=True,
        )

    assert merged == {
        "aws_access_key_id": "id",
        "aws_secret_access_key": "secret",
        "aws_session_token": "token",
        "aws_region": "us-east-1",
        "aws_endpoint_url": "http://minio:9000",
        "aws_allow_http": True,
    }


def test_storage_options_takes_precedence_over_aws_kwargs() -> None:
    """When both are set, storage_options wins (documented contract)."""
    with pytest.warns(DeprecationWarning):
        merged = _merge_storage_options(
            {"aws_region": "eu-west-1"},
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            region="us-east-1",
            endpoint_url=None,
            allow_http=False,
        )

    assert merged["aws_region"] == "eu-west-1"


def test_allow_http_alone_triggers_deprecation() -> None:
    with pytest.warns(DeprecationWarning, match=re.escape("allow_http")):
        merged = _merge_storage_options(
            None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            region=None,
            endpoint_url=None,
            allow_http=True,
        )
    assert merged == {"aws_allow_http": True}


def test_mixing_gcs_storage_options_with_aws_kwargs_is_allowed() -> None:
    """A user migrating from AWS kwargs should still be able to add GCS keys."""
    with pytest.warns(DeprecationWarning):
        merged = _merge_storage_options(
            {"google_service_account_key": "sa-json"},
            aws_access_key_id="id",
            aws_secret_access_key="secret",
            aws_session_token=None,
            region=None,
            endpoint_url=None,
            allow_http=False,
        )

    assert merged["google_service_account_key"] == "sa-json"
    assert merged["aws_access_key_id"] == "id"


def test_gcs_uri_accepts_generic_storage_options_without_warning() -> None:
    """Smoke-test the API boundary: Context.__init__ must not emit a warning
    when only generic storage_options are used, even for gs:// URIs.

    We can't actually connect to GCS here, but we can confirm no spurious
    DeprecationWarning is emitted from our wrapper when no AWS kwargs are
    provided. Any downstream lance error is caught and ignored.
    """
    from lance_context.api import Context

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            Context.create(
                "gs://nonexistent-bucket-xyz-unit-test/ctx.lance",
                storage_options={
                    "google_service_account_key": '{"type":"service_account"}',
                    "google_skip_signature": "true",
                },
            )
        except Exception:
            # Expected: bucket/credentials are bogus. We only care that no
            # DeprecationWarning was emitted by our wrapper.
            pass

    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    our_deprecations = [w for w in deprecation if "storage_options" in str(w.message)]
    assert our_deprecations == []
