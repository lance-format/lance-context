"""The example knowledge base.

A small support/KB corpus for a fictional "Acme Cloud" product, spanning four
topics (auth, billing, deployment, storage). Each record carries a stable
``external_id`` — that is the identifier the labeled query set
(``queries.jsonl``) points at, and the identifier the eval harness matches
retrieved records against.
"""

from __future__ import annotations

# (external_id, role, text)
Doc = tuple[str, str, str]

CORPUS: list[Doc] = [
    # --- Authentication -------------------------------------------------
    (
        "kb-auth-login",
        "assistant",
        "Sign in to Acme Cloud with your email and password. After three failed "
        "login attempts the account is locked for fifteen minutes.",
    ),
    (
        "kb-auth-oauth",
        "assistant",
        "Acme Cloud supports OAuth single sign-on with Google and Okta. An admin "
        "enables SSO under Settings, then users authenticate through the identity "
        "provider instead of a password.",
    ),
    (
        "kb-auth-token",
        "assistant",
        "Create an API token from the developer console to authenticate requests. "
        "Tokens expire after 90 days; rotate them before expiry to avoid 401 "
        "authentication errors.",
    ),
    # --- Billing --------------------------------------------------------
    (
        "kb-billing-invoice",
        "assistant",
        "Invoices are generated on the first of each month and emailed to the "
        "billing contact. Download past invoices as PDF from the Billing page.",
    ),
    (
        "kb-billing-refund",
        "assistant",
        "To request a refund, open a billing ticket within 30 days of the charge. "
        "Refunds are issued to the original payment method within five business "
        "days.",
    ),
    (
        "kb-billing-payment",
        "assistant",
        "Update your credit card or add a payment method on the Billing page. A "
        "failed payment retries for seven days before the subscription is "
        "suspended.",
    ),
    # --- Deployment -----------------------------------------------------
    (
        "kb-deploy-release",
        "assistant",
        "Deploy a new release by pushing to the main branch; the pipeline builds "
        "the image and rolls it out to production automatically.",
    ),
    (
        "kb-deploy-rollback",
        "assistant",
        "If a deployment introduces a regression, roll back to the previous "
        "release from the Deployments page. Rollbacks take effect in under a "
        "minute.",
    ),
    (
        "kb-deploy-canary",
        "assistant",
        "Enable canary deployments to roll a new release out to five percent of "
        "traffic first. If error rates stay healthy the rollout continues, "
        "otherwise it rolls back automatically.",
    ),
    # --- Storage --------------------------------------------------------
    (
        "kb-storage-upload",
        "assistant",
        "Upload files to a storage bucket with the CLI or the web console. "
        "Individual objects are limited to 5 GB on the standard plan.",
    ),
    (
        "kb-storage-lifecycle",
        "assistant",
        "Configure bucket lifecycle rules to move objects to cold storage or "
        "delete them after a retention period, reducing storage cost.",
    ),
    (
        "kb-storage-encryption",
        "assistant",
        "Objects in a storage bucket are encrypted at rest by default. Bring your "
        "own KMS key under bucket settings for customer-managed encryption.",
    ),
]
