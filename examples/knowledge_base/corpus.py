"""A small in-memory corpus for the knowledge-base example.

Each document is intentionally long enough that returning the full body would
waste meaningful context. The retrieval tool's job is to pre-digest these
into short fragments — that's the compression pattern this example
demonstrates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    tags: tuple[str, ...]
    body: str


CORPUS: tuple[Document, ...] = (
    Document(
        doc_id="kb-001",
        title="Returns and refunds policy",
        tags=("policy", "returns"),
        body=(
            "Items purchased from ShopFast can be returned within 30 days of delivery. "
            "Returns must be in their original packaging with all tags attached. "
            "Once we receive your return, refunds are processed within 5-7 business days "
            "to the original payment method. Items marked 'final sale' or 'gift card' "
            "are non-returnable. For defective items, contact support within 90 days "
            "of delivery — extended warranty terms apply. Returns initiated outside "
            "the 30-day window will be rejected automatically unless escalated by an agent."
        ),
    ),
    Document(
        doc_id="kb-002",
        title="Shipping and delivery times",
        tags=("policy", "shipping"),
        body=(
            "Standard shipping is free for orders over $50 and takes 3-5 business days. "
            "Expedited shipping (2-day) is available at checkout for $9.99. Overnight "
            "shipping is $24.99 and orders must be placed by 2pm local time to ship "
            "the same day. We ship via FedEx, UPS, and USPS — the carrier is selected "
            "automatically based on package size and destination. International "
            "shipping is not currently supported. Tracking numbers are emailed when "
            "your order ships and become active in the carrier's system within 24 hours."
        ),
    ),
    Document(
        doc_id="kb-003",
        title="Order cancellation policy",
        tags=("policy", "cancellations"),
        body=(
            "Orders can be cancelled at no charge while their status is 'pending' or "
            "'processing'. Once an order's status changes to 'shipped', it can no "
            "longer be cancelled — you must wait for delivery and then initiate a "
            "return. Refunds for cancelled orders are processed within 3-5 business "
            "days. Partial cancellations (cancelling individual items in a multi-item "
            "order) are not supported; you must cancel the entire order and place a "
            "new one."
        ),
    ),
    Document(
        doc_id="kb-004",
        title="Account security and password resets",
        tags=("account", "security"),
        body=(
            "If you've forgotten your password, click 'Forgot password' on the login "
            "page and we'll email a reset link. The link expires after 60 minutes. "
            "For account recovery without email access, contact support — we'll "
            "verify your identity using your most recent order details. We never "
            "ask for your password by email or phone. Two-factor authentication is "
            "available in account settings; we strongly recommend enabling it."
        ),
    ),
    Document(
        doc_id="kb-005",
        title="Loyalty program: ShopFast Rewards",
        tags=("loyalty", "policy"),
        body=(
            "ShopFast Rewards earns 1 point per dollar spent. 100 points = $1 in "
            "store credit. Points expire 12 months after they're earned. Members "
            "with Gold status (5,000+ points in a calendar year) earn 1.5 points "
            "per dollar and get free expedited shipping on every order. Points are "
            "credited to your account when an order ships, not at checkout. "
            "Returning an item subtracts the points originally earned for it."
        ),
    ),
    Document(
        doc_id="kb-006",
        title="Defective items and warranty claims",
        tags=("policy", "returns", "warranty"),
        body=(
            "If an item arrives defective or breaks within 90 days of delivery, "
            "open a warranty claim by contacting support. We do NOT process "
            "warranty claims through the standard returns flow — the agent "
            "must escalate so the warranty team can verify the defect. Manufacturer "
            "warranties beyond 90 days are honored when applicable; the agent "
            "should provide the manufacturer's contact information from the "
            "product detail page."
        ),
    ),
)


def all_documents() -> tuple[Document, ...]:
    return CORPUS


def by_id(doc_id: str) -> Document | None:
    for d in CORPUS:
        if d.doc_id == doc_id:
            return d
    return None
