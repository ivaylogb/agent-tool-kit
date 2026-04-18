"""In-memory order database for the e-commerce example.

Order IDs and shapes mirror agent-eval-loop's customer_support/scenarios.yaml,
so the toolkit-built handlers can be dropped into that example by passing
``get_handlers()`` to ``ImprovementLoop(tool_handlers=...)``.

Order dates are computed relative to "today" so 30-day return-window logic
stays correct whenever the example runs.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

_TODAY = datetime.now(timezone.utc).date()


def _d(offset_days: int) -> str:
    return (_TODAY - timedelta(days=offset_days)).isoformat()


# Order ID → record. IDs match agent-eval-loop's customer support scenarios.
ORDERS: dict[str, dict[str, Any]] = {
    "ORD-78234": {
        "order_id": "ORD-78234",
        "items": ["Wireless Headphones"],
        "status": "shipped",
        "order_date": _d(5),
        "total": 129.99,
        "payment_method_last4": "4242",
    },
    "ORD-45123": {
        "order_id": "ORD-45123",
        "items": ["Running Shoes"],
        "status": "delivered",
        "order_date": _d(10),
        "total": 89.00,
        "payment_method_last4": "1111",
    },
    "ORD-91001": {
        "order_id": "ORD-91001",
        "items": ["T-Shirt"],
        "status": "processing",
        "order_date": _d(1),
        "total": 25.00,
        "payment_method_last4": "5555",
    },
    "ORD-33201": {
        "order_id": "ORD-33201",
        "items": ["Coffee Maker"],
        "status": "shipped",
        "order_date": _d(3),
        "total": 79.50,
        "payment_method_last4": "9999",
    },
    "ORD-20100": {
        "order_id": "ORD-20100",
        "items": ["Desk Lamp"],
        "status": "delivered",
        "order_date": _d(45),
        "total": 59.00,
        "payment_method_last4": "2222",
    },
    "ORD-55400": {
        "order_id": "ORD-55400",
        "items": ["Laptop Stand", "External Monitor"],
        "status": "processing",
        "order_date": _d(7),
        "total": 800.00,
        "payment_method_last4": "7777",
    },
    "ORD-67890": {
        "order_id": "ORD-67890",
        "items": ["Backpack"],
        "status": "shipped",
        "order_date": _d(8),
        "total": 110.00,
        "payment_method_last4": "3333",
    },
    "ORD-11200": {
        "order_id": "ORD-11200",
        "items": ["Bluetooth Speaker"],
        "status": "delivered",
        "order_date": _d(14),
        "total": 65.00,
        "payment_method_last4": "8888",
    },
    "ORD-44100": {
        "order_id": "ORD-44100",
        "items": ["Phone Case"],
        "status": "shipped",
        "order_date": _d(2),
        "total": 24.99,
        "payment_method_last4": "6161",
    },
}


def days_since_order(order: dict[str, Any]) -> int:
    """How many days have passed since the order was placed."""
    return (_TODAY - date.fromisoformat(order["order_date"])).days


def tracking_payload(order: dict[str, Any]) -> dict[str, Any]:
    """Synthesize a plausible carrier tracking payload for a shipped/delivered order."""
    if order["status"] == "delivered":
        return {
            "carrier": "FedEx",
            "tracking_number": f"FDX{order['order_id'][-5:]}",
            "current_status": "delivered",
            "estimated_delivery": order["order_date"],
            "last_update": f"{order['order_date']}T18:00:00Z",
        }
    return {
        "carrier": "FedEx",
        "tracking_number": f"FDX{order['order_id'][-5:]}",
        "current_status": "in_transit",
        "estimated_delivery": _d(-2),
        "last_update": f"{_TODAY.isoformat()}T09:00:00Z",
    }
