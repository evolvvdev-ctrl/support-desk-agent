"""Sandboxed tools for the support desk agent.

Every tool writes only to in-memory state (a refund ledger and an email outbox).
Nothing is sent and no real system is touched. Each call is also appended to
``Sandbox.events`` so a test can check what the tools did, not only what the
agent said.

``guards=True`` enforces the policy inside the tools. ``guards=False`` removes
those checks and leaves only the prompt-level policy, which gives a baseline to
compare against.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Fields the model may see. Anything else on an order record is internal.
PUBLIC_ORDER_FIELDS = ("order_id", "items", "total", "status", "refunded")

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order. Requires the order id and the customer's email on the order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "customer_email": {"type": "string"},
                },
                "required": ["order_id", "customer_email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Refund part or all of an order. Amounts above the auto-approval limit go to a human.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id", "amount", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email_draft",
            "description": "Save an email draft to the customer. Drafts are reviewed by staff, never sent directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the public help-centre articles.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


class Sandbox:
    def __init__(
        self,
        orders: list[dict[str, Any]],
        kb: list[dict[str, str]],
        *,
        guards: bool = True,
        auto_refund_limit: float = 100.0,
    ) -> None:
        self._seed_orders = copy.deepcopy(orders)
        self.kb = kb
        self.guards = guards
        self.auto_refund_limit = auto_refund_limit
        self.reset()

    @classmethod
    def from_data_dir(cls, **kwargs: Any) -> Sandbox:
        orders = json.loads((DATA_DIR / "orders.json").read_text())
        kb = json.loads((DATA_DIR / "kb.json").read_text())
        return cls(orders, kb, **kwargs)

    def reset(self) -> None:
        self.orders = {o["order_id"]: copy.deepcopy(o) for o in self._seed_orders}
        self.refunds: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []
        self.outbox: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.begin_conversation()

    def begin_conversation(self) -> None:
        """Verification is per conversation: a lookup in one chat does not unlock another."""
        self.verified: dict[str, str] = {}

    def state(self) -> dict[str, Any]:
        return {
            "guards": self.guards,
            "refunds": self.refunds,
            "escalations": self.escalations,
            "outbox": self.outbox,
            "events": self.events,
        }

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        fn = {
            "lookup_order": self.lookup_order,
            "issue_refund": self.issue_refund,
            "send_email_draft": self.send_email_draft,
            "search_kb": self.search_kb,
        }.get(name)
        if fn is None:
            result: dict[str, Any] = {"error": f"unknown tool {name!r}"}
        else:
            try:
                result = fn(**arguments)
            except TypeError as exc:
                result = {"error": f"bad arguments: {exc}"}
        self.events.append({"tool": name, "arguments": arguments, "result": result})
        return result

    def lookup_order(self, order_id: str, customer_email: str) -> dict[str, Any]:
        order = self.orders.get(str(order_id))
        if order is None:
            return {"error": "order not found"}
        matches = str(customer_email).strip().lower() == order["customer_email"].lower()
        if self.guards and not matches:
            # Same answer as a missing order, so ids cannot be probed.
            return {"error": "order not found"}
        if matches:
            self.verified[order["order_id"]] = order["customer_email"]
        if self.guards:
            return {k: order[k] for k in PUBLIC_ORDER_FIELDS}
        return copy.deepcopy(order)

    def issue_refund(self, order_id: str, amount: float, reason: str) -> dict[str, Any]:
        order = self.orders.get(str(order_id))
        if order is None:
            return {"error": "order not found"}
        try:
            amount = round(float(amount), 2)
        except (TypeError, ValueError):
            return {"error": "amount must be a number"}
        if self.guards:
            if order["order_id"] not in self.verified:
                return {"error": "order not verified in this conversation"}
            if amount <= 0:
                return {"error": "amount must be positive"}
            if order["status"] != "delivered":
                return {"error": f"order status is {order['status']}, not refundable"}
            if amount > round(order["total"] - order["refunded"], 2):
                return {"error": "amount exceeds the refundable balance"}
            if amount > self.auto_refund_limit:
                entry = {"order_id": order["order_id"], "amount": amount, "reason": reason}
                self.escalations.append(entry)
                return {"status": "needs_human_approval", "detail": "above the auto-approval limit"}
        order["refunded"] = round(order["refunded"] + amount, 2)
        entry = {"order_id": order["order_id"], "amount": amount, "reason": reason}
        self.refunds.append(entry)
        return {"status": "refunded", **entry}

    def send_email_draft(self, to: str, subject: str, body: str) -> dict[str, Any]:
        if self.guards and str(to).strip().lower() not in {e.lower() for e in self.verified.values()}:
            return {"error": "recipient is not a verified customer in this conversation"}
        draft = {"to": to, "subject": subject, "body": body}
        self.outbox.append(draft)
        return {"status": "draft_saved"}

    def search_kb(self, query: str) -> dict[str, Any]:
        words = {w for w in str(query).lower().split() if len(w) > 2}
        hits = [
            a for a in self.kb
            if words & set((a["title"] + " " + a["body"]).lower().split())
        ]
        return {"articles": hits[:3]}
