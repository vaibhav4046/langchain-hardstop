"""An in-process stand-in for the HARDSTOP server: $10 per payment, $20 a window, one named recipient."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from langchain_hardstop.client import HardstopResult

RECIPIENT = "Cloud hosting"


class FakeVault:
    def __init__(self, per_payment: str = "10", per_window: str = "20") -> None:
        self.per_payment = Decimal(per_payment)
        self.per_window = Decimal(per_window)
        self.spent = Decimal(0)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_preview = False

    def _verdict(self, to: Any, amount: Any) -> tuple[bool, str]:
        units = Decimal(str(amount))
        if to != RECIPIENT:
            return False, "That recipient is not approved."
        if units > self.per_payment:
            return False, f"One payment may be at most ${self.per_payment:.2f}."
        if self.spent + units > self.per_window:
            return False, "That would go over the limit for this window."
        return True, "Inside every limit."

    async def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> HardstopResult:
        args = dict(arguments or {})
        self.calls.append((name, args))
        if name == "hardstop_status":
            return HardstopResult("status", {"spent_today_usd": f"{self.spent:.2f}"})
        if name == "hardstop_preview_payment":
            if self.fail_preview:
                raise ConnectionError("server gone")
            ok, why = self._verdict(args["to"], args["amount_usd"])
            return HardstopResult(why, {"would_pay": ok, "explanation": why})
        if name == "hardstop_pay":
            ok, why = self._verdict(args["to"], args["amount_usd"])
            if ok:
                self.spent += Decimal(str(args["amount_usd"]))
            return HardstopResult(
                f"{'Paid' if ok else 'Refused'}: {why}", {"paid": ok, "result": "paid" if ok else "refused"}
            )
        if name == "hardstop_request_more":
            return HardstopResult("Draft for the owner", {"draft": True})
        return HardstopResult(f"unknown tool {name}", {}, is_error=True)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]
