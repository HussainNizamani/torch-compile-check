"""Report renderers: terminal, JSON, Markdown, and a regression test case.

PLAN.md "Reports": three outputs, one model behind them. The terminal report is
plain ANSI with no third-party dependency, the JSON report is the
CI-consumable artifact and the unit of cross-architecture comparison, and the
Markdown report is an issue draft the human reads, edits, and files.
"""

from __future__ import annotations
