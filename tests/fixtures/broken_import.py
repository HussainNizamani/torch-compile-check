"""A target that raises while being imported, for the import-failure error."""

from __future__ import annotations

raise ValueError("this module refuses to import")
