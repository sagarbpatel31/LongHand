"""`python -m longhand.bench` — naive chop vs Longhand on a synthetic session."""

from __future__ import annotations

import asyncio

from .fixtures import demo_drift_fixture
from .harness import compare_drift, format_table


def main() -> None:
    drift = demo_drift_fixture()
    results = asyncio.run(compare_drift(drift, chop_s=110.0))
    print(format_table(results, drift.fixture))


if __name__ == "__main__":
    main()
