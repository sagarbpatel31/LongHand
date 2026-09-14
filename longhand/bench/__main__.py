"""`python -m longhand.bench` — naive chop vs Longhand on a synthetic session."""

from __future__ import annotations

import asyncio

from .fixtures import demo_fixture
from .harness import compare, format_table


def main() -> None:
    fixture = demo_fixture()
    results = asyncio.run(compare(fixture, chop_s=110.0))
    print(format_table(results, fixture))


if __name__ == "__main__":
    main()
