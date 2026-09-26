"""Verify that recorded episodes replay exactly.

    uv run python -m blinddrive.replay runs/*.jsonl

To *watch* a replay instead: ``python -m blinddrive.runner --replay FILE``.
"""

from __future__ import annotations

import sys

from .recorder import load_log, verify_log


def main(argv: list[str] | None = None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print("usage: python -m blinddrive.replay LOG.jsonl [...]")
        return 2
    failed = 0
    for path in paths:
        problems = verify_log(load_log(path))
        if problems:
            failed += 1
            print(f"MISMATCH {path}")
            for p in problems[:10]:
                print(f"   {p}")
        else:
            print(f"OK       {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
