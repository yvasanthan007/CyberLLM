"""Remove orphaned trailing lines left by a failed insert.

An interrupted ``insert`` can leave the tail of a function body stranded at
end-of-file. This tool detects trailing lines that are indented but appear
after a top-level statement, and truncates them. Run with ``--check`` to only
report.
"""
import ast
import sys
from pathlib import Path


def trailing_orphan(path: Path) -> tuple[int, str]:
    """Return (line_no, text) of the first orphaned tail line, or (0, '')."""
    lines = path.read_text(encoding="utf-8").splitlines()
    # Find the last top-level (column-0) statement start.
    last_top = -1
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and not line.startswith("#"):
            last_top = i
    for i in range(last_top + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line[0].isspace():
            last_top = i
        elif line.strip() and i > last_top + 1:
            return i + 1, line
    return 0, ""


def main() -> int:
    check_only = "--check" in sys.argv
    targets = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    bad = 0
    for path in targets:
        if not path.exists():
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            continue
        except SyntaxError:
            pass
        lineno, text = trailing_orphan(path)
        if not lineno:
            continue
        bad += 1
        print(f"{path}:{lineno}: orphaned -> {text!r}")
        if not check_only:
            lines = path.read_text(encoding="utf-8").splitlines()
            # Cut back to the last top-level line.
            keep = max(i for i, l in enumerate(lines[:lineno - 1])
                       if l and not l[0].isspace() and not l.startswith("#")) + 1
            path.write_text("\n".join(lines[:keep]).rstrip() + "\n", encoding="utf-8")
            print(f"  -> truncated to {keep} lines")
    return 1 if bad and check_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
