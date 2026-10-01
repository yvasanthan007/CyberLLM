"""Static sanity checks for the CyberLLM package.

These go beyond ``py_compile``: a file can parse yet still have a function
whose body is only a docstring, a bare ``pass`` hiding a stub, or an
``except`` that swallows everything. Each check below targets a defect class
that actually occurred while building this project, so CI fails loudly
instead of the symptom surfacing much later at runtime.

Run standalone::

    python scripts/static_check.py
    pytest tests/test_static_quality.py
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "cyberllm"

#: Name prefixes allowed to be a bare stub (protocol/abstract surface).
STUB_ALLOWED_PREFIXES = ("Base", "Abstract")


@dataclass
class Issue:
    path: str
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.code}] {self.message}"


def _func_body_is_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when a function body has no executable statement."""
    body = [
        stmt for stmt in node.body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str))
    ]
    if not body:
        return True
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        return True
    if len(body) == 1 and isinstance(body[0], ast.Raise):
        # `raise NotImplementedError` is a legitimate interface method.
        exc = body[0].exc
        if isinstance(exc, ast.Call):
            name = getattr(exc.func, "id", "") or getattr(exc.func, "attr", "")
            return name.endswith(("NotImplementedError", "Error"))
    return False


def _bare_except_returns(node: ast.ExceptHandler) -> bool:
    """True for a broad ``except Exception`` that silently returns/passes.

    Narrow handlers such as ``except ValueError: return False`` are normal
    guard clauses and are not flagged - only genuinely broad catches are,
    because those are the ones that hide real failures.
    """
    type_name = getattr(node.type, "id", getattr(node.type, "attr", ""))
    if type_name not in {"Exception", "BaseException"}:
        return False
    body = [
        s for s in node.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                and isinstance(s.value.value, str))
    ]
    if not body:
        return True
    return all(
        isinstance(s, (ast.Pass, ast.Return)) or
        (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        for s in body
    )


def _is_abstract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when a method is decorated ``@abstractmethod``."""
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if getattr(target, "id", getattr(target, "attr", "")) == "abstractmethod":
            return True
    return False


def check_file(path: Path) -> list[Issue]:
    """Return every issue found in one module."""
    issues: list[Issue] = []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [Issue(str(path), 0, "unreadable", str(exc))]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Issue(str(path), exc.lineno or 0, "syntax_error", exc.msg or "parse failure")]

    rel = path.name
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            if node.name.startswith(STUB_ALLOWED_PREFIXES) or _is_abstract(node):
                continue
            if _func_body_is_stub(node):
                issues.append(Issue(
                    rel, node.lineno, "stub_body",
                    f"function '{node.name}' has no implementation",
                ))
        elif isinstance(node, ast.ExceptHandler):
            if _bare_except_returns(node):
                name = getattr(node.type, "id", getattr(node.type, "attr", "?"))
                issues.append(Issue(
                    rel, node.lineno, "silent_except",
                    f"except {name} silently returns/passes - errors would be hidden",
                ))

    # A module that imports nothing at all is almost certainly dead code.
    has_import = any(
        isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)
    )
    has_definition = any(
        isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body
    )
    if has_definition and not has_import:
        issues.append(Issue(rel, 1, "no_imports", "module defines code but imports nothing"))
    return issues


def check_package(root: Path = PACKAGE_ROOT) -> list[Issue]:
    """Check every module in the package."""
    issues: list[Issue] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        issues.extend(check_file(path))
    return issues


def main(argv: Iterable[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    root = Path(args[0]) if args else PACKAGE_ROOT
    issues = check_package(root)
    for issue in issues:
        print(issue)
    print(f"\nstatic check: {len(issues)} issue(s) in {root}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
