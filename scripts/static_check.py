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


def _module_scope(tree: ast.Module) -> dict[str, ast.AST]:
    """Names bound at module level (imports, defs, classes, assignments)."""
    bound: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound[(alias.asname or alias.name).split(".")[0]] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    bound[alias.asname or alias.name] = node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound[node.target.id] = node
    return bound


def _function_locals(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Every name bound inside a function: args, assignments, comprehensions."""
    names: set[str] = set()

    # Parameters, including *args / **kwargs and posonly/kwonly forms.
    args = node.args
    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        for arg in group:
            names.add(arg.arg)
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            names.add(extra.arg)

    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            names.update(child.names)
    return names


def _builtins() -> set[str]:
    import builtins

    return set(dir(builtins))


def check_undefined_names(tree: ast.Module, path_name: str) -> list[Issue]:
    """Flag module-level names that are used but never imported or defined.

    A missing ``import numpy as np`` compiles perfectly and only fails at
    runtime, in the middle of a long training run - so it is worth catching
    statically.

    The check is deliberately *conservative*: its only goal is to find names
    that appear in a function body, are not Python builtins, are not module
    level bindings, are not bound anywhere in that function (parameters,
    assignments, comprehensions, lambdas, nested defs), and are not imported
    conditionally. Anything ambiguous is skipped rather than reported.
    """
    issues: list[Issue] = []
    allowed = _builtins() | set(_module_scope(tree)) | _conditional_bindings(tree)
    reported: set[str] = set()

    # Map every node to its parent so closure capture can be resolved by
    # walking *outward* to the enclosing function.
    parent: dict[ast.AST, ast.AST] = {}
    for outer in ast.walk(tree):
        for child in ast.iter_child_nodes(outer):
            parent[child] = outer

    # Walk every function in the file, including methods on classes - a
    # missing import inside a method fails just as hard as one in a function.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Names bound anywhere inside this function, nested scopes included.
        # Over-approximating only suppresses findings, which is the safe
        # direction for a lint that must not produce false alarms.
        local = _all_local_names(node)
        for child in ast.walk(node):
            if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Load):
                continue
            name = child.id
            if name in allowed or name in local or name in reported:
                continue
            # A closure may read a variable bound in an enclosing function.
            if _captured_from_enclosing(parent, child, name):
                continue
            reported.add(name)
            issues.append(Issue(
                path_name, child.lineno, "undefined_name",
                f"'{name}' is used but not imported or defined at module level",
            ))
    return issues


def _captured_from_enclosing(parent: dict[ast.AST, ast.AST], node: ast.AST,
                             name: str) -> bool:
    """True when ``name`` resolves to a binding in an enclosing function."""
    current: ast.AST | None = node
    while current is not None:
        current = parent.get(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if name in _all_local_names(current):
                return True
    return False


def _all_local_names(node: ast.AST) -> set[str]:
    """Every name bound anywhere within ``node``, nested scopes included.

    Over-approximating here only suppresses findings, which is the safe
    direction for a lint that must not produce false alarms.
    """
    names: set[str] = set()

    for current in ast.walk(node):
        args = getattr(current, "args", None)
        if isinstance(args, ast.arguments):
            # Covers regular functions, lambdas and methods alike.
            names.update(_names_bound_by(args))
        if isinstance(current, ast.Name) and isinstance(current.ctx, (ast.Store, ast.Del)):
            names.add(current.id)
        elif isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(current.name)
        elif isinstance(current, (ast.Import, ast.ImportFrom)):
            for alias in current.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(current, ast.ExceptHandler) and current.name:
            names.add(current.name)
        elif isinstance(current, (ast.Global, ast.Nonlocal)):
            names.update(current.names)
        elif isinstance(current, ast.alias):
            names.add((current.asname or current.name).split(".")[0])
    return names


def _names_bound_by(args: ast.arguments) -> set[str]:
    names = {a.arg for group in (args.posonlyargs, args.args, args.kwonlyargs)
             for a in group}
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            names.add(extra.arg)
    return names


def _conditional_bindings(tree: ast.Module) -> set[str]:
    """Names bound inside ``try``/``if``/``with`` blocks (optional imports)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Try, ast.If, ast.With)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                names.add(child.name)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    if alias.name != "*":
                        names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(child, ast.alias):
                names.add((child.asname or child.name).split(".")[0])
    return names


def check_line_continuations(source: str, path_name: str) -> list[Issue]:
    """Flag backslash line continuations.

    They are legal Python but are a reliable source of subtle breakage: an
    interrupted edit leaves a trailing ``\\`` with nothing valid after it, and
    ``py_compile`` can even accept a dangling one inside a string. The
    project's style uses parentheses instead, so treat them as errors.
    """
    issues: list[Issue] = []
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.rstrip()
        if not stripped.endswith("\\"):
            continue
        # `\\` at end of a non-raw string is an escaped backslash, not a
        # continuation; require whitespace-or-backslash preceded by a
        # non-string context, approximated by checking the quote balance.
        if stripped.count('"') % 2 == 1 or stripped.count("'") % 2 == 1:
            continue
        issues.append(Issue(
            path_name, number, "line_continuation",
            "trailing backslash continuation - use parentheses instead",
        ))
    return issues


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

    issues.extend(check_undefined_names(tree, rel))
    issues.extend(check_line_continuations(source, rel))
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
