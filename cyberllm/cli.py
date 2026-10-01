"""Command-line interface for CyberLLM.

Documented commands (all work as written)::

    python -m cyberllm.cli datasets list
    python -m cyberllm.cli datasets download [--name N] [--force] [--all]
    python -m cyberllm.cli datasets validate [--name N]
    python -m cyberllm.cli datasets preprocess [--name N]
    python -m cyberllm.cli models train --task nids|phishing --data PATH
    python -m cyberllm.cli models evaluate --task nids|phishing
    python -m cyberllm.cli rag ingest [--source NAME] [--rebuild]
    python -m cyberllm.cli rag search "query"
    python -m cyberllm.cli chat "question"
    python -m cyberllm.cli health
    python -m cyberllm.cli app [--api | --ui]

Every command degrades honestly: an unconfigured capability reports what to
configure instead of printing a fabricated result.
"""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from .core.config import get_settings
from .core.errors import CyberLLMError
from .core.logging_config import configure_logging, get_logger

console = Console()
logger = get_logger("cli")

#: Set by --json; consulted by every command's output helper.
_JSON = False

app = typer.Typer(
    name="cyberllm",
    help="CyberLLM - AI-powered cybersecurity platform (defensive use only).",
    no_args_is_help=True,
    add_completion=False,
)
datasets_app = typer.Typer(help="Dataset registry, acquisition and validation.")
models_app = typer.Typer(help="Train and evaluate detection models.")
rag_app = typer.Typer(help="Ingest, search and rebuild the knowledge base.")
app.add_typer(datasets_app, name="datasets")
app.add_typer(models_app, name="models")
app.add_typer(rag_app, name="rag")


def _print_json(payload: Any) -> None:
    console.print_json(json.dumps(payload, default=str))


def _wants_json() -> bool:
    return _JSON


def _fail(message: str, hint: str = "") -> None:
    """Print an actionable error and exit non-zero."""
    console.print(f"[bold red]Error:[/] {message}")
    if hint:
        console.print(f"[yellow]Hint:[/] {hint}")
    raise typer.Exit(code=1)


@app.callback()
def main_callback(
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG/INFO/WARNING/ERROR"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Configure logging once for every subcommand."""
    global _JSON
    configure_logging(log_level)
    _JSON = json_output


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@datasets_app.command("list")
def datasets_list(
    name: str = typer.Option(None, "--name", help="Show one dataset in detail."),
    kind: str = typer.Option(None, "--kind", help="Filter by kind."),
) -> None:
    """Show the dataset registry and its real on-disk status."""
    from .datasets.manager import DatasetManager

    manager = DatasetManager()
    if name:
        rows = [r for r in manager.inventory() if r["name"] == name]
        if not rows:
            _fail(f"Unknown dataset '{name}'.",
                  "Run `datasets list` with no arguments to see valid ids.")
            return
        if _wants_json():
            _print_json(rows[0])
            return
        row = rows[0]
        console.print(f"[bold]{row['name']}[/] - {row['kind']}")
        console.print(f"  status  : {row['status']}")
        console.print(f"  access  : {row['access']}")
        console.print(f"  license : {row['license']}")
        console.print(f"  source  : {row['source_url']}")
        console.print(f"  size    : {row['actual_size_mb']} MB on disk "
                      f"(~{row['approx_size_mb']} MB published)")
        if row["error"]:
            console.print(f"  [red]error[/]   : {row['error']}")
        if row["notes"]:
            console.print(f"  notes   : {row['notes']}")
        if row["manual_steps"]:
            console.print("\n[bold yellow]Manual steps required:[/]")
            for step in row["manual_steps"]:
                console.print(f"  {step}")
        return

    rows = manager.inventory(kind=kind)
    if _wants_json():
        _print_json(rows)
        return

    table = Table(title="CyberLLM dataset registry")
    for column in ("name", "kind", "status", "access", "on disk", "~size"):
        table.add_column(column)
    for row in rows:
        colour = {
            "downloaded": "green", "validated": "green", "preprocessed": "green",
            "manual": "yellow", "unavailable": "red", "synthetic": "magenta",
        }.get(row["status"], "white")
        table.add_row(row["name"], row["kind"], f"[{colour}]{row['status']}[/]",
                      row["access"], f"{row['actual_size_mb']} MB",
                      f"{row['approx_size_mb']:.0f} MB")
    console.print(table)
    console.print(f"Summary: {manager.manifest.summary()}")


@datasets_app.command("download")
def datasets_download(
    name: str = typer.Option(None, "--name", help="Dataset id to fetch."),
    force: bool = typer.Option(False, "--force", help="Re-download even if present."),
    all_datasets: bool = typer.Option(False, "--all", help="Fetch every automated one."),
) -> None:
    """Download datasets that permit automated access."""
    from .datasets.manager import DatasetManager

    manager = DatasetManager()
    if all_datasets:
        results = manager.download_all()
    elif name:
        try:
            results = [manager.download(name, force=force)]
        except CyberLLMError as exc:
            _fail(exc.message, f"Run `datasets list --name {name}`.")
            return
    else:
        _fail("Specify --name or --all.")
        return

    if _wants_json():
        _print_json([r.as_dict() for r in results])
        return
    for result in results:
        if result.ok:
            console.print(f"[green]OK[/]   {result.url} -> {result.bytes_downloaded:,} "
                          f"bytes ({result.elapsed_seconds:.1f}s)")
        else:
            console.print(f"[red]FAIL[/] {result.url or '(no url)'}: {result.error}")
    console.print("Manifest updated: data/dataset_manifest.csv")


@datasets_app.command("validate")
def datasets_validate(
    name: str = typer.Option(None, "--name", help="Validate one dataset."),
) -> None:
    """Check file signatures, schemas and label distributions."""
    from .datasets.manager import DatasetManager

    reports = DatasetManager().validate(name)
    if _wants_json():
        _print_json({k: v.as_dict() for k, v in reports.items()})
        return
    if not reports:
        console.print("[yellow]No dataset files found to validate.[/]")
        return
    for dataset, report in reports.items():
        mark = "[green]PASS[/]" if report.ok else "[red]FAIL[/]"
        console.print(f"{mark} {dataset}: {report.row_count:,} rows, "
                      f"{len(report.columns)} columns, format={report.detected_format}")
        for error in report.errors[:5]:
            console.print(f"       [red]x[/] {error}")
        for warning in report.warnings[:3]:
            console.print(f"       [yellow]![/] {warning}")


@datasets_app.command("preprocess")
def datasets_preprocess(
    name: str = typer.Option(None, "--name", help="Preprocess one dataset."),
) -> None:
    """Run dataset-specific preprocessing (threat-intel -> RAG documents)."""
    from .datasets.preprocess import DatasetPreprocessor

    results = DatasetPreprocessor().run(name)
    if _wants_json():
        _print_json(results)
        return
    if not results:
        console.print("[yellow]Nothing to preprocess.[/]")
        return
    for dataset, result in results.items():
        mark = "[green]OK[/]" if result.get("ok") else "[red]FAIL[/]"
        console.print(f"{mark} {dataset}: {result.get('detail', '')}")
        if result.get("hint"):
            console.print(f"       [yellow]hint[/] {result['hint']}")



# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
@models_app.command("train")
def models_train(
    task: str = typer.Option("nids", "--task", help="nids | phishing | malware"),
    data: str = typer.Option(None, "--data", help="Path to the training CSV."),
    algorithms: str = typer.Option(
        "random_forest,lightgbm", "--algorithms", help="Comma-separated list."),
    seed: int = typer.Option(42, "--seed", help="Random seed (recorded in the model card)."),
    test_size: float = typer.Option(0.2, "--test-size", help="Held-out fraction."),
    max_rows: int = typer.Option(60_000, "--max-rows", help="Cap rows for tractable runs."),
    allow_synthetic: bool = typer.Option(
        True, "--allow-synthetic/--no-synthetic",
        help="Generate a labelled synthetic fixture if the dataset is absent.",
    ),
) -> None:
    """Train a detection model and write a self-describing artefact bundle."""
    from pathlib import Path

    from .detection.service import train_nids
    from .phishing.service import train_phishing

    if task == "phishing":
        result = train_phishing(Path(data) if data else None, seed=seed,
                                test_size=test_size,
                                algorithms=tuple(a.strip() for a in algorithms.split(",")))
    elif task in {"nids", "network", "ids"}:
        if not data:
            _fail("--data is required for nids training.",
                  "Example: --data data/raw/unsw_nb15/UNSW_NB15_features.csv")
            return
        result = train_nids(
            Path(data),
            algorithms=tuple(a.strip() for a in algorithms.split(",")),
            seed=seed, test_size=test_size, max_rows=max_rows,
            make_fixture_if_missing=allow_synthetic,
        ).as_dict()
    else:
        _fail(f"Unknown task '{task}'.", "Choose nids or phishing.")
        return

    if _wants_json():
        _print_json(result)
        return
    if task == "phishing" and not result.get("trained"):
        console.print(f"[yellow]Not trained.[/] {result.get('reason', '')}")
        if result.get("setup_hint"):
            console.print(f"[yellow]Hint:[/] {result['setup_hint']}")
        return
    _render_training_report(result)


def _render_training_report(result: dict) -> None:
    """Print a training report, always surfacing the synthetic-data caveat."""
    if result.get("is_synthetic"):
        console.print("[bold magenta]SYNTHETIC DATA[/] - metrics below validate the "
                      "pipeline only and are NOT a real-world benchmark.\n")
    schema = result.get("schema") or {}
    if schema:
        console.print(f"Data      : {result.get('data_path')}")
        console.print(f"Rows      : {schema.get('rows', 0):,} x "
                      f"{len(schema.get('columns', []))} columns")
        console.print(f"Label col : {schema.get('label_column')}")
        console.print(f"Classes   : {len(schema.get('label_distribution', {}))}")
        console.print(f"Duplicates: {schema.get('duplicate_rows', 0):,}")
        if schema.get("warnings"):
            for warning in schema["warnings"][:3]:
                console.print(f"  [yellow]![/] {warning}")
        console.print()

    preprocessing = result.get("preprocessing") or {}
    if preprocessing.get("transformations"):
        console.print("[bold]Preprocessing:[/]")
        for step in preprocessing["transformations"]:
            console.print(f"  - {step}")
        if preprocessing.get("dropped_columns"):
            console.print(f"  - dropped as leakage: "
                          f"{', '.join(preprocessing['dropped_columns'])}")
        console.print()

    table = Table(title="Model comparison (same held-out split)")
    for column in ("algorithm", "accuracy", "f1_weighted", "f1_macro", "fit (s)"):
        table.add_column(column)
    for model in result.get("models", []):
        metrics = model.get("metrics", {})
        table.add_row(
            model.get("algorithm", "?"),
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('f1_weighted', 0):.4f}",
            f"{metrics.get('f1_macro', 0):.4f}",
            f"{metrics.get('fit_seconds', 0):.1f}",
        )
    if result.get("models"):
        console.print(table)
        comparison = result.get("comparison") or {}
        if comparison.get("best_by_f1_macro"):
            console.print(f"Best by macro-F1: [bold]{comparison['best_by_f1_macro']}[/]"
                          f"  ({comparison.get('ranked_on', '')})")

    if result.get("artifacts"):
        console.print("\n[bold]Artefacts:[/]")
        for path in result["artifacts"]:
            console.print(f"  {path}")
    for error in result.get("errors", []):
        console.print(f"  [yellow]note[/] {error}")



@models_app.command("evaluate")
def models_evaluate(
    task: str = typer.Option("nids", "--task", help="nids | phishing"),
    model: str = typer.Option("random_forest", "--model", help="Algorithm directory."),
) -> None:
    """Print the saved model card: metrics, seed, split and limitations."""
    from pathlib import Path

    from .core.config import get_settings
    from .detection.service import evaluate_nids

    root = get_settings().paths.models_dir / ("phishing" if task == "phishing" else "nids")
    directory = Path(model) if Path(model).is_absolute() else root / model
    if not directory.exists():
        candidates = sorted(p.name for p in root.iterdir()
                            if (p / "model_card.json").exists()) if root.exists() else []
        _fail(f"No model bundle at {directory}.",
              f"Available: {candidates or 'none - run `models train` first'}.")
        return
    try:
        report = evaluate_nids(directory)
    except Exception as exc:
        _fail(f"Could not read the model card: {type(exc).__name__}: {exc}")
        return
    if _wants_json():
        _print_json(report)
        return
    if report.get("is_synthetic"):
        console.print("[bold magenta]SYNTHETIC[/] - these numbers validate the "
                      "pipeline and are not a real-world benchmark.\n")
    console.print(f"[bold]{report['model']}[/] ({report['algorithm']})")
    console.print(f"  dataset   : {report['dataset']}  (sha256 {report['dataset_sha256']})")
    console.print(f"  seed      : {report['seed']}")
    console.print(f"  split     : {report['split_strategy']}")
    console.print(f"  rows      : {report['rows']['train']:,} train / "
                  f"{report['rows']['test']:,} test")
    metrics = report.get("metrics", {})
    console.print(f"  accuracy  : {metrics.get('accuracy', 0):.4f}")
    console.print(f"  F1 (macro): {metrics.get('f1_macro', 0):.4f}")
    if metrics.get("top_features"):
        console.print("  top features: " + ", ".join(
            f["feature"] for f in metrics["top_features"][:8]))
    if report.get("limitations"):
        console.print("\n[bold yellow]Limitations:[/]")
        for limitation in report["limitations"]:
            console.print(f"  - {limitation}")


# ---------------------------------------------------------------------------
# rag
# ---------------------------------------------------------------------------
@rag_app.command("ingest")
def rag_ingest(
    source: str = typer.Option(None, "--source", help="Dataset id to index."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Wipe the store first."),
    limit: int = typer.Option(None, "--limit", help="Cap records per source."),
) -> None:
    """Index downloaded threat-intel datasets into the knowledge base."""
    from .datasets.manager import DatasetManager
    from .datasets.registry import DatasetKind
    from .rag.pipeline import RAGPipeline

    manager = DatasetManager()
    pipeline = RAGPipeline()
    specs = [s for s in manager.inventory() if s["kind"] == DatasetKind.THREAT_INTEL]
    if source:
        specs = [s for s in specs if s["name"] == source]
        if not specs:
            _fail(f"No threat-intel dataset named '{source}'.")

    if rebuild:
        console.print("[yellow]Rebuilding the knowledge base from scratch...[/]")
        pipeline.store.reset()

    indexed = 0
    for spec in specs:
        files = sorted(manager.dataset_dir(spec["name"]).glob("*.json"))
        if not files:
            console.print(f"[yellow]skip[/] {spec['name']}: no file on disk")
            continue
        outcome = pipeline.ingest(spec["name"], files[0], limit=limit)
        indexed += 1
        mark = "[green]OK[/]" if not outcome.errors else "[red]FAIL[/]"
        console.print(f"{mark} {spec['name']}: {outcome.documents_loaded} records -> "
                      f"{outcome.chunks_indexed} chunks ({outcome.elapsed_seconds:.1f}s)")
    if not indexed:
        console.print("[yellow]Nothing indexed. Download a dataset first:[/]")
        console.print("  python -m cyberllm.cli datasets download --all")
    else:
        console.print(f"\nKnowledge base: {pipeline.store.count()} chunks "
                      f"from {pipeline.store.sources()}")


@rag_app.command("search")
def rag_search(
    query: str = typer.Argument(..., help="Search text."),
    top_k: int = typer.Option(5, "--top-k", help="Number of results."),
) -> None:
    """Search the knowledge base and print sources with scores."""
    from .rag.pipeline import RAGPipeline

    results = RAGPipeline().search(query, top_k)
    if not results:
        console.print("[yellow]No matches.[/] Index data first with `rag ingest`.")
        return
    if _wants_json():
        _print_json([r.as_dict() for r in results])
        return
    for result in results:
        console.print(f"\n[bold]#{result.rank}  score={result.score:.3f}[/] "
                      f"{result.document.title}")
        console.print(f"    source: {result.document.source}  {result.document.url}")
        console.print(f"    [dim]{' '.join(result.document.text.split())[:280]}...[/]")


@rag_app.command("status")
def rag_status() -> None:
    """Report knowledge-base size, sources and embedding backend."""
    from .rag.pipeline import RAGPipeline

    status = RAGPipeline().status()
    if _wants_json():
        _print_json(status)
        return
    if not status.get("available"):
        console.print(f"[red]Vector store unavailable:[/] {status.get('error')}")
        return
    console.print(f"Documents : {status['documents']:,}")
    console.print(f"Sources   : {', '.join(status['sources']) or 'none'}")
    console.print(f"Backend   : {status['backend']}")
    console.print(f"Embedding : {status['embedding']}")
    if status.get("empty"):
        console.print("\n[yellow]The knowledge base is empty. "
                      "Run `rag ingest` after downloading datasets.[/]")



# ---------------------------------------------------------------------------
# chat / health / app
# ---------------------------------------------------------------------------
@app.command("chat")
def chat(
    question: str = typer.Argument(..., help="Your question."),
    top_k: int = typer.Option(5, "--top-k", help="Retrieved documents."),
    no_rag: bool = typer.Option(False, "--no-rag", help="Answer without retrieval."),
) -> None:
    """Ask a cybersecurity question. Answers are cited."""
    from .service.chat import ChatService

    try:
        turn = ChatService().ask(question, top_k=top_k, use_rag=not no_rag)
    except CyberLLMError as exc:
        hint = exc.details.get("setup_hint", "") if exc.details else ""
        _fail(exc.message, str(hint))
        return
    if _wants_json():
        _print_json(turn.as_dict())
        return
    if turn.is_demo:
        console.print("[bold yellow]DEMO MODE[/] - no language model is running; the "
                      "answer is assembled from retrieved sources by a template.\n")
    console.print(f"[bold]Q:[/] {turn.question}\n")
    console.print(turn.answer)
    if turn.warnings:
        console.print("\n[yellow]Notes:[/]")
        for warning in dict.fromkeys(turn.warnings):
            console.print(f"  - {warning}")


@app.command("health")
def health() -> None:
    """Report what is configured, what is ready, and what is not."""
    from . import __version__
    from .detection.inference import IntrusionDetector
    from .llm import probe
    from .phishing import PhishingDetector
    from .rag.pipeline import RAGPipeline
    from .service.chat import ChatService

    settings = get_settings()
    status = ChatService().status()
    directory = settings.paths.models_dir / "nids" / "random_forest"
    payload = {
        "version": __version__,
        "config": settings.redacted_summary(),
        "llm": status["llm"],
        "demo_mode": status["demo_mode"],
        "knowledge_base": status["knowledge_base"],
        "phishing": PhishingDetector().status(),
        "providers": probe(settings),
        "intrusion_detection": IntrusionDetector(directory).status(),
    }
    if _wants_json():
        _print_json(payload)
        return

    console.print(f"[bold]CyberLLM {payload['version']}[/]  ({payload['config']})\n")
    table = Table(title="Capability")
    for column in ("capability", "status", "detail"):
        table.add_column(column)
    llm, kb = status["llm"], status["knowledge_base"]
    table.add_row("chat", "demo" if status["demo_mode"] else "ready",
                  f"{llm.get('provider', '?')} / {llm.get('model', '?')}")
    table.add_row("knowledge base", "ready" if kb.get("documents") else "empty",
                  f"{kb.get('documents', 0):,} chunks, "
                  f"{len(kb.get('sources', []))} sources")
    nids = payload["intrusion_detection"]
    if nids.get("available"):
        detail = (f"{nids.get('name', '?')} ({nids.get('algorithm', '?')}), "
                  f"{len(nids.get('classes', []))} classes")
        if nids.get("is_synthetic"):
            detail += " [SYNTHETIC]"
    else:
        detail = str(nids.get("reason", "not configured"))
    table.add_row("intrusion detection",
                  "ready" if nids.get("available") else "not configured", detail)
    phishing = payload["phishing"]
    table.add_row("phishing", "ready",
                  f"{phishing.get('method')} - no network access")
    console.print(table)
    if status["limitations"]:
        console.print("\n[bold yellow]Known limitations:[/]")
        for limitation in status["limitations"]:
            console.print(f"  - {limitation}")


@app.command("app")
def run_app(
    api: bool = typer.Option(False, "--api", help="Run the FastAPI server only."),
    ui: bool = typer.Option(False, "--ui", help="Run the Streamlit console only."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="API port."),
    ui_port: int = typer.Option(8501, "--port-ui", help="Streamlit port."),
) -> None:
    """Launch the web application (API + Streamlit by default)."""
    import subprocess
    import sys
    import time

    settings = get_settings()
    if host not in {"127.0.0.1", "localhost"} and not settings.security.api_key:
        _fail(
            f"Refusing to bind to '{host}' without an API key.",
            "CyberLLM has no authentication and no TLS. Keep it on localhost, or "
            "set CYBERLLM_API_KEY and put it behind a TLS-terminating proxy.",
        )
        return

    processes = []
    if not ui:
        console.print(f"[green]API[/]  http://{host}:{port}/docs")
        processes.append(subprocess.Popen([
            sys.executable, "-m", "uvicorn", "cyberllm.api.app:app",
            "--host", host, "--port", str(port),
        ]))
    if not api:
        console.print(f"[green]UI[/]   http://{host}:{ui_port}")
        processes.append(subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "cyberllm/app/ui.py",
            "--server.port", str(ui_port), "--server.address", host,
        ]))
    if not processes:
        _fail("Nothing to launch.", "Pass --api and/or --ui.")
        return

    console.print("\nPress Ctrl+C to stop.\n")
    try:
        while any(p.poll() is None for p in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\nStopping...")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


def main() -> None:
    """Entry point for ``python -m cyberllm.cli``."""
    try:
        app()
    except CyberLLMError as exc:
        console.print(f"[bold red]Error:[/] {exc.message}")
        if exc.details.get("setup_hint"):
            console.print(f"[yellow]Hint:[/] {exc.details['setup_hint']}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

