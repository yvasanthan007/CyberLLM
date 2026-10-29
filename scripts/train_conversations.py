"""
CyberLLM conversational training pipeline (one command).

Stages (each fails loudly, never silently skipped):
  1. Hardware & dependency check
  2. Dataset discovery
  3. Synthetic generation (if missing, when --generate enabled)
  4. Validation / preprocessing / dedupe
  5. Train/validation/test split
  6. Index build for the conversational engine (TF-IDF vectors cached)
  7. Evaluation report

NOTE on "training": this project uses CPU-only PyTorch (verified in
AUDIT_REPORT.md). Full LLM fine-tuning (LoRA/QLoRA) is NOT feasible here.
This pipeline therefore builds a retrieval-based conversational model
(RAG hybrid) which IS honest about what it is. If a CUDA-enabled torch
build is installed later, the LoRA path in train_conversations_lora.py
becomes available; this script will detect and report that.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "artifacts" / "reports" / "conversation"
CONV_DIR = ROOT / "data" / "conversation"


def log(stage, message):
    print(f"[{stage}] {message}", flush=True)


def run_step(name, command):
    """Run a subprocess step; raise on failure with captured output."""
    log("PIPELINE", f"Running: {name}")
    result = subprocess.run(
        command, shell=False, capture_output=True, text=True, cwd=str(ROOT)
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Step '{name}' failed (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def stage_hardware():
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip synthetic data generation")
    parser.add_argument("--num-per-category", type=int, default=4)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0",
        "stages": {},
    }

    # ------------------------------------------------ Stage 1: hardware
    log("HARDWARE", "Inspecting environment...")
    hw = stage_hardware()
    report["stages"]["hardware"] = hw
    log("HARDWARE", json.dumps(hw, default=str))
    if not hw.get("cuda_available"):
        log("HARDWARE",
            "CUDA unavailable -> using retrieval-based conversational model "
            "(RAG hybrid). LLM fine-tuning skipped honestly, not silently.")

    # ------------------------------------------------ Stage 2: dataset discovery
    log("DATASET", "Discovering conversational datasets...")
    raw_files = sorted(CONV_DIR.glob("*.jsonl"))
    raw_files = [f for f in raw_files
                 if "processed" not in f.parts and "splits" not in f.parts]
    if not raw_files and args.skip_generation:
        raise RuntimeError("No conversational datasets found and "
                           "--skip-generation was passed.")
    report["stages"]["dataset_discovery"] = {
        "files": [str(f.relative_to(ROOT)) for f in raw_files],
    }

    # ------------------------------------------------ Stage 3: generation
    if raw_files and not args.skip_generation:
        log("GENERATE", "Synthetic dataset exists; regenerating is skipped "
            "(pass --skip-generation=false behavior or delete the raw file to force).")
    elif not raw_files:
        log("GENERATE", "No raw dataset found. Generating synthetic conversations...")
        run_step("generate_synthetic", [
            sys.executable, str(ROOT / "scripts" / "generate_conversations.py"),
            "--num-per-category", str(args.num_per_category),
        ])
        report["stages"]["generation"] = {"ran": True,
            "output": str((CONV_DIR / "raw_synthetic_conversations.jsonl").relative_to(ROOT))}
    else:
        report["stages"]["generation"] = {"ran": False, "reason": "already_exists"}

    # ------------------------------------------------ Stage 4: preprocess
    log("PREPROCESS", "Validating, deduplicating, and splitting...")
    preprocess_out = run_step("preprocess", [
        sys.executable, str(ROOT / "scripts" / "preprocess_conversations.py"),
    ])
    report["stages"]["preprocess"] = {"output": preprocess_out.strip()}

    # ------------------------------------------------ Stage 5: build index
    log("INDEX", "Building conversational retrieval index...")
    from cyberllm.conversation.engine import get_engine
    engine = get_engine()
    index_stats = {
        "records_indexed": len(engine.records),
        "kb_documents": len(engine.kb_documents),
        "vocabulary_size": len(engine.idf),
    }
    report["stages"]["index"] = index_stats
    log("INDEX", json.dumps(index_stats))

    if index_stats["records_indexed"] == 0:
        raise RuntimeError("Conversational index has 0 records; "
                           "preprocessing produced nothing usable.")

    # ------------------------------------------------ Stage 6: smoke test
    log("EVAL", "Running smoke-test questions...")
    smoke_tests = [
        "What is ethical hacking?",
        "What is phishing?",
        "What is ransomware?",
        "What is multi-factor authentication?",
        "What is TLS?",
        "How do I secure SSH?",
    ]
    eval_results = []
    for question in smoke_tests:
        answer, meta = engine.respond(question)
        eval_results.append({
            "question": question,
            "source": meta.get("source"),
            "confidence": meta.get("confidence"),
            "answer_preview": answer[:150],
            "nonempty_answer": bool(answer.strip()),
            "answered_from_knowledge": meta.get("source") != "fallback",
        })
    passed = sum(1 for r in eval_results if r["answered_from_knowledge"])
    report["stages"]["smoke_evaluation"] = {
        "questions": eval_results,
        "passed": passed,
        "total": len(eval_results),
    }
    log("EVAL", f"Smoke eval: {passed}/{len(eval_results)} answered from trained/retrieved knowledge")

    # ------------------------------------------------ Save report
    report["duration_seconds"] = round(time.time() - started, 2)
    report_path = REPORT_DIR / f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    log("REPORT", f"Saved: {report_path.relative_to(ROOT)}")

    # Latest pointer
    latest = REPORT_DIR / "training_report_latest.json"
    shutil.copy(report_path, latest)
    log("DONE", "Pipeline complete.")


if __name__ == "__main__":
    main()
