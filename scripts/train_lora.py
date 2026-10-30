"""
CyberLLM LoRA fine-tuning script (Phase 4) — honest hardware gating.

Behavior:
  - Checks torch CUDA availability FIRST.
  - If CUDA is unavailable (verified baseline: torch is CPU-only on this
    machine), the script prints the exact blocker and exits with code 2.
    It NEVER pretends training happened and NEVER silently substitutes
    the retrieval pipeline.
  - If CUDA IS available, runs a LoRA SFT on the conversational dataset
    (data/conversation/splits/train.jsonl) with configurable subset size,
    saves adapter weights + tokenizer + config + fingerprint to
    artifacts/lora/<run_id>/.

Usage:
  python scripts/train_lora.py                 # gate check + plan
  python scripts/train_lora.py --smoke         # 32-example smoke run (CUDA only)
  python scripts/train_lora.py --full          # full configured run (CUDA only)
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN_PATH = ROOT / "data" / "conversation" / "splits" / "train.jsonl"
OUTPUT_DIR = ROOT / "artifacts" / "lora"

BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"


def cuda_report():
    try:
        import torch
    except ImportError:
        return {"torch_installed": False, "cuda_available": False,
                "blocker": "torch is not installed"}
    info = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    else:
        info["blocker"] = (
            f"torch build '{torch.__version__}' is CPU-only; the GPU cannot "
            "be used. Install a CUDA-enabled torch build to train "
            "(see TRAINING_REPORT.md).")
    return info


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def load_train_records(limit=None):
    records = []
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[:limit] if limit else records


def train(args, cuda_info):
    """Actual LoRA SFT. Only called when CUDA is available."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_smoke" if args.smoke else "_full")
    out_dir = OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    records = load_train_records(args.subset)
    formatted = [
        {"text": (
            f"<|im_start|>user\n{r['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{r['output']}<|im_end|>")}
        for r in records
    ]
    dataset = Dataset.from_list(formatted)

    def tokenize(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=512, padding=False)
        encoded["labels"] = [
            [-100 if tok == tokenizer.pad_token_id else tok for tok in seq]
            for seq in encoded["input_ids"]
        ]
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch" if not args.smoke else "no",
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()

    # Save adapter + tokenizer + provenance
    model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    meta = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha},
        "dataset": {
            "path": str(TRAIN_PATH.relative_to(ROOT)),
            "fingerprint_sha256_16": fingerprint(TRAIN_PATH),
            "examples_used": len(formatted),
            "smoke": args.smoke,
        },
        "training_args": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum, "lr": args.lr,
        },
        "hardware": cuda_info,
        "completed": True,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    (out_dir / "adapter_done.txt").write_text("completed", encoding="utf-8")
    print(f"[DONE] Adapter saved to {out_dir / 'adapter'}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="32-example smoke run")
    parser.add_argument("--full", action="store_true", help="full run")
    parser.add_argument("--subset", type=int, default=None,
                        help="override example count (else 32 smoke / all full)")
    parser.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    if args.smoke:
        args.subset = args.subset or 32
        args.epochs = args.epochs or 1
    elif args.full:
        args.epochs = args.epochs or 3

    cuda_info = cuda_report()
    print(json.dumps(cuda_info, indent=2, default=str))

    if not cuda_info["cuda_available"]:
        print("\n[BLOCKED] LoRA fine-tuning CANNOT run on this machine:")
        print(f"  Blocker: {cuda_info.get('blocker')}")
        print("  Nothing was trained. The retrieval engine remains the")
        print("  conversational path and every API response is labeled")
        print("  accordingly. Dataset preparation is unaffected:")
        print("    python scripts/preprocess_conversations.py")
        sys.exit(2)

    if not args.smoke and not args.full:
        print("\n[PLAN] CUDA is available. Re-run with --smoke first, then --full.")
        sys.exit(0)

    if not TRAIN_PATH.is_file() or TRAIN_PATH.stat().st_size == 0:
        print(f"[BLOCKED] No training data at {TRAIN_PATH}.")
        print("  Run: python scripts/preprocess_conversations.py")
        sys.exit(2)

    train(args, cuda_info)


if __name__ == "__main__":
    main()
