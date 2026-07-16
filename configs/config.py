# -------------------------
# Project Root
# -------------------------
from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

NVD_API_KEY = os.getenv("NVD_API_KEY")

# -------------------------
# Raw Dataset
# -------------------------

CVE_DATASET = PROJECT_ROOT / "dataset" / "raw" / "cve"
CVE_DATASET.mkdir(parents=True, exist_ok=True)

MITRE_DATASET = PROJECT_ROOT / "dataset" / "raw" / "mitre"
MITRE_DATASET.mkdir(parents=True, exist_ok=True)

# -------------------------
# Processed Dataset
# -------------------------

PROCESSED_DATASET = PROJECT_ROOT / "dataset" / "processed"
PROCESSED_DATASET.mkdir(parents=True, exist_ok=True)

MITRE_OUTPUT = PROCESSED_DATASET / "mitre_clean.jsonl"

# -------------------------
# Instruction Dataset
# -------------------------

INSTRUCTION_DATASET = PROJECT_ROOT / "dataset" / "instruction"
INSTRUCTION_DATASET.mkdir(parents=True, exist_ok=True)

MITRE_INSTRUCTION = INSTRUCTION_DATASET / "mitre_instruction.jsonl"

# Final Dataset Directory
FINAL_DATASET = PROJECT_ROOT / "dataset" / "final"
FINAL_DATASET.mkdir(parents=True, exist_ok=True)

CYBERLLM_TRAIN = FINAL_DATASET / "cyberllm_train.jsonl"

# Instruction Dataset Directory
INSTRUCTION_DATASET = PROJECT_ROOT / "dataset" / "instruction"