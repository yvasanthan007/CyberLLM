from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env explicitly
load_dotenv(PROJECT_ROOT / ".env")

NVD_API_KEY = os.getenv("NVD_API_KEY")

print("Loaded API Key:", NVD_API_KEY)

CVE_DATASET = PROJECT_ROOT / "dataset" / "raw" / "cve"
CVE_DATASET.mkdir(parents=True, exist_ok=True)

# MITRE Dataset Path
MITRE_DATASET = PROJECT_ROOT / "dataset" / "raw" / "mitre"
MITRE_DATASET.mkdir(parents=True, exist_ok=True)

# Processed Dataset
PROCESSED_DATASET = PROJECT_ROOT / "dataset" / "processed"
PROCESSED_DATASET.mkdir(parents=True, exist_ok=True)

MITRE_OUTPUT = PROCESSED_DATASET / "mitre_clean.jsonl"

# Instruction Dataset
INSTRUCTION_DATASET = PROJECT_ROOT / "dataset" / "instruction"
INSTRUCTION_DATASET.mkdir(parents=True, exist_ok=True)

MITRE_INSTRUCTION = INSTRUCTION_DATASET / "mitre_instruction.jsonl"