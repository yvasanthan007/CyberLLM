"""
CyberLLM
Project Configuration
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

NVD_API_KEY = os.getenv("NVD_API_KEY")

# ==========================================================
# Dataset Directories
# ==========================================================

DATASET_DIR = PROJECT_ROOT / "dataset"

RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
INSTRUCTION_DIR = DATASET_DIR / "instruction"
FINAL_DIR = DATASET_DIR / "final"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INSTRUCTION_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Raw Dataset Paths
# ==========================================================

CVE_DATASET = RAW_DIR / "cve"
MITRE_DATASET = RAW_DIR / "mitre"
SIGMA_DATASET = RAW_DIR / "sigma"

CVE_DATASET.mkdir(parents=True, exist_ok=True)
MITRE_DATASET.mkdir(parents=True, exist_ok=True)
SIGMA_DATASET.mkdir(parents=True, exist_ok=True)

# ==========================================================
# CVE Paths
# ==========================================================

CVE_RAW = CVE_DATASET

CVE_PROCESSED = PROCESSED_DIR / "cve_clean.jsonl"
CVE_CLEAN = CVE_PROCESSED

CVE_INSTRUCTION = INSTRUCTION_DIR / "cve_instruction.jsonl"

# ==========================================================
# MITRE Paths
# ==========================================================

MITRE_RAW = MITRE_DATASET

MITRE_OUTPUT = PROCESSED_DIR / "mitre_clean.jsonl"
MITRE_CLEAN = MITRE_OUTPUT

MITRE_INSTRUCTION = INSTRUCTION_DIR / "mitre_instruction.jsonl"

# ==========================================================
# Sigma Paths
# ==========================================================

SIGMA_RAW = SIGMA_DATASET

SIGMA_PROCESSED = PROCESSED_DIR / "sigma_clean.jsonl"
SIGMA_CLEAN = SIGMA_PROCESSED

SIGMA_INSTRUCTION = INSTRUCTION_DIR / "sigma_instruction.jsonl"

# ==========================================================
# Final Dataset
# ==========================================================

CYBERLLM_TRAIN = FINAL_DIR / "cyberllm_train.jsonl"

# ==========================================================
# Models
# ==========================================================

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Training
# ==========================================================

TRAINING_DIR = PROJECT_ROOT / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Inference
# ==========================================================

INFERENCE_DIR = PROJECT_ROOT / "inference"
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# API
# ==========================================================

API_DIR = PROJECT_ROOT / "api"
API_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# RAG
# ==========================================================

RAG_DIR = PROJECT_ROOT / "rag"
RAG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Documentation
# ==========================================================

DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Tests
# ==========================================================

TESTS_DIR = PROJECT_ROOT / "tests"
TESTS_DIR.mkdir(parents=True, exist_ok=True)