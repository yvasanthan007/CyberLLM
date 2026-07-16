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