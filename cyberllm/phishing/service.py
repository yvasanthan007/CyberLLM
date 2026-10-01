"""Training and inference for the phishing URL classifier.

Reuses the detection module's training/report machinery so that model cards,
metrics and artefacts are identical in shape across tasks - one reviewer can
learn one format.

The dataset is a licensed research CSV with a ``url`` and a ``label`` column
(1 = phishing, 0 = benign). It requires manual acquisition; see
``data/README.md`` for the exact steps. Until it is present, the platform
falls back to the explainable lexical heuristic and says so.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..core.config import get_settings
from ..core.errors import ValidationError
from ..core.logging_config import get_logger
from .features import FEATURE_NAMES, PhishingVerdict, heuristic_score, url_to_vector

logger = get_logger("phishing.service")

#: A probability above this is reported as phishing.
DEFAULT_THRESHOLD = 0.5

#: Label spellings seen across published phishing corpora.
_PHISHING_LABELS = {"1", "true", "phishing", "yes", "malicious"}
_BENIGN_LABELS = {"0", "false", "legitimate", "benign", "no", "normal"}


def find_dataset(settings: Any = None) -> Path | None:
    """Locate a phishing URL CSV if the operator has supplied one."""
    settings = settings or get_settings()
    root = settings.paths.raw_dir / "phishing_urls"
    if not root.exists():
        return None
    for name in ("urls.csv", "phishing_urls.csv", "PhiUSIIL.csv"):
        candidate = root / name
        if candidate.exists():
            return candidate
    csvs = sorted(root.glob("*.csv"))
    return csvs[0] if csvs else None


def load_url_dataset(path: Path) -> tuple[Any, list[int], int]:
    """Read a URL/label CSV into a feature matrix.

    Returns ``(X, y, dropped)``. Rows whose URL cannot be parsed are dropped
    and counted rather than failing the whole load - published corpora
    routinely contain a few malformed entries.
    """
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip().lower() for c in frame.columns]

    url_col = next((c for c in ("url", "urls", "uri") if c in frame.columns), None)
    label_col = next(
        (c for c in ("label", "labels", "class", "isphishing", "phishing", "type")
         if c in frame.columns), None
    )
    if url_col is None or label_col is None:
        raise ValidationError(
            f"{path.name} must contain a url column and a label column "
            f"(found: {', '.join(frame.columns)})."
        )

    vectors: list[list[float]] = []
    labels: list[int] = []
    dropped = 0
    for url, label in zip(frame[url_col], frame[label_col]):
        try:
            vector, _ = url_to_vector(str(url))
        except (ValidationError, ValueError):
            dropped += 1
            continue
        text = str(label).strip().lower()
        if text in _PHISHING_LABELS:
            vectors.append(vector)
            labels.append(1)
        elif text in _BENIGN_LABELS:
            vectors.append(vector)
            labels.append(0)

    if not vectors:
        raise ValidationError(
            f"No usable rows in {path.name} - check the label values "
            "(expected 0/1, phishing/benign, or true/false)."
        )
    logger.info("loaded %d URLs from %s (%d unparsable dropped)",
                len(vectors), path.name, dropped)
    return np.asarray(vectors, dtype="float32"), labels, dropped


def train_phishing(
    data_path: Path | None = None, *, seed: int = 42, test_size: float = 0.2,
    out_dir: Path | None = None, algorithms: tuple[str, ...] = ("random_forest",),
) -> dict[str, Any]:
    """Train the phishing classifier if a dataset is available."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    from ..detection.preprocess import PreparedData, file_sha256
    from ..detection.train import save_bundle, train_one

    settings = get_settings()
    path = Path(data_path) if data_path else find_dataset(settings)
    if path is None:
        return {
            "trained": False,
            "reason": (
                "No phishing URL dataset found. This dataset requires manual "
                "acquisition - see `datasets list --name phishing_urls` for the "
                "exact steps. The platform serves the lexical heuristic until a "
                "model is trained."
            ),
            "setup_hint": "Place a url,label CSV at data/raw/phishing_urls/urls.csv",
        }

    started = time.monotonic()
    out_root = Path(out_dir) if out_dir else settings.paths.models_dir / "phishing"
    X, y, dropped = load_url_dataset(path)
    y_arr = np.asarray(y)

    stratify = y_arr if min(np.bincount(y_arr)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=test_size, random_state=seed, stratify=stratify
    )

    data = PreparedData(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        feature_names=list(FEATURE_NAMES),
        label_encoder={"BENIGN": 0, "PHISHING": 1},
        transformations=[
            f"loaded {len(X):,} URLs from {path.name}",
            f"dropped {dropped:,} unparsable URL(s)",
            f"train/test split {1 - test_size:.0%}/{test_size:.0%} (seed={seed})"
            + (", stratified" if stratify is not None else ", unstratified"),
        ],
        class_distribution={"BENIGN": int((y_arr == 0).sum()),
                           "PHISHING": int((y_arr == 1).sum())},
        seed=seed, rows_in=len(X) + dropped, rows_out=len(X),
        test_size=test_size, stratify_applied=stratify is not None,
    )
    data.source_hash = file_sha256(path)

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for algorithm in algorithms:
        try:
            model, card = train_one(
                data, algorithm, seed, dataset=path.stem,
                source_hash=data.source_hash, is_synthetic=False,
            )
            card.task = "phishing URL classification (offline lexical features)"
            directory = out_root / algorithm
            save_bundle(model, card, directory)
            (directory / "model_card.json").write_text(
                json.dumps(card.as_dict(), indent=2), encoding="utf-8"
            )
            results.append({"algorithm": algorithm, "metrics": card.metrics,
                            "path": str(directory)})
        except Exception as exc:
            logger.error("phishing training failed for %s: %s", algorithm, exc)
            errors.append(f"{algorithm}: {type(exc).__name__}: {exc}")

    return {
        "trained": bool(results),
        "data_path": str(path),
        "models": results,
        "errors": errors,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }



class PhishingDetector:
    """Scores URLs, using a trained model when one exists."""

    def __init__(self, directory: Path | None = None,
                 threshold: float = DEFAULT_THRESHOLD) -> None:
        settings = get_settings()
        self.directory = (Path(directory) if directory
                          else settings.paths.models_dir / "phishing")
        self.threshold = threshold
        self._model: Any = None

    def _load(self) -> Any:
        """Load the first available trained bundle, if any."""
        if self._model is not None:
            return self._model
        from ..detection.train import load_bundle

        for algorithm in ("random_forest", "lightgbm"):
            candidate = self.directory / algorithm
            if (candidate / "model.pkl").exists():
                self._model = load_bundle(candidate)
                return self._model
        return None

    def analyse(self, url: str) -> PhishingVerdict:
        """Score a single URL. Never performs any network access."""
        from .features import analyse_heuristically

        vector, signals = url_to_vector(url)
        model = self._load()
        if model is None:
            verdict = analyse_heuristically(url, self.threshold)
            verdict.features = dict(zip(FEATURE_NAMES, vector))
            return verdict

        import numpy as np

        matrix = np.asarray([vector], dtype="float32")
        probability = float(model.estimator.predict_proba(matrix)[0][1])
        is_phishing = probability >= self.threshold
        return PhishingVerdict(
            url=url, is_phishing=is_phishing, score=probability,
            threshold=self.threshold, signals=signals,
            features=dict(zip(FEATURE_NAMES, vector)),
            model=model.card.name, method="classifier",
            is_synthetic_model=model.card.is_synthetic,
        )

    def analyse_many(self, urls: list[str], *, max_urls: int = 100) -> list[PhishingVerdict]:
        """Score a bounded batch; the cap keeps request cost predictable."""
        if len(urls) > max_urls:
            raise ValidationError(
                f"{len(urls)} URLs submitted; the limit is {max_urls} per request."
            )
        return [self.analyse(url) for url in urls]

    def status(self) -> dict[str, Any]:
        """Capability report. The heuristic is always available."""
        model = self._load()
        info: dict[str, Any] = {
            "available": True,
            "method": "classifier" if model else "heuristic",
            "threshold": self.threshold,
            "network_access": "none (offline lexical analysis only)",
        }
        if model is not None:
            info["model"] = model.describe()
            info["is_synthetic_model"] = model.card.is_synthetic
        else:
            info["model"] = "lexical-heuristic (no trained classifier)"
            info["note"] = (
                "No trained phishing model; serving the explainable lexical "
                "heuristic. Train one with `models train --task phishing`."
            )
        return info


def evaluate_phishing(directory: Path) -> dict[str, Any]:
    """Read back a saved phishing model card."""
    from ..detection.service import evaluate_nids

    return evaluate_nids(directory)


__all__ = [
    "PhishingDetector", "train_phishing", "evaluate_phishing",
    "load_url_dataset", "find_dataset", "DEFAULT_THRESHOLD",
]

