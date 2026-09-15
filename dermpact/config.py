"""Configuration for the active dermatology evidence-contract study."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
PROJECT_ROOT = BASE_DIR.parent


def _value(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _value_alias(primary: str, alias: str, default: str) -> str:
    """Canonical DERMPACT_* name with backward-compatible project alias."""
    return os.environ.get(primary) or os.environ.get(alias) or default


# Bounded deterministic remediation.
MAX_REMEDIATION_STEPS = int(_value("DERMPACT_MAX_REMEDIATION_STEPS", "4"))
RETRIEVAL_TOP_K = int(_value("DERMPACT_TOP_K", "5"))
EXPAND_K = int(_value("DERMPACT_EXPAND_K", "20"))

# Gateway.  No file is opened at import time.
GATEWAY_API_KEY = os.environ.get("DERMPACT_API_KEY") or os.environ.get("KILOCODE_API_KEY", "")
GATEWAY_BASE_URL = _value(
    "DERMPACT_GATEWAY_BASE_URL",
    os.environ.get("KILOCODE_BASE_URL", "https://api.kilo.ai/api/gateway"),
)
GENERATOR_MODEL = _value_alias("DERMPACT_GENERATOR_MODEL", "KILO_GENERATOR_MODEL", "qwen/qwen3-vl-32b-instruct")
JUDGE_MODEL = _value_alias("DERMPACT_JUDGE_MODEL", "JUDGE_MODEL", "openai/gpt-oss-20b")
API_CALL_DELAY_SEC = float(_value_alias("DERMPACT_API_DELAY", "API_CALL_DELAY_SEC", "0.1"))
API_MAX_RETRIES = int(_value("DERMPACT_API_RETRIES", "3"))

# Provenance-preserving retrieval.
EMBEDDING_MODEL = _value(
    "DERMPACT_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
RERANKER_MODEL = _value("DERMPACT_RERANKER_MODEL", "ncbi/MedCPT-Cross-Encoder")
EVIDENCE_BUNDLE_DIR = Path(_value(
    "DERMPACT_EVIDENCE_BUNDLE",
    str(PROJECT_ROOT / "research" / "evidence" / "dermatology_corpus_v1"),
))
EVIDENCE_RECEIPT_PATH = _value(
    "DERMPACT_EVIDENCE_RECEIPT", str(EVIDENCE_BUNDLE_DIR / "receipt.json"))
CORPUS_PATH = _value("DERMPACT_CORPUS", str(EVIDENCE_BUNDLE_DIR / "corpus.jsonl"))
TEXT_INDEX_PATH = _value("DERMPACT_TEXT_INDEX", str(EVIDENCE_BUNDLE_DIR / "text_dense.faiss"))
TEXT_INDEX_META = _value("DERMPACT_TEXT_INDEX_META", str(EVIDENCE_BUNDLE_DIR / "corpus.jsonl"))
BM25_INDEX_PATH = _value("DERMPACT_BM25_INDEX", str(EVIDENCE_BUNDLE_DIR / "bm25_index.json"))
SOURCE_POLICY_PATH = _value(
    "DERMPACT_SOURCE_POLICY", str(EVIDENCE_BUNDLE_DIR / "source_policy.json"))
CITATION_REGISTRY_PATH = _value(
    "DERMPACT_CITATION_REGISTRY", str(EVIDENCE_BUNDLE_DIR / "citation_registry.jsonl"))

# Frozen dermatology perception/exemplar branch.
IMAGE_ENCODER_MODEL = _value(
    "DERMPACT_IMAGE_ENCODER",
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
)
EXEMPLAR_ENCODER_MODEL = _value(
    "DERMPACT_EXEMPLAR_ENCODER", "hf-hub:redlessone/DermLIP_ViT-B-16")
EXEMPLAR_INDEX_PATH = _value(
    "DERMPACT_EXEMPLAR_INDEX", str(EVIDENCE_BUNDLE_DIR / "exemplar.faiss"))
EXEMPLAR_META_PATH = _value(
    "DERMPACT_EXEMPLAR_META", str(EVIDENCE_BUNDLE_DIR / "exemplar_meta.jsonl"))
EXEMPLAR_MALIGNANT_CLF = _value(
    "DERMPACT_MALIGNANT_CLF", str(DATA_DIR / "index" / "malignant_clf.joblib"))
EXEMPLAR_DIAG_CLF = _value(
    "DERMPACT_DIAG_CLF", str(DATA_DIR / "index" / "diagnosis_clf.joblib"))
EXEMPLAR_TOP_K = int(_value("DERMPACT_EXEMPLAR_TOP_K", "10"))
EXEMPLAR_PRIOR_ALPHA = float(_value("DERMPACT_EXEMPLAR_PRIOR_ALPHA", "0.3"))
EXEMPLAR_CALIBRATION = _value("DERMPACT_EXEMPLAR_CALIBRATION", "none")

DIAGNOSIS_LABELS = ["melanoma", "nevus", "bcc", "akiec", "bkl", "df", "vasc"]
MALIGNANT_LABELS = {"melanoma", "bcc", "akiec"}
LABEL_FULL_NAMES = {
    "melanoma": "melanoma", "nevus": "melanocytic nevus",
    "bcc": "basal cell carcinoma", "akiec": "actinic keratosis",
    "bkl": "benign keratosis", "df": "dermatofibroma", "vasc": "vascular lesion",
}


def has_api_key() -> bool:
    return bool(GATEWAY_API_KEY)


def _load_env_file_without_dependency(path: Path) -> None:
    """Minimal dotenv reader used when python-dotenv is unavailable."""
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or not (name[0].isalpha() or name[0] == "_") or not all(
            char.isalnum() or char == "_" for char in name
        ):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def load_env(path: str | None = None) -> bool:
    """Opt-in dotenv loading; expose only whether a credential exists."""
    global GATEWAY_API_KEY, GATEWAY_BASE_URL, GENERATOR_MODEL, JUDGE_MODEL, API_CALL_DELAY_SEC
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    for candidate in (path, os.environ.get("DERMPACT_ENV"), str(PROJECT_ROOT / ".env"), str(BASE_DIR / ".env")):
        if candidate and Path(candidate).exists():
            if load_dotenv is not None:
                load_dotenv(candidate, override=False)
            else:
                _load_env_file_without_dependency(Path(candidate))
            if path or os.environ.get("DERMPACT_ENV"):
                break
    GATEWAY_API_KEY = os.environ.get("DERMPACT_API_KEY") or os.environ.get("KILOCODE_API_KEY", "")
    GATEWAY_BASE_URL = os.environ.get("DERMPACT_GATEWAY_BASE_URL") or os.environ.get(
        "KILOCODE_BASE_URL", "https://api.kilo.ai/api/gateway"
    )
    GENERATOR_MODEL = _value_alias("DERMPACT_GENERATOR_MODEL", "KILO_GENERATOR_MODEL", "qwen/qwen3-vl-32b-instruct")
    JUDGE_MODEL = _value_alias("DERMPACT_JUDGE_MODEL", "JUDGE_MODEL", "openai/gpt-oss-20b")
    API_CALL_DELAY_SEC = float(_value_alias("DERMPACT_API_DELAY", "API_CALL_DELAY_SEC", "0.1"))
    return bool(GATEWAY_API_KEY)
