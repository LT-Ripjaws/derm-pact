"""Configuration for the active dermatology evidence-contract study.

Secrets are read from the process environment only.  ``load_env`` is opt-in and
returns a boolean; it never returns or prints a credential.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
PROJECT_ROOT = BASE_DIR.parent


def _value(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Bounded deterministic remediation.
MAX_REMEDIATION_STEPS = int(_value("MEDGUARD_MAX_REMEDIATION_STEPS", "4"))
RETRIEVAL_TOP_K = int(_value("MEDGUARD_TOP_K", "5"))
EXPAND_K = int(_value("MEDGUARD_EXPAND_K", "20"))

# Gateway.  No file is opened at import time.
GATEWAY_API_KEY = os.environ.get("MEDGUARD_API_KEY") or os.environ.get("KILOCODE_API_KEY", "")
GATEWAY_BASE_URL = _value(
    "MEDGUARD_GATEWAY_BASE_URL",
    os.environ.get("KILOCODE_BASE_URL", "https://api.kilo.ai/api/gateway"),
)
GENERATOR_MODEL = _value("MEDGUARD_GENERATOR_MODEL", "qwen/qwen3-vl-32b-instruct")
JUDGE_MODEL = _value("MEDGUARD_JUDGE_MODEL", "openai/gpt-oss-20b")
API_CALL_DELAY_SEC = float(_value("MEDGUARD_API_DELAY", "0.1"))
API_MAX_RETRIES = int(_value("MEDGUARD_API_RETRIES", "3"))

# Provenance-preserving retrieval.
EMBEDDING_MODEL = _value(
    "MEDGUARD_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
RERANKER_MODEL = _value("MEDGUARD_RERANKER_MODEL", "ncbi/MedCPT-Cross-Encoder")
EVIDENCE_BUNDLE_DIR = Path(_value(
    "MEDGUARD_EVIDENCE_BUNDLE",
    str(PROJECT_ROOT / "research" / "evidence" / "dermatology_corpus_v1"),
))
EVIDENCE_RECEIPT_PATH = _value(
    "MEDGUARD_EVIDENCE_RECEIPT", str(EVIDENCE_BUNDLE_DIR / "receipt.json"))
CORPUS_PATH = _value("MEDGUARD_CORPUS", str(EVIDENCE_BUNDLE_DIR / "corpus.jsonl"))
TEXT_INDEX_PATH = _value("MEDGUARD_TEXT_INDEX", str(EVIDENCE_BUNDLE_DIR / "text_dense.faiss"))
TEXT_INDEX_META = _value("MEDGUARD_TEXT_INDEX_META", str(EVIDENCE_BUNDLE_DIR / "corpus.jsonl"))
BM25_INDEX_PATH = _value("MEDGUARD_BM25_INDEX", str(EVIDENCE_BUNDLE_DIR / "bm25_index.json"))
SOURCE_POLICY_PATH = _value(
    "MEDGUARD_SOURCE_POLICY", str(EVIDENCE_BUNDLE_DIR / "source_policy.json"))
CITATION_REGISTRY_PATH = _value(
    "MEDGUARD_CITATION_REGISTRY", str(EVIDENCE_BUNDLE_DIR / "citation_registry.jsonl"))

# Frozen dermatology perception/exemplar branch.
IMAGE_ENCODER_MODEL = _value(
    "MEDGUARD_IMAGE_ENCODER",
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
)
EXEMPLAR_ENCODER_MODEL = _value(
    "MEDGUARD_EXEMPLAR_ENCODER", "hf-hub:redlessone/DermLIP_ViT-B-16")
EXEMPLAR_INDEX_PATH = _value(
    "MEDGUARD_EXEMPLAR_INDEX", str(EVIDENCE_BUNDLE_DIR / "exemplar.faiss"))
EXEMPLAR_META_PATH = _value(
    "MEDGUARD_EXEMPLAR_META", str(EVIDENCE_BUNDLE_DIR / "exemplar_meta.jsonl"))
EXEMPLAR_MALIGNANT_CLF = _value(
    "MEDGUARD_MALIGNANT_CLF", str(DATA_DIR / "index" / "malignant_clf.joblib"))
EXEMPLAR_DIAG_CLF = _value(
    "MEDGUARD_DIAG_CLF", str(DATA_DIR / "index" / "diagnosis_clf.joblib"))
EXEMPLAR_TOP_K = int(_value("MEDGUARD_EXEMPLAR_TOP_K", "10"))
EXEMPLAR_PRIOR_ALPHA = float(_value("MEDGUARD_EXEMPLAR_PRIOR_ALPHA", "0.3"))
EXEMPLAR_CALIBRATION = _value("MEDGUARD_EXEMPLAR_CALIBRATION", "none")

DIAGNOSIS_LABELS = ["melanoma", "nevus", "bcc", "akiec", "bkl", "df", "vasc"]
MALIGNANT_LABELS = {"melanoma", "bcc", "akiec"}
LABEL_FULL_NAMES = {
    "melanoma": "melanoma", "nevus": "melanocytic nevus",
    "bcc": "basal cell carcinoma", "akiec": "actinic keratosis",
    "bkl": "benign keratosis", "df": "dermatofibroma", "vasc": "vascular lesion",
}


def has_api_key() -> bool:
    return bool(GATEWAY_API_KEY)


def load_env(path: str | None = None) -> bool:
    """Opt-in dotenv loading; expose only whether a credential exists."""
    global GATEWAY_API_KEY
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    for candidate in (path, os.environ.get("MEDGUARD_ENV"), str(BASE_DIR / ".env")):
        if candidate and load_dotenv and Path(candidate).exists():
            load_dotenv(candidate, override=False)
            break
    GATEWAY_API_KEY = os.environ.get("MEDGUARD_API_KEY") or os.environ.get("KILOCODE_API_KEY", "")
    return bool(GATEWAY_API_KEY)
