"""Every version string and release anchor for the current DermPact configuration."""
from __future__ import annotations

RECORD_SCHEMA_VERSION = "dermpact-phase5-case-record-v2"
ENGINE_POLICY_VERSION = "dermpact-phase5-engine-policy-v2"
GATE_POLICY_VERSION = "dermpact-phase5-gate-v2"
RENDERER_VERSION = "dermpact-phase5-renderer-v2"
LEDGER_VERSION = "dermpact-phase5-ledger-v2"
EPISODE_VERSION = "dermpact-phase5-review-episode-v1"
CLARIFICATION_VERSION = "dermpact-phase5-decision-relevance-v1"
XAI_PROTOCOL_VERSION = "dermpact-phase5-measured-xai-v1"
MEMORY_PROTOCOL_VERSION = "dermpact-phase6-comparison-memory-v2"
WORKFLOW_CONFIG_VERSION = "dermpact-phase6-workflow-config-v1"
PROMPT_VERSION = "dermpact-phase7-agent-prompt-v2-grounded"
PROTOTYPE_GRAPH_VERSION = "dermpact-phase6-prototype-graph-v1"
PREPROCESSING_VERSION = "dermpact-phase2-5-selected-inference-v1"

PRIMARY_MODEL_ID = "phase2_5_stage2_cefix_v1"
PRIMARY_LIVE_AGENT_MODEL = "qwen/qwen3-vl-32b-instruct"
PHASE3_BUNDLE_VERSION = "dermpact-phase3-reliability-bundle-v3"

# Repository-relative release anchors (hash-verified by their own loaders).
RELIABILITY_BUNDLE = "research/manifests/dermpact_phase3/reliability_bundle_v3.json"
SELECTION_MANIFEST = "research/manifests/dermpact_phase2_final/selection_2026-09-06.json"
CATALOGUE_MANIFEST = "research/manifests/dermpact_phase4/rule_catalog_v2/catalog_manifest.json"
CATALOGUE_PATH = "dermpact/data/corpus/eado_melanoma_part1_rules_v2.jsonl"

# WP-M: comparison memory embedding identity.
MEMORY_EMBEDDING_MODEL = "redlessone/DermLIP_PanDerm-base-w-PubMed-256 sentinel 224px"

__all__ = [name for name in dir() if name.isupper()]
