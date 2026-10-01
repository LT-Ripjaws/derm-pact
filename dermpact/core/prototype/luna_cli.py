"""CLI routing for fixed-model, persistent Luna clarification."""
import json
from dermpact.core.agent.redesign.evidence import EvidenceError
from dermpact.core.agent.redesign.provider import GenerationError
from .host import HostServices, HostBuildError, DEFAULT_STATE_DIR
from dermpact.core.store import StoreErrorV2
from .inventory import DEFAULT_INVENTORY_DIR
from .luna_clarification import execute


def register(subparsers):
    parser = subparsers.add_parser("luna-clarify", help="start/resume the fixed Luna clarification workflow")
    parser.add_argument("action", choices=["start", "reply", "confirm", "cancel", "status"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--inventory-dir", default=str(DEFAULT_INVENTORY_DIR))
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--rule", dest="rule_id")
    parser.add_argument("--workflow", choices=["clarification", "amendment"], default="clarification")
    parser.add_argument("--request")
    parser.add_argument("--actor", default="clinician")
    parser.add_argument("--text")
    parser.add_argument("--proposal-id")
    parser.add_argument("--preview-sha256")


def run(args):
    host = None
    try:
        host = HostServices.build(args.state_dir, inventory_dir=args.inventory_dir, mode="live", inference="cached")
        result = execute(host, run_id=args.run_id, action=args.action, case_id=args.case_id,
            rule_id=args.rule_id, request=args.request, actor=args.actor, text=args.text,
            proposal_id=args.proposal_id, preview_sha256=args.preview_sha256, workflow=args.workflow)
        print(json.dumps(result, ensure_ascii=True))
        if result.get("resume_error"):
            return 3
        outcome = result["result"].get("outcome")
        return 5 if outcome == "cancelled" else 4 if outcome in {"failed", "safe_fallback", "incomplete"} else 0
    except (EvidenceError, StoreErrorV2) as exc:
        print(json.dumps({"outcome": "rejected", "reason": str(exc)})); return 3
    except (GenerationError, HostBuildError) as exc:
        print(json.dumps({"outcome": "unavailable", "reason": str(exc)})); return 4
    except ValueError as exc:
        print(json.dumps({"outcome": "invalid_request", "reason": str(exc)})); return 2
    finally:
        if host is not None:
            host.close()
