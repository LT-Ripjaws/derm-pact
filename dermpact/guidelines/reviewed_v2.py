"""Approved source-faithful Phase 4 entry point, separate from historical v1."""
from pathlib import Path
from dermpact.guidelines.derivation import derive_case_facts as derive_legacy_facts
from dermpact.guidelines.facts import FactProvenance,TV
from dermpact.guidelines.engine import ApplicabilityEngine
from dermpact.guidelines.rules import RuleCatalogue,ApplicabilityState

ROOT=Path(__file__).resolve().parents[2]
CATALOGUE_PATH=ROOT/'dermpact/data/corpus/eado_melanoma_part1_rules_v2.jsonl'
DERIVATION_POLICY_VERSION='dermpact-phase4-asserted-clinical-facts-v2'

def derive_case_facts(case_id,*,clinician_equivocal=None,lesion_under_assessment=True,**kwargs):
    for name,value in [('clinician_equivocal',clinician_equivocal),('lesion_under_assessment',lesion_under_assessment)]:
        if value is not None and type(value) is not bool:raise TypeError(f'{name} must be bool or None')
    facts=derive_legacy_facts(case_id,**kwargs)
    facts.set('lesion.under_assessment',lesion_under_assessment,FactProvenance('record','host case scope'))
    facts.set('clinician.equivocal',clinician_equivocal,FactProvenance('asserted','explicit clinician finding' if clinician_equivocal is not None else 'not supplied'))
    # Preserve raw count, but a numeric risk example is not an EADO R6 cutoff.
    if kwargs.get('high_total_nevus_count') is None:
        facts.set('patient.high_total_nevus_count',None,FactProvenance('asserted','requires clinician high-count assessment'))
    # Legacy suggestion fields are diagnostic summaries, not clinical assertions.
    for name in ('lesion.melanoma_suspected','lesion.equivocal'):
        if facts.provenance.get(name) and facts.provenance[name].source=='derived':
            facts.provenance[name]=FactProvenance('derived','model/checklist suggestion only; not consumed by v2 clinical rules')
    return facts

class ReviewedApplicabilityEngine(ApplicabilityEngine):
    def __init__(self,catalogue=None,**kwargs):
        if catalogue is None:
            import json,hashlib
            manifest=ROOT/'research/manifests/dermpact_phase4/rule_catalog_v2/catalog_manifest.json'
            if hashlib.sha256(manifest.read_bytes()).hexdigest()!=Path(str(manifest)+'.sha256').read_text().strip():
                raise ValueError('reviewed catalogue manifest integrity failure')
            binding=json.loads(manifest.read_text())
            for relative,expected in binding['source_bindings'].items():
                path=(ROOT/relative).resolve()
                if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    raise ValueError('reviewed rule source drift')
            catalogue=RuleCatalogue.load_jsonl(CATALOGUE_PATH,expected_sha256=binding['catalogue_sha256'])
        if catalogue.catalogue_version!='v2':raise ValueError('reviewed engine requires v2 catalogue')
        super().__init__(catalogue,**kwargs)

    def evaluate_rule(self,rule,facts):
        working=facts.copy()
        for name in rule.required_facts:
            if name.startswith('clinician.') or name in ('patient.high_risk_status','patient.high_total_nevus_count','site.rcm_available'):
                provenance=working.provenance.get(name)
                if provenance is None or provenance.source!='asserted':
                    working.values[name]=TV.UNKNOWN
                    working.provenance[name]=FactProvenance('engine','explicit assertion required for clinical rule')
        result=super().evaluate_rule(rule,working)
        if result.state is ApplicabilityState.UNKNOWN:
            from dataclasses import replace
            questions={
                'clinician.equivocal':'Is this lesion clinically or dermatoscopically equivocal (yes/no)?',
                'patient.high_total_nevus_count':'Do you assess this patient as having a high total nevus count (yes/no)?',
                'lesion.under_assessment':'Is a pigmented or non-pigmented skin lesion being assessed (yes/no)?',
            }
            if result.decisive_missing_fact in questions:result=replace(result,question=questions[result.decisive_missing_fact])
        return result
