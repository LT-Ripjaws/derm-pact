"""Source-based truth table for the approved clinical-assertion boundary."""
import itertools
from datetime import date
import pytest
from dermpact.guidelines.reviewed_v2 import ReviewedApplicabilityEngine,derive_case_facts
from dermpact.guidelines.facts import FactProvenance
from dermpact.guidelines.engine import EntityMismatchError

@pytest.fixture(scope='module')
def engine():return ReviewedApplicabilityEngine()

def state(engine,rule,**kwargs):
    facts=derive_case_facts('example',evaluation_date=date(2026,9,6),**kwargs)
    return engine.evaluate_case(facts).result_for('eado-mel-p1-2024:'+rule)

@pytest.mark.parametrize('model',['melanoma','nevus','abstained',None])
@pytest.mark.parametrize('image',[True,False,None])
@pytest.mark.parametrize('assertion',[True,False,None])
def test_r3_source_truth_independent_of_model_and_image(engine,model,image,assertion):
    result=state(engine,'R3',provisional_class=model,dermoscopic_present=image,clinician_melanoma_suspected=assertion)
    expected='applicability_unknown' if assertion is None else ('applicable' if assertion else 'not_applicable')
    assert result.state.value==expected
    if assertion is None:assert result.decisive_missing_fact=='clinician.melanoma_suspected'

@pytest.mark.parametrize('equivocal,available',list(itertools.product([True,False,None],repeat=2)))
def test_r7_three_valued_source_truth(engine,equivocal,available):
    result=state(engine,'R7',clinician_equivocal=equivocal,rcm_available=available,dermoscopic_present=False)
    expected='not_applicable' if False in (equivocal,available) else ('applicability_unknown' if None in (equivocal,available) else 'applicable')
    assert result.state.value==expected

def test_r4_not_circular_and_quality_not_attributed_to_guideline(engine):
    result=state(engine,'R4',dermoscopic_present=False,iqa_state='fail')
    assert result.state.value=='applicable'
    assert 'reacquire' not in result.action_text
    assert state(engine,'R4',lesion_under_assessment=None).state.value=='applicability_unknown'

def test_wrong_instrument_does_not_erase_clinical_assertion(engine):
    assert state(engine,'R3',clinician_melanoma_suspected=True,instrument_id='nice_ng12_weighted_7point').state.value=='applicable'

def test_count_example_does_not_supply_clinician_assessment(engine):
    result=state(engine,'R6',high_risk_status=True,total_nevus_count=250)
    assert result.state.value=='applicability_unknown'
    assert result.decisive_missing_fact=='patient.high_total_nevus_count'
    assert '100' not in result.question
    assert state(engine,'R6',high_risk_status=True,total_nevus_count=250,high_total_nevus_count=True).state.value=='applicable'

@pytest.mark.parametrize('provenance',['derived','record','observed','engine'])
def test_model_cannot_supply_clinician_truth(engine,provenance):
    facts=derive_case_facts('example',evaluation_date=date(2026,9,6))
    facts.set('clinician.melanoma_suspected',True,FactProvenance(provenance,'untrusted model claim'))
    result=engine.evaluate_case(facts).result_for('eado-mel-p1-2024:R3')
    assert result.state.value=='applicability_unknown'
    assert facts.is_known('clinician.melanoma_suspected')  # engine did not rewrite input

def test_source_entity_jurisdiction_and_expiry(engine):
    facts=derive_case_facts('example',evaluation_date=date(2027,1,1),clinician_melanoma_suspected=True)
    assert engine.evaluate_case(facts).result_for('eado-mel-p1-2024:R3').state.value=='no_catalogue_coverage'
    with pytest.raises(EntityMismatchError):engine.evaluate_case(facts,expected_case_id='someone-else')
    facts=derive_case_facts('example',evaluation_date=date(2026,9,6),jurisdiction='US',clinician_melanoma_suspected=True)
    assert engine.evaluate_case(facts).result_for('eado-mel-p1-2024:R3').state.value=='no_catalogue_coverage'

def test_original_sources_unchanged(engine):
    from dermpact.guidelines.rules import RuleCatalogue
    from dermpact.guidelines.reviewed_v2 import ROOT
    old=RuleCatalogue.load_jsonl(ROOT/'dermpact/data/corpus/eado_melanoma_part1_rules_v1.jsonl',expected_sha256='927cfd2c8d06afbff6f7ffde4d7e205e8a472b4d2dcd85efc6a30461e53f0c61')
    assert all(r.source==old.get(r.rule_id).source for r in engine.catalogue.records)
    assert len(engine.catalogue.records)==14 and len(engine.catalogue.executable)==5
