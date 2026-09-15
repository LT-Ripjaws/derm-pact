"""Independent reference transcription of reviewed EADO R3-R7 predicates."""
def reference_states(*,suspected,equivocal,high_risk,high_count,rcm_available,lesion_under_assessment,
                     jurisdiction='EU',date_iso='2026-09-06'):
    conditions={'R3':(suspected,), 'R4':(lesion_under_assessment,), 'R5':(high_risk,),
                'R6':(high_risk,high_count), 'R7':(equivocal,rcm_available)}
    if jurisdiction!='EU' or not '2024-11-28'<=date_iso<='2026-12-31':
        return {key:'no_catalogue_coverage' for key in conditions}
    results={}
    for key,values in conditions.items():
        if any(v is not None and type(v) is not bool for v in values):raise ValueError('explicit boolean or unknown required')
        results[key]=('not_applicable' if any(v is False for v in values) else
                      'applicability_unknown' if any(v is None for v in values) else 'applicable')
    return results
