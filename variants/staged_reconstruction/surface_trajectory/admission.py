"""Evidence gates are separate from model/optimizer preflight and user authorization."""


def evidence_gates(bindings, split, visual_review):
    counts = [sum(b['time_band'] == band for b in bindings) for band in range(3)]
    gates = dict(at_least_50_fixed_support_tracks=len(bindings) >= 50,
        at_least_10_per_time_band=all(n >= 10 for n in counts),
        at_least_30_supervised_tracks=len(split.get('training_track_ids', [])) >= 30,
        at_least_20_holdout_tracks=len(split.get('holdout_track_ids', [])) >= 20,
        at_least_20_disjoint_holdout_groups=len(split.get('holdout_groups', [])) >= 20,
        grouped_split_admitted=bool(split['admitted']),
        actual_visual_review_completed=bool(visual_review['actual_visual_inspection_completed']),
        stable_surface_feature_evidence_sufficient=bool(visual_review['surface_feature_evidence_sufficient']))
    return dict(evidence_admitted=all(gates.values()), gates=gates,
        failed_gates=[k for k, value in gates.items() if not value],
        fixed_support_tracks=len(bindings), counts_by_time_band=counts,
        training_preflight_completed=False, formal_training_admitted=False,
        optimizer_updates=0, status='evidence_gate_passed_preflight_pending' if all(gates.values()) else 'evidence_gate_failed',
        reason='Do not weaken observation or grouping thresholds to start T0/T1')
