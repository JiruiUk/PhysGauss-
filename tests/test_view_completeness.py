import numpy as np
import pytest
from variants.view_completeness.metrics import evaluate, observed_coverage, shape_proxies, summarize


def solid():
    x = np.zeros((64, 80))
    x[12:52, 20:60] = 1
    return x


def test_disabled_entry_does_not_read_images_or_change_baseline():
    assert evaluate(object(), object()) is None


def test_unobserved_view_does_not_invent_true_completeness():
    result = evaluate(solid(), enabled=True)
    assert result['observed_coverage'] is None
    assert result['full_object_completeness_score'] is None
    assert not result['independent_alpha_target_available']
    assert not result['shape_proxies']['physical_completeness_verified']


def test_empty_render_cannot_rank_as_perfect_integrity():
    record = evaluate(np.zeros((64, 80)), enabled=True)
    assert not record['shape_proxies']['defined']
    assert record['shape_proxies']['thin_support_fraction'] is None
    rows = [dict(frame=1, **record)]
    assert summarize(rows)['undefined_shape_samples'] == 1
    assert summarize(rows)['thin_support_fraction'] is None


def test_small_detached_artifact_and_thin_spur_are_detected():
    base = solid()
    modified = base.copy()
    modified[2:4, 2:4] = 1
    modified[28, 60:77] = 1
    old, new = shape_proxies(base), shape_proxies(modified)
    assert new['small_island_fraction'] > old['small_island_fraction']
    assert new['thin_support_fraction'] > old['thin_support_fraction']


def test_real_hole_is_a_descriptor_not_evidence_of_bad_geometry():
    donut = solid()
    donut[25:35, 32:44] = 0
    record = evaluate(donut, donut, enabled=True)
    assert record['shape_proxies']['enclosed_hole_fraction'] > 0
    assert record['observed_coverage']['mask_IoU'] == 1
    assert record['observed_coverage']['foreground_missing_fraction'] == 0
    assert record['full_object_completeness_score'] is None


def test_legitimate_separate_objects_are_not_forced_into_one_component():
    target = solid()
    target[20:40, 4:12] = 1
    record = evaluate(target, target, enabled=True)
    assert record['shape_proxies']['component_count'] == 2
    assert record['observed_coverage']['mask_IoU'] == 1


def test_missing_true_surface_and_background_fill_have_opposite_errors():
    target = solid()
    low = target*.3
    observed = observed_coverage(low, target)
    assert observed['foreground_missing_fraction'] == 1
    assert observed['background_spill_fraction'] == 0
    inflated = observed_coverage(np.ones_like(target), target)
    assert inflated['foreground_missing_fraction'] == 0
    assert inflated['background_spill_fraction'] == 1


def test_frame_clipping_is_recorded_without_creating_a_hole():
    target = np.zeros((64, 80))
    target[10:50, :40] = 1
    record = shape_proxies(target)
    assert record['touches_image_border']
    assert record['enclosed_hole_fraction'] == 0


def test_metrics_do_not_repair_or_mutate_input_images():
    target = solid()
    target[20:30, 30:40] = 0
    before = target.copy()
    evaluate(target, enabled=True)
    assert np.array_equal(target, before)


@pytest.mark.parametrize('bad', [np.full((8, 8), np.nan), np.full((8, 8), -1), np.ones((8, 8, 3))])
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        evaluate(bad, enabled=True)
