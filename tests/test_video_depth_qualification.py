import pytest
from variants.staged_reconstruction.video_depth.core import ANCHORS,windows,check


def test_windows_use_only_unique_training_ids_and_preserve_order():
    ids=sorted(set(range(1,201,3))|set(ANCHORS)); clips=windows(ids)
    assert len(clips)==6
    for anchor,clip in clips.items():
        assert clip==sorted(clip) and len(clip)==len(set(clip))==12
        assert anchor in clip and set(clip)<=set(ids)
    with pytest.raises(ValueError): windows(ids+[2])


def test_each_quality_failure_rejects_instead_of_averaging_away():
    row=dict(median=.04,p90=.14,dynamic_extrapolation_fraction=.09,static_support=1001)
    assert check(row)
    for key,value in [('median',.051),('p90',.151),('dynamic_extrapolation_fraction',.101),('static_support',999)]:
        assert not check(dict(row,**{key:value}))
