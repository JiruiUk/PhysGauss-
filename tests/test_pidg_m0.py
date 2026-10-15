import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_pidg_m0 import audit


def test_completed_original_baselines_are_reusable_without_deleted_variants():
    root=Path(__file__).resolve().parents[1]
    if not (root/'runs/pidg_official/rubber_duck/results.json').exists():
        pytest.skip('Local official baselines unavailable')
    report=audit()
    assert report['status']=='passed_reused_baselines' and report['retrained'] is False
    assert len(report['baselines'])==5
    assert all(row['protocol']['iterations']==50000 for row in report['baselines'])
    assert len(report['config_inventory']['dnerf'])==8
    assert len(report['config_inventory']['hypernerf'])==11
