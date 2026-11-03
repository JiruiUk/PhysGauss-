"""Process-local adapter for frozen J1; never enables a formal run."""
from ..prepare_data import ROOT, read, sha
from .core import condition_adam

BASE = ROOT/'experiments/staged_reconstruction/S1_joint_rgb_stability'
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_joint_rgb_stability/v1'


def configure():
    from ..joint_rgb import io, runtime
    # The original J1 source/input receipt remains independently enforced.
    original_receipt = ROOT/'experiments/staged_reconstruction/S1_joint_rgb/v1/source_freeze.json'
    for name, expected in read(original_receipt).items():
        assert sha(ROOT/name) == expected, name
    assert read(BASE/'protocol.json')['Adam_eps'] == 1e-8
    io.BASE, io.REPORT, io.RUNS = BASE, REPORT, RUNS
    io.OUT = ROOT/'output/staged_reconstruction/S1_joint_rgb_stability/v1'
    original_scene = runtime.JointScene

    class EpsilonScene(original_scene):
        def optimizers(self, protocol):
            assert protocol['Adam_eps'] == 1e-8 and protocol['preflight_only']
            return condition_adam(super().optimizers(protocol), enabled=True)

    runtime.JointScene = EpsilonScene
    original_freeze = io.freeze

    def extended_freeze():
        original_freeze()
        manifest = read(REPORT/'source_freeze.json')
        paths = list((ROOT/'variants/staged_reconstruction/joint_rgb_stability').glob('*.py'))
        paths += [ROOT/'tests/test_joint_rgb_stability.py', original_receipt]
        for path in paths:
            manifest[path.relative_to(ROOT).as_posix()] = sha(path)
        io.atomic_json(REPORT/'source_freeze.json', manifest)

    io.freeze = extended_freeze
    return io
