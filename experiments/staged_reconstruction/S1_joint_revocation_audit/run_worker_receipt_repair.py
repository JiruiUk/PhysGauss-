"""Evaluation-only receipt repair: ignore nonexistent Torch virtual source paths."""
from pathlib import Path
import sys
from variants.staged_reconstruction.prepare_data import ROOT, sha
from variants.staged_reconstruction.joint_revocation_audit import worker


def source_files(modules):
    result = {}
    for module in modules:
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and path.suffix == '.py' and 'envs' not in path.parts:
            result[path.relative_to(ROOT).as_posix()] = sha(path)
    return result


def runtime_sources():
    return source_files(list(sys.modules.values()))


if __name__ == '__main__':
    worker.runtime_sources = runtime_sources
    worker.main()
