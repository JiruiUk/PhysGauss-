from distutils.command.build import build
import os
from torch.utils.cpp_extension import load
from pathlib import Path

Path('./tmp_build/').mkdir(parents=True, exist_ok=True)

_src_path = os.environ.get('PIDG_HASH_SOURCE_DIR', os.path.dirname(os.path.abspath(__file__)))

_backend = load(name='_hash_encoder',
                extra_cflags=['/O2', '/std:c++17'] if os.name == 'nt' else ['-O3', '-std=c++17'],
                extra_cuda_cflags=[
                    '-O3', '-std=c++17', '-allow-unsupported-compiler',
                    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
                ],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'hashencoder.cu',
                    'bindings.cpp',
                ]],
                build_directory=os.environ.get('PIDG_HASH_BUILD_DIR', './tmp_build/'),
                verbose=True,
                )

__all__ = ['_backend']