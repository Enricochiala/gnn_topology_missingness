"""Build RT on Linux/macOS, with optional OpenMP acceleration."""
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import tempfile


def build(openmp=True):
    root = Path(__file__).resolve().parent
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    flags = []
    if openmp:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'probe.cpp'
            source.write_text('#include <omp.h>\nint main(){return omp_get_max_threads()<1;}\n')
            probe = subprocess.run([*compiler, '-fopenmp', str(source), '-o', str(Path(directory)/'probe')],
                                   capture_output=True)
            if probe.returncode == 0:
                flags = ['-fopenmp']
    pending = root / 'shared_seed_growth.pending.so'
    subprocess.run([*compiler, '-O3', '-std=c++17', '-fPIC', '-shared', *flags,
                    'shared_seed_growth.cpp', '-o', str(pending)], cwd=root, check=True)
    pending.replace(root / 'shared_seed_growth.so')
    print('Built shared_seed_growth.so (' + ('OpenMP' if flags else 'serial') + ')')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-openmp', action='store_true')
    build(openmp=not parser.parse_args().no_openmp)
