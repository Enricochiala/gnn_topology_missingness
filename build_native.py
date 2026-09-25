"""Build the RT shared-seed accelerator (Linux, C++17, OpenMP)."""
from pathlib import Path
import os,subprocess

def build():
    root=Path(__file__).resolve().parent
    compiler=os.environ.get('CXX','g++')
    subprocess.run([compiler,'-O3','-std=c++17','-fPIC','-shared','-fopenmp',
                    'shared_seed_growth.cpp','-o','shared_seed_growth.so'],cwd=root,check=True)
    print('Built shared_seed_growth.so')

if __name__=='__main__':build()
